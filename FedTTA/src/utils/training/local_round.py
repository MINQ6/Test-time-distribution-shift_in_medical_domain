from typing import Any, Dict

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

from data.loader import get_client_dataloader
from models.dual_lora_model import setup_model_with_dual_lora
from utils.training.common import clone_state_dict_to_cpu, load_hf_token, logger, set_seed


def _sanitize_tensor(tensor: torch.Tensor, *, label: str) -> tuple[torch.Tensor, bool]:
    sanitized = tensor.detach().clone().cpu()
    is_finite = torch.isfinite(sanitized)
    if bool(is_finite.all()):
        return sanitized, False

    invalid_count = int((~is_finite).sum().item())
    logger.warning(
        "[Non-finite Guard] %s contained %d non-finite values. Replacing them with 0.0 for saved payloads.",
        label,
        invalid_count,
    )
    sanitized = torch.nan_to_num(sanitized, nan=0.0, posinf=0.0, neginf=0.0)
    return sanitized, True


def _sanitize_upload_payload(upload_payload: Dict[str, Any], *, client_id: str, round_idx: int, epoch_idx: int | None = None) -> Dict[str, Any]:
    sanitized_global_lora: Dict[str, Dict[str, torch.Tensor]] = {}
    had_non_finite = False

    for module_name, module_payload in upload_payload.get("global_lora", {}).items():
        sanitized_a, bad_a = _sanitize_tensor(
            module_payload["A"],
            label=f"round={round_idx} client={client_id} epoch={epoch_idx or 'final'} module={module_name} A",
        )
        sanitized_b, bad_b = _sanitize_tensor(
            module_payload["B"],
            label=f"round={round_idx} client={client_id} epoch={epoch_idx or 'final'} module={module_name} B",
        )
        had_non_finite = had_non_finite or bad_a or bad_b
        sanitized_global_lora[module_name] = {
            "A": sanitized_a,
            "B": sanitized_b,
        }

    return {
        "global_lora": sanitized_global_lora,
        "had_non_finite": had_non_finite,
    }


def _align_local_state_from_global_state(global_lora_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    aligned_local_state: Dict[str, torch.Tensor] = {}
    for key, value in global_lora_state.items():
        if ".global_lora." not in key:
            continue
        local_key = key.replace(".global_lora.", ".local_lora.")
        aligned_local_state[local_key] = value.detach().cpu().clone()
    return aligned_local_state


def _zero_global_state(global_lora_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        key: torch.zeros_like(value).cpu()
        for key, value in global_lora_state.items()
        if ".global_lora." in key
    }


def _run_stage(
    *,
    model,
    dataloader,
    device: torch.device,
    optimizer: AdamW,
    trainable_params: list[torch.nn.Parameter],
    round_idx: int,
    client_id: str,
    stage_name: str,
    epochs: int,
    gradient_accumulation_steps: int,
    epoch_end_callback=None,
) -> list[float]:
    epoch_losses: list[float] = []
    for epoch in range(epochs):
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(dataloader):
            inputs = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**inputs)
            loss = outputs.loss
            if not torch.isfinite(loss):
                logger.warning(
                    "Non-finite loss @ round=%s client=%s stage=%s epoch=%d step=%d -> skip step",
                    round_idx, client_id, stage_name, epoch + 1, step,
                )
                optimizer.zero_grad()
                continue
            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()

            nonfinite_grad = any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in trainable_params
            )
            should_step = (
                (step + 1) % gradient_accumulation_steps == 0
                or (step + 1) == len(dataloader)
            )
            if nonfinite_grad:
                logger.warning(
                    "Non-finite gradient @ round=%s client=%s stage=%s epoch=%d step=%d -> skip accumulation window",
                    round_idx, client_id, stage_name, epoch + 1, step,
                )
                optimizer.zero_grad()
            elif should_step:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item()

            if step % 10 == 0:
                logger.info(
                    f"[Round {round_idx}] {client_id} | {stage_name} | Epoch {epoch+1}/{epochs} | "
                    f"Step {step}/{len(dataloader)} | Loss {loss.item():.4f} | "
                    f"Accum {(step % gradient_accumulation_steps) + 1}/{gradient_accumulation_steps}"
                )

        avg_loss = total_loss / max(1, len(dataloader))
        epoch_losses.append(avg_loss)
        logger.info(
            f"[Round {round_idx}] {client_id} | {stage_name} epoch {epoch+1} done | avg_loss={avg_loss:.4f}"
        )
        if epoch_end_callback is not None:
            epoch_end_callback(epoch + 1, avg_loss)
    return epoch_losses


def _individual_upload_payload(model) -> Dict[str, Dict[str, torch.Tensor]]:
    """FedALT: client uploads its Individual(local) LoRA (A,B) per module.

    Reuses the 'global_lora' payload key so the existing parallel/sequential
    plumbing and the server-side stacking helper can consume it unchanged.
    The server then computes each client's RoW = leave-one-out mean over these.
    """
    individual_payload: Dict[str, Dict[str, torch.Tensor]] = {}
    for module_name, wrapped in model.dual_lora_adapter._wrapped_modules.items():
        individual_payload[module_name] = {
            "A": wrapped.local_lora.lora_A.weight.detach().clone().cpu(),
            "B": wrapped.local_lora.lora_B.weight.detach().clone().cpu(),
        }
    return individual_payload


def _run_fedalt_round(
    *,
    model,
    dataloader,
    device: torch.device,
    round_idx: int,
    client_id: str,
    epochs: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
) -> Dict[str, Any]:
    """FedALT local update: train Individual(local) + mixer with RoW(global) frozen,
    forwarding through the per-token mixer gate."""
    model.dual_lora_adapter.set_use_mixer(True)
    model.dual_lora_adapter.configure_fedalt_training()
    trainable_params = [
        param
        for param in (
            list(model.dual_lora_adapter.local_parameters())
            + list(model.dual_lora_adapter.mixer_parameters())
        )
        if param.requires_grad
    ]
    optimizer = AdamW(trainable_params, lr=learning_rate)
    logger.info(
        "[Round %s] %s FedALT stage | train Individual+mixer, RoW frozen, mixer forward",
        round_idx, client_id,
    )
    epoch_losses = _run_stage(
        model=model,
        dataloader=dataloader,
        device=device,
        optimizer=optimizer,
        trainable_params=trainable_params,
        round_idx=round_idx,
        client_id=client_id,
        stage_name="FedALT",
        epochs=epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    individual_payload = _individual_upload_payload(model)
    sanitized = _sanitize_upload_payload({"global_lora": individual_payload}, client_id=client_id, round_idx=round_idx)
    upload_payload = {
        "global_lora": sanitized["global_lora"],  # carries Individual(local) matrices
        "had_non_finite": sanitized["had_non_finite"],
    }
    local_state_out = clone_state_dict_to_cpu(model.dual_lora_adapter.state_dict())
    mixer_state_out = clone_state_dict_to_cpu(model.dual_lora_adapter.mixer_state_dict())
    alpha_indiv_mean = model.dual_lora_adapter.alpha_indiv_mean()

    del model
    del optimizer
    torch.cuda.empty_cache()

    return {
        "local_state": local_state_out,
        "mixer_state": mixer_state_out,
        "upload_payload": upload_payload,
        "epoch_global_uploads": [],
        "metrics": {
            "fedalt_epoch_losses": epoch_losses,
            "fedalt_final_loss": epoch_losses[-1] if epoch_losses else None,
            "alpha_indiv_mean": alpha_indiv_mean,
        },
    }


def run_local_federated_round(
    model_id: str,
    client_id: str,
    dataset_name: str,
    round_idx: int,
    local_state: Dict[str, torch.Tensor] | None,
    global_lora_state: Dict[str, torch.Tensor] | None,
    epochs: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    num_samples: int,
    max_length: int,
    seed: int,
    rank: int,
    lora_alpha: int,
    lora_dropout: float,
    learning_rate: float,
    train_on_inputs: bool,
    lora_target_modules: list[str],
    stage1_only: bool = False,
    train_mode: str = "feddpa",
    mixer_state: Dict[str, torch.Tensor] | None = None,
) -> Dict[str, Any]:
    """한 client의 한 federated round local update를 수행한다.

    train_mode:
      - "feddpa": 기존 동작 (global stage1 + optional local stage2).
      - "fedalt": Individual(local)+mixer만 학습, RoW(global) frozen, mixer forward 사용.
                  Individual만 업로드(upload_payload의 'global_lora' 키에 individual을 담아
                  서버가 leave-one-out RoW를 계산), mixer는 결과에 따로 반환.
    """
    hf_token = load_hf_token()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available inside the training process. "
            "Stopping early because this experiment is expected to run on GPU."
        )
    device = torch.device("cuda")
    set_seed(seed + round_idx)

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dataloader = get_client_dataloader(
        client_id=client_id,
        tokenizer=tokenizer,
        batch_size=micro_batch_size,
        num_samples=num_samples,
        dataset_name=dataset_name,
        max_length=max_length,
        train_on_inputs=train_on_inputs,
    )

    model = setup_model_with_dual_lora(
        model_id,
        hf_token,
        rank=rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        target_modules=lora_target_modules,
    )

    if local_state is not None:
        model.dual_lora_adapter.load_state_dict(local_state, strict=True)

    model.dual_lora_adapter.load_global_lora(global_lora_state)
    if train_mode == "fedalt" and mixer_state is not None:
        model.dual_lora_adapter.load_mixer_state_dict(mixer_state)
    model.train()
    epoch_global_uploads: list[Dict[str, Any]] = []

    if train_mode == "fedalt":
        return _run_fedalt_round(
            model=model,
            dataloader=dataloader,
            device=device,
            round_idx=round_idx,
            client_id=client_id,
            epochs=epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
        )

    logger.info(
        f"[Round {round_idx}] {client_id} local training start | "
        f"global init={'server' if global_lora_state is not None else 'broadcast'} | "
        f"micro_batch_size={micro_batch_size} | gradient_accumulation_steps={gradient_accumulation_steps} | "
        f"effective_batch_size={micro_batch_size * gradient_accumulation_steps} | "
        f"train_on_inputs={train_on_inputs}"
    )

    # FedDPA-T Step 2-1: global adapter만 학습한다.
    model.dual_lora_adapter.configure_global_training()
    global_trainable_params = [
        param for param in model.dual_lora_adapter.global_parameters() if param.requires_grad
    ]
    global_optimizer = AdamW(global_trainable_params, lr=learning_rate)
    logger.info(f"[Round {round_idx}] {client_id} stage 1 start | train global adapter only")

    def _save_epoch_global_snapshot(epoch_idx: int, avg_loss: float) -> None:
        upload_payload = _sanitize_upload_payload(
            model.dual_lora_adapter.get_upload_payload(),
            client_id=client_id,
            round_idx=round_idx,
            epoch_idx=epoch_idx,
        )
        epoch_global_uploads.append(
            {
                "epoch": epoch_idx,
                "global_lora": upload_payload["global_lora"],
                "avg_loss": avg_loss,
                "had_non_finite": upload_payload["had_non_finite"],
            }
        )

    global_epoch_losses = _run_stage(
        model=model,
        dataloader=dataloader,
        device=device,
        optimizer=global_optimizer,
        trainable_params=global_trainable_params,
        round_idx=round_idx,
        client_id=client_id,
        stage_name="Global",
        epochs=epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        epoch_end_callback=_save_epoch_global_snapshot,
    )

    local_optimizer = None
    local_epoch_losses: list[float] = []
    if stage1_only:
        logger.info(f"[Round {round_idx}] {client_id} stage 2 skipped | stage1_only=True")
    else:
        # FedDPA-T Step 2-2: global adapter를 freeze하고 local adapter만 학습한다.
        model.dual_lora_adapter.configure_local_training()
        local_trainable_params = [
            param for param in model.dual_lora_adapter.local_parameters() if param.requires_grad
        ]
        local_optimizer = AdamW(local_trainable_params, lr=learning_rate)
        logger.info(f"[Round {round_idx}] {client_id} stage 2 start | freeze global, train local adapter only")
        local_epoch_losses = _run_stage(
            model=model,
            dataloader=dataloader,
            device=device,
            optimizer=local_optimizer,
            trainable_params=local_trainable_params,
            round_idx=round_idx,
            client_id=client_id,
            stage_name="Local",
            epochs=epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

    local_state_out = clone_state_dict_to_cpu(model.dual_lora_adapter.state_dict())
    upload_payload = _sanitize_upload_payload(
        model.dual_lora_adapter.get_upload_payload(),
        client_id=client_id,
        round_idx=round_idx,
    )

    del model
    del global_optimizer
    if local_optimizer is not None:
        del local_optimizer
    torch.cuda.empty_cache()

    return {
        "local_state": local_state_out,
        "upload_payload": upload_payload,
        "epoch_global_uploads": epoch_global_uploads,
        "metrics": {
            "global_epoch_losses": global_epoch_losses,
            "global_final_loss": global_epoch_losses[-1] if global_epoch_losses else None,
            "local_epoch_losses": local_epoch_losses,
            "local_final_loss": local_epoch_losses[-1] if local_epoch_losses else None,
        },
    }


def run_local_personalization_round(
    model_id: str,
    client_id: str,
    dataset_name: str,
    aligned_global_state: Dict[str, torch.Tensor],
    epochs: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    num_samples: int,
    max_length: int,
    seed: int,
    rank: int,
    lora_alpha: int,
    lora_dropout: float,
    learning_rate: float,
    train_on_inputs: bool,
    lora_target_modules: list[str],
) -> Dict[str, Any]:
    """Stage 2 personalization.

    Assigned cluster-global LoRA로 local LoRA를 initialize한 뒤,
    personalization 동안에는 global branch를 0으로 두고 local branch만 학습한다.
    최종 checkpoint에는 personalized local LoRA와 assigned cluster-global LoRA를 함께 저장한다.
    """
    hf_token = load_hf_token()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available inside the training process. "
            "Stopping early because this experiment is expected to run on GPU."
        )
    device = torch.device("cuda")
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dataloader = get_client_dataloader(
        client_id=client_id,
        tokenizer=tokenizer,
        batch_size=micro_batch_size,
        num_samples=num_samples,
        dataset_name=dataset_name,
        max_length=max_length,
        train_on_inputs=train_on_inputs,
    )

    model = setup_model_with_dual_lora(
        model_id,
        hf_token,
        rank=rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        target_modules=lora_target_modules,
    )

    aligned_local_state = _align_local_state_from_global_state(aligned_global_state)
    zero_global_state = _zero_global_state(aligned_global_state)
    initial_state = {**aligned_local_state, **zero_global_state}
    model.dual_lora_adapter.load_state_dict(initial_state, strict=False)
    model.train()

    logger.info(
        "[Personalization] %s start | epochs=%d | micro_batch_size=%d | grad_accum=%d | "
        "local initialized from assigned cluster-global LoRA | global branch zeroed during adaptation",
        client_id,
        epochs,
        micro_batch_size,
        gradient_accumulation_steps,
    )

    model.dual_lora_adapter.configure_local_training()
    local_trainable_params = [
        param for param in model.dual_lora_adapter.local_parameters() if param.requires_grad
    ]
    local_optimizer = AdamW(local_trainable_params, lr=learning_rate)
    local_epoch_losses = _run_stage(
        model=model,
        dataloader=dataloader,
        device=device,
        optimizer=local_optimizer,
        trainable_params=local_trainable_params,
        round_idx=0,
        client_id=client_id,
        stage_name="Personalization",
        epochs=epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    final_state = clone_state_dict_to_cpu(model.dual_lora_adapter.state_dict())
    personalized_local_state = {
        key: value.detach().cpu().clone()
        for key, value in final_state.items()
        if ".local_lora." in key
    }
    checkpoint_state = {
        **personalized_local_state,
        **{
            key: value.detach().cpu().clone()
            for key, value in aligned_global_state.items()
        },
    }

    del model
    del local_optimizer
    torch.cuda.empty_cache()

    return {
        "local_state": personalized_local_state,
        "checkpoint_state": checkpoint_state,
        "metrics": {
            "personalization_epoch_losses": local_epoch_losses,
            "personalization_final_loss": local_epoch_losses[-1] if local_epoch_losses else None,
        },
    }


def run_client_round(
    client,
    model_id: str,
    round_idx: int,
    dataset_name: str,
    epochs: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    num_samples: int,
    max_length: int,
    seed: int,
    rank: int,
    lora_alpha: int,
    lora_dropout: float,
    learning_rate: float,
    train_on_inputs: bool,
    lora_target_modules: list[str],
    stage1_only: bool = False,
    train_mode: str = "feddpa",
) -> Dict[str, Any]:
    """FederatedClient가 들고 있는 state를 사용해 한 round local training을 실행한다."""
    if client.local_lora_state is None or client.global_lora_state is None:
        raise ValueError(f"{client.client_id} has no initial local/global LoRA state. Broadcast state first.")

    result = run_local_federated_round(
        model_id=model_id,
        client_id=client.client_id,
        dataset_name=dataset_name,
        round_idx=round_idx,
        local_state=client.get_adapter_state(),
        global_lora_state=client.global_lora_state,
        epochs=epochs,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_samples=num_samples,
        max_length=max_length,
        seed=seed,
        rank=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        learning_rate=learning_rate,
        train_on_inputs=train_on_inputs,
        lora_target_modules=lora_target_modules,
        stage1_only=stage1_only,
        train_mode=train_mode,
        mixer_state=getattr(client, "mixer_state", None),
    )
    client.set_adapter_state(result["local_state"])
    if result.get("mixer_state") is not None:
        client.mixer_state = result["mixer_state"]
    return result
