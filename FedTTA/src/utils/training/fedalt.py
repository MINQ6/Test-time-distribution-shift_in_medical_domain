"""FedALT training (iterative federation with leave-one-out RoW aggregation).

Mapping to our dual-LoRA codebase:
  - Individual LoRA = ``local_lora`` (trainable, accumulated across rounds, uploaded).
  - RoW LoRA        = ``global_lora`` (FROZEN during local training, set by server).
  - mixer           = per-LoRA-layer Linear(in_features -> 2) + softmax (kept local).

Round t = 1..R:
  1. Server sends RoW (frozen) to each client (broadcast of per-client global_lora).
  2. Client trains Individual + mixer only (RoW & base frozen) for ``local_epochs``,
     forwarding through the per-token mixer gate
     (out = base + alpha_indiv * Individual(x) + alpha_row * RoW(x),
      [alpha_indiv, alpha_row] = softmax(mixer(x)); index 0=Individual, 1=RoW).
  3. Client uploads Individual only (mixer kept local, never aggregated).
  4. Server sets each client's RoW = leave-one-out mean of OTHER clients' Individual
     (A and B averaged separately). Individual persists/accumulates per client
     across rounds via client.get_adapter_state()/set_adapter_state().

Checkpoints: dual_lora_adapter_client_{k}.pth (local=Individual, global=RoW)
             + mixer_client_{k}.pth (per-client mixer).
"""
from statistics import mean
from typing import Dict

import os

import torch

from utils.training.common import checkpoint_suffix, logger
from utils.training.federated import (
    _build_checkpoint_output_dir,
    broadcast_personalized_global_lora,
    initialize_broadcast_state,
)
from utils.training.local_round import run_client_round
from utils.training.parallel import collect_client_uploads_parallel, resolve_num_gpus
from utils.training.wandb import wandb_logger


def build_leave_one_out_row_states(
    server,
    uploads: Dict[str, Dict],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """RoW for each client k = (1/(K-1)) * sum_{m != k} Individual_m (A and B separately).

    uploads[client]["global_lora"][module_name] = {"A": ..., "B": ...} carries the
    client's uploaded Individual(local) LoRA matrices. Result is keyed as global_lora
    state so it can be loaded as the (frozen) RoW branch.
    """
    client_ids = list(server.client_ids)
    reference = client_ids[0]
    module_names = sorted(uploads[reference]["global_lora"].keys())

    row_states: Dict[str, Dict[str, torch.Tensor]] = {cid: {} for cid in client_ids}
    for module_name in module_names:
        a_by_client = {
            cid: uploads[cid]["global_lora"][module_name]["A"].float()
            for cid in client_ids
            if module_name in uploads[cid].get("global_lora", {})
        }
        b_by_client = {
            cid: uploads[cid]["global_lora"][module_name]["B"].float()
            for cid in client_ids
            if module_name in uploads[cid].get("global_lora", {})
        }
        for k in client_ids:
            others_a = [a for cid, a in a_by_client.items() if cid != k]
            others_b = [b for cid, b in b_by_client.items() if cid != k]
            if not others_a or not others_b:
                continue
            row_a = torch.stack(others_a, dim=0).mean(dim=0).cpu()
            row_b = torch.stack(others_b, dim=0).mean(dim=0).cpu()
            row_states[k][f"{module_name}.global_lora.lora_A.weight"] = row_a
            row_states[k][f"{module_name}.global_lora.lora_B.weight"] = row_b
    return row_states


def _save_fedalt_checkpoints(server, checkpoints_dir: str) -> dict[str, str]:
    saved_paths: dict[str, str] = {}
    for client_id, client in server.clients.items():
        if client.local_lora_state is None or client.global_lora_state is None:
            raise ValueError(f"{client_id} has no Individual/RoW LoRA state to save.")
        suffix = checkpoint_suffix(client_id)
        # Dual LoRA checkpoint: local=Individual, global=RoW (same format as feddpa).
        checkpoint_state = client.get_adapter_state()
        save_path = os.path.join(checkpoints_dir, f"dual_lora_adapter_client_{suffix}.pth")
        torch.save(checkpoint_state, save_path)
        saved_paths[client_id] = save_path
        logger.info("[Checkpoint] Saved FedALT dual LoRA for %s -> %s", client_id, save_path)
        # Mixer checkpoint (per-client, kept local).
        if client.mixer_state is not None:
            mixer_path = os.path.join(checkpoints_dir, f"mixer_client_{suffix}.pth")
            torch.save(
                {k: v.detach().cpu().clone() for k, v in client.mixer_state.items()},
                mixer_path,
            )
            logger.info("[Checkpoint] Saved FedALT mixer for %s -> %s", client_id, mixer_path)
    return saved_paths


def run_fedalt_training(server) -> None:
    """FedALT: train Individual+mixer per round, upload Individual, RoW = leave-one-out mean."""
    wandb_logger.init(
        enabled=server.use_wandb,
        project=server.wandb_project,
        run_name=server.wandb_run_name,
        entity=server.wandb_entity,
        config={
            "method": "fedalt",
            "model_id": server.model_id,
            "dataset_name": server.dataset_name,
            "num_clients": len(server.client_ids),
            "num_rounds": server.num_rounds,
            "local_epochs": server.local_epochs,
            "batch_size": server.batch_size,
            "micro_batch_size": server.micro_batch_size,
            "gradient_accumulation_steps": server.gradient_accumulation_steps,
            "num_samples": server.num_samples,
            "max_length": server.max_length,
            "seed": server.seed,
            "rank": server.rank,
            "lora_alpha": server.lora_alpha,
            "lora_dropout": server.lora_dropout,
            "learning_rate": server.learning_rate,
            "lora_target_modules": server.lora_target_modules,
        },
    )

    logger.info("\n" + "=" * 70)
    logger.info("[FedALT] Initializing dual LoRA state (Individual random, RoW zero)")
    logger.info("=" * 70)
    init_state = initialize_broadcast_state(server)
    # RoW (global_lora) starts at zero; the LoRA B init is already zero so the
    # broadcast state has RoW == 0 contribution. Keep per-client RoW states.
    initial_row_state = {
        key: torch.zeros_like(value).cpu()
        for key, value in init_state.items()
        if ".global_lora." in key
    }
    server.global_lora_states = {
        client_id: {k: v.clone() for k, v in initial_row_state.items()}
        for client_id in server.client_ids
    }
    for client in server.clients.values():
        client.set_initial_state(init_state)

    num_gpus = resolve_num_gpus()
    parallel = num_gpus > 1
    logger.info("[FedALT] client training mode: %s (PARALLEL_CLIENT_GPUS=%d)",
                "PARALLEL" if parallel else "sequential", num_gpus)

    for round_idx in range(1, server.num_rounds + 1):
        logger.info("\n" + "=" * 70)
        logger.info("[FedALT] Round %d/%d (train Individual+mixer, RoW frozen)", round_idx, server.num_rounds)
        logger.info("=" * 70)

        # Server -> client: send current RoW (frozen). Individual persists per client.
        broadcast_personalized_global_lora(
            server, server.global_lora_states,
            phase_title="FedALT RoW broadcast (Individual persists per client)",
        )

        if parallel:
            uploads = collect_client_uploads_parallel(
                server, round_idx, epochs_override=server.local_epochs,
                phase_label="FedALT", num_gpus=num_gpus, stage1_only=False,
                train_mode="fedalt",
            )
        else:
            uploads = {}
            for client_id, client in server.clients.items():
                result = run_client_round(
                    client=client, model_id=server.model_id, round_idx=round_idx,
                    dataset_name=server.dataset_name, epochs=server.local_epochs,
                    micro_batch_size=server.micro_batch_size,
                    gradient_accumulation_steps=server.gradient_accumulation_steps,
                    num_samples=server.num_samples, max_length=server.max_length,
                    seed=server.seed, rank=server.rank, lora_alpha=server.lora_alpha,
                    lora_dropout=server.lora_dropout, learning_rate=server.learning_rate,
                    train_on_inputs=server.train_on_inputs,
                    lora_target_modules=server.lora_target_modules,
                    stage1_only=False, train_mode="fedalt",
                )
                uploads[client_id] = result["upload_payload"]
                uploads[client_id]["metrics"] = result.get("metrics", {})

        for client_id in server.client_ids:
            m = uploads[client_id].get("metrics", {})
            logger.info("[FedALT] %s round %d | loss=%.4f alpha_indiv_mean=%s",
                        client_id, round_idx,
                        m.get("fedalt_final_loss") if m.get("fedalt_final_loss") is not None else -1,
                        ("%.3f" % m["alpha_indiv_mean"]) if m.get("alpha_indiv_mean") is not None else "n/a")

        # Server: each client's RoW = leave-one-out mean of OTHER clients' Individual.
        server.global_lora_states = build_leave_one_out_row_states(server, uploads)

        losses = [
            p["metrics"].get("fedalt_final_loss")
            for p in uploads.values()
            if p["metrics"].get("fedalt_final_loss") is not None
        ]
        alphas = [
            p["metrics"].get("alpha_indiv_mean")
            for p in uploads.values()
            if p["metrics"].get("alpha_indiv_mean") is not None
        ]
        wandb_logger.log(
            {"train/round": round_idx,
             "train/mean_fedalt_final_loss": mean(losses) if losses else None,
             "train/mean_alpha_indiv_mean": mean(alphas) if alphas else None},
            step=round_idx,
        )
        logger.info("[FedALT] Round %d aggregation complete", round_idx)

    # Final RoW broadcast so each client's stored RoW matches the last aggregation.
    broadcast_personalized_global_lora(
        server, server.global_lora_states,
        phase_title="FedALT final RoW broadcast before checkpoint",
    )
    checkpoints_dir = _build_checkpoint_output_dir()
    os.makedirs(checkpoints_dir, exist_ok=True)
    saved = _save_fedalt_checkpoints(server, checkpoints_dir)
    logger.info("[FedALT] done | %d client checkpoints under %s", len(saved), checkpoints_dir)
    wandb_logger.summary_update({"artifacts/checkpoints_dir": checkpoints_dir,
                                 "artifacts/final_checkpoint_count": len(saved)})
    wandb_logger.finish()
