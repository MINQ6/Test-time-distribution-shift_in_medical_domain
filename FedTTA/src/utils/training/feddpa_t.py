"""FedDPA-T training (iterative joint global+local).

FedDPA-F와의 차이:
  - FedDPA-F: 매 round global만 학습(FedAvg), local은 **맨 끝 1회** personalization
              → local이 global 복사+few epoch라 global≈local.
  - FedDPA-T: 매 round 각 client가 **global→local 둘 다 학습**하고 local을 client에 **누적 유지**.
              global만 업로드해 FedAvg, local은 client별로 남아 round마다 더 특화됨
              → local이 전 round에 걸쳐(local_epochs×num_rounds) 학습되어 global과 더 벌어짐.

구현: run_client_round(stage1_only=False)가 (1) client.get_adapter_state()로 누적 local을
입력받고 (2) global→local 순차 학습 후 (3) client.set_adapter_state로 local을 되돌려 저장한다.
서버는 global upload만 FedAvg한다. 별도 Stage2 없음.

per-round epoch 수는 local_epochs를 사용(global·local 각각). personalization_epochs는 미사용.
"""
from statistics import mean
from typing import Dict

import torch

from utils.training.common import logger
from utils.training.federated import (
    _build_checkpoint_output_dir,
    broadcast_personalized_global_lora,
    build_single_global_broadcast_state,
    initialize_broadcast_state,
)
from utils.training.local_round import run_client_round
from utils.training.parallel import collect_client_uploads_parallel, resolve_num_gpus
from utils.training.wandb import wandb_logger


def _save_feddpa_t_checkpoints(server, checkpoints_dir: str) -> dict[str, str]:
    saved_paths: dict[str, str] = {}
    for client_id, client in server.clients.items():
        if client.local_lora_state is None or client.global_lora_state is None:
            raise ValueError(f"{client_id} has no local/global LoRA state to save.")
        checkpoint_state = client.get_adapter_state()
        save_path = f"{checkpoints_dir}/dual_lora_adapter_{client_id}.pth"
        torch.save(checkpoint_state, save_path)
        saved_paths[client_id] = save_path
        logger.info("[Checkpoint] Saved FedDPA-T checkpoint for %s -> %s", client_id, save_path)
    return saved_paths


def run_feddpa_t_training(server) -> None:
    """FedDPA-T: 매 round global+local joint 학습, global만 FedAvg, local은 client별 누적."""
    wandb_logger.init(
        enabled=server.use_wandb,
        project=server.wandb_project,
        run_name=server.wandb_run_name,
        entity=server.wandb_entity,
        config={
            "method": "feddpa_t",
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
    logger.info("[FedDPA-T] Initializing shared dual LoRA state")
    logger.info("=" * 70)
    init_state = initialize_broadcast_state(server)
    initial_global_lora_state = {
        key: value.detach().cpu().clone()
        for key, value in init_state.items()
        if ".global_lora." in key
    }
    server.global_lora_states = {
        client_id: {k: v.detach().cpu().clone() for k, v in initial_global_lora_state.items()}
        for client_id in server.client_ids
    }
    for client in server.clients.values():
        client.set_initial_state(init_state)

    num_gpus = resolve_num_gpus()
    parallel = num_gpus > 1
    logger.info("[FedDPA-T] client training mode: %s (PARALLEL_CLIENT_GPUS=%d)",
                "PARALLEL" if parallel else "sequential", num_gpus)

    for round_idx in range(1, server.num_rounds + 1):
        logger.info("\n" + "=" * 70)
        logger.info("[FedDPA-T] Round %d/%d (joint global+local)", round_idx, server.num_rounds)
        logger.info("=" * 70)

        # 서버 -> client: 최신 FedAvg global 배포 (client.global 덮어씀, local은 유지)
        broadcast_personalized_global_lora(
            server, server.global_lora_states,
            phase_title="FedDPA-T global broadcast (local persists per client)",
        )

        # stage1_only=False → global→local 둘 다 학습, local은 client에 누적 저장.
        # 병렬판도 client.get_adapter_state 입력 + client.set_adapter_state 저장으로 누적 보존.
        if parallel:
            uploads = collect_client_uploads_parallel(
                server, round_idx, epochs_override=server.local_epochs,
                phase_label="FedDPA-T joint", num_gpus=num_gpus, stage1_only=False,
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
                    stage1_only=False,
                )
                uploads[client_id] = result["upload_payload"]
                uploads[client_id]["metrics"] = result.get("metrics", {})

        for client_id in server.client_ids:
            m = uploads[client_id].get("metrics", {})
            logger.info("[FedDPA-T] %s round %d | global_loss=%.4f local_loss=%.4f",
                        client_id, round_idx,
                        m.get("global_final_loss") or -1, m.get("local_final_loss") or -1)

        # 서버: global upload만 single-global FedAvg
        server.global_lora_states = build_single_global_broadcast_state(server, uploads)

        gl = [p["metrics"].get("global_final_loss") for p in uploads.values() if p["metrics"].get("global_final_loss") is not None]
        ll = [p["metrics"].get("local_final_loss") for p in uploads.values() if p["metrics"].get("local_final_loss") is not None]
        wandb_logger.log(
            {"train/round": round_idx,
             "train/mean_global_final_loss": mean(gl) if gl else None,
             "train/mean_local_final_loss": mean(ll) if ll else None},
            step=round_idx,
        )
        logger.info("[FedDPA-T] Round %d aggregation complete", round_idx)

    # 최종 FedAvg global을 broadcast해 client.global을 최종값으로 맞춘 뒤 저장
    broadcast_personalized_global_lora(
        server, server.global_lora_states,
        phase_title="FedDPA-T final global broadcast before checkpoint",
    )
    checkpoints_dir = _build_checkpoint_output_dir()
    import os
    os.makedirs(checkpoints_dir, exist_ok=True)
    saved = _save_feddpa_t_checkpoints(server, checkpoints_dir)
    logger.info("[FedDPA-T] done | %d client checkpoints under %s", len(saved), checkpoints_dir)
    wandb_logger.summary_update({"artifacts/checkpoints_dir": checkpoints_dir,
                                 "artifacts/final_checkpoint_count": len(saved)})
    wandb_logger.finish()
