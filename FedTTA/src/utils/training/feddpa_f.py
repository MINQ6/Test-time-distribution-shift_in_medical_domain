import os
from statistics import mean
from typing import Dict

import torch

from utils.analysis import analyze_upload_payloads
from utils.training.common import logger
from utils.training.federated import (
    _build_analysis_output_dir,
    _build_checkpoint_output_dir,
    _phase_banner,
    broadcast_personalized_global_lora,
    build_single_global_broadcast_state,
    collect_client_uploads,
    initialize_broadcast_state,
    run_stage2_local_personalization,
)
from utils.training.parallel import (
    collect_client_uploads_parallel,
    resolve_num_gpus,
    run_stage2_local_personalization_parallel,
)
from utils.training.wandb import wandb_logger


def _save_feddpa_f_checkpoints(server, checkpoints_dir: str) -> dict[str, str]:
    saved_paths: dict[str, str] = {}
    for client_id, client in server.clients.items():
        if client.local_lora_state is None or client.global_lora_state is None:
            raise ValueError(f"{client_id} has no local/global LoRA state to save.")

        checkpoint_state = client.get_adapter_state()
        save_path = f"{checkpoints_dir}/dual_lora_adapter_{client_id}.pth"
        torch.save(checkpoint_state, save_path)
        saved_paths[client_id] = save_path
        logger.info("[Checkpoint] Saved FedDPA-F checkpoint for %s -> %s", client_id, save_path)
    return saved_paths


def run_feddpa_f_training(server) -> None:
    """FedDPA-F training with final local personalization.

    - Broadcast initial dual LoRA state to every client
    - Stage 1: every round trains/uploads only the global LoRA branch, then applies single-global FedAvg
    - Stage 2: broadcast the final shared global LoRA, align local LoRA from it, and run local-only personalization
    """
    wandb_logger.init(
        enabled=server.use_wandb,
        project=server.wandb_project,
        run_name=server.wandb_run_name,
        entity=server.wandb_entity,
        config={
            "method": "feddpa_f",
            "model_id": server.model_id,
            "dataset_name": server.dataset_name,
            "num_clients": len(server.client_ids),
            "num_rounds": server.num_rounds,
            "local_epochs": server.local_epochs,
            "personalization_epochs": server.personalization_epochs,
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
            "train_on_inputs": server.train_on_inputs,
            "lora_target_modules": server.lora_target_modules,
        },
    )

    num_gpus = resolve_num_gpus()
    parallel = num_gpus > 1
    logger.info("[FedDPA-F] client training mode: %s (PARALLEL_CLIENT_GPUS=%d)",
                "PARALLEL" if parallel else "sequential", num_gpus)

    logger.info("\n" + "=" * 70)
    logger.info("[FedDPA-F] Initializing shared dual LoRA state and single global broadcast")
    logger.info("=" * 70)
    init_state = initialize_broadcast_state(server)
    initial_global_lora_state = {
        key: value.detach().cpu().clone()
        for key, value in init_state.items()
        if ".global_lora." in key
    }
    server.global_lora_states = {
        client_id: {
            key: value.detach().cpu().clone()
            for key, value in initial_global_lora_state.items()
        }
        for client_id in server.client_ids
    }
    for client in server.clients.values():
        client.set_initial_state(init_state)

    final_round_uploads: Dict[str, Dict] | None = None

    for round_idx in range(1, server.num_rounds + 1):
        logger.info("\n" + "=" * 70)
        logger.info("[FedDPA-F] Round %d/%d", round_idx, server.num_rounds)
        logger.info("=" * 70)

        broadcast_personalized_global_lora(
            server,
            server.global_lora_states,
            phase_title="Shared global LoRA broadcast for FedDPA-F",
        )
        if parallel:
            uploads = collect_client_uploads_parallel(
                server, round_idx,
                epochs_override=server.local_epochs,
                phase_label="FedDPA-F Global Training",
                num_gpus=num_gpus,
            )
        else:
            uploads = collect_client_uploads(
                server,
                round_idx,
                epochs_override=server.local_epochs,
                phase_label="FedDPA-F Global Training",
            )
        final_round_uploads = uploads
        server.global_lora_states = build_single_global_broadcast_state(server, uploads)

        round_global_losses = [
            payload["metrics"]["global_final_loss"]
            for payload in uploads.values()
            if payload.get("metrics", {}).get("global_final_loss") is not None
        ]
        wandb_logger.log(
            {
                "train/round": round_idx,
                "train/mean_global_final_loss": mean(round_global_losses) if round_global_losses else None,
            },
            step=round_idx,
        )

        _phase_banner(4, "Next round will start from the single FedAvg global LoRA")
        logger.info("[FedDPA-F] Round %d aggregation complete", round_idx)

    if final_round_uploads is None:
        raise ValueError("No FedDPA-F round upload payload was collected.")

    logger.info("\n" + "=" * 70)
    logger.info("[FedDPA-F] Saving stage-1 analysis and starting final personalization")
    logger.info("=" * 70)

    broadcast_personalized_global_lora(
        server,
        server.global_lora_states,
        phase_title="Final shared global LoRA broadcast before local personalization",
    )

    checkpoints_dir = _build_checkpoint_output_dir()
    analysis_output_dir = _build_analysis_output_dir(server, server.num_rounds)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(analysis_output_dir, exist_ok=True)

    # SAFETY: persist Stage-1 results immediately so a Stage-2 crash never throws
    # away 20 rounds of training. We save (a) each client's pre-FedAvg upload, and
    # (b) the aggregated final shared global LoRA that Stage 2 starts from.
    stage1_dir = os.path.join(checkpoints_dir, "stage1_global_lora")
    os.makedirs(stage1_dir, exist_ok=True)
    for client_id, payload in final_round_uploads.items():
        torch.save(payload.get("global_lora", {}), os.path.join(stage1_dir, f"stage1_upload_{client_id}.pth"))
    reference_client = next(iter(server.client_ids))
    torch.save(server.global_lora_states[reference_client], os.path.join(stage1_dir, "stage1_global_lora_aggregated.pth"))
    logger.info("[FedDPA-F Stage 1] saved safety checkpoints -> %s", stage1_dir)

    # SKIPPED: analyze_upload_payloads(...). It runs an O(28 pairs × 224 modules)
    # SVD+QR similarity diagnostic over 4096×4096 ΔW = B@A — CPU-bound, can take
    # >24h. Not part of FedDPA-F and produces no artifact used downstream.
    artifact_paths: dict[str, str] = {}

    if parallel:
        final_checkpoint_paths = run_stage2_local_personalization_parallel(
            server, checkpoints_dir, num_gpus=num_gpus
        )
    else:
        final_checkpoint_paths = run_stage2_local_personalization(server, checkpoints_dir)

    wandb_logger.summary_update(
        {
            "artifacts/checkpoints_dir": checkpoints_dir,
            "artifacts/analysis_output_dir": analysis_output_dir,
            "artifacts/final_checkpoint_count": len(final_checkpoint_paths),
            "train/personalization_epochs": server.personalization_epochs,
        }
    )
    wandb_logger.log_analysis_artifacts(artifact_paths)
    wandb_logger.finish()
