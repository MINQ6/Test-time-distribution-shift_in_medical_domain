from __future__ import annotations

import os
from statistics import mean

from utils.training.federated import (
    _save_cluster_assignments_json,
    _save_cluster_global_checkpoints,
    broadcast_personalized_global_lora,
    collect_client_uploads,
    initialize_broadcast_state,
    initialize_clustered_training,
)
from utils.training.common import logger
from utils.training.wandb import wandb_logger


def _build_checkpoint_output_dir() -> str:
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    )
    timestamp = os.environ.get("RUN_TIMESTAMP")
    if timestamp:
        return os.path.join(project_root, "checkpoints", timestamp)
    return os.path.join(project_root, "checkpoints")


def run_warmup_clustering_only(server) -> None:
    """Phase 0 warm-up -> server-side clustering only."""
    wandb_logger.init(
        enabled=server.use_wandb,
        project=server.wandb_project,
        run_name=server.wandb_run_name,
        entity=server.wandb_entity,
        config={
            "mode": "warmup_clustering_only",
            "model_id": server.model_id,
            "dataset_name": server.dataset_name,
            "num_clients": len(server.client_ids),
            "warmup_epochs": server.warmup_epochs,
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
            "cluster_similarity_factor": server.cluster_similarity_factor,
            "cluster_orthogonalization": server.cluster_orthogonalization,
        },
    )

    logger.info("\n" + "=" * 70)
    logger.info("[Warm-up Clustering Only] Initializing common dual LoRA weights")
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

    logger.info("\n" + "=" * 70)
    logger.info("[Warm-up Clustering Only] Phase 0 warm-up")
    logger.info("=" * 70)
    broadcast_personalized_global_lora(
        server,
        server.global_lora_states,
        phase_title="Warm-up broadcast of the single shared global LoRA",
    )
    warmup_uploads = collect_client_uploads(
        server,
        0,
        epochs_override=server.warmup_epochs,
        phase_label="Phase 0 Warm-up",
    )
    clustering_artifacts = initialize_clustered_training(server, warmup_uploads, round_idx=0)

    warmup_global_losses = [
        payload["metrics"]["global_final_loss"]
        for payload in warmup_uploads.values()
        if payload.get("metrics", {}).get("global_final_loss") is not None
    ]
    wandb_logger.log(
        {
            "warmup/epochs": server.warmup_epochs,
            "warmup/mean_global_final_loss": mean(warmup_global_losses) if warmup_global_losses else None,
        },
        step=0,
    )

    checkpoints_dir = _build_checkpoint_output_dir()
    os.makedirs(checkpoints_dir, exist_ok=True)
    cluster_checkpoint_paths = _save_cluster_global_checkpoints(
        server.cluster_global_lora_states,
        checkpoints_dir,
    )
    cluster_assignments_path = _save_cluster_assignments_json(
        checkpoints_dir,
        server.cluster_assignments,
        {
            cluster_id: sorted(
                [
                    client_id
                    for client_id, assigned_cluster in server.cluster_assignments.items()
                    if assigned_cluster == cluster_id
                ]
            )
            for cluster_id in sorted(server.cluster_global_lora_states.keys())
        },
    )
    wandb_logger.summary_update(
        {
            "cluster/factor": server.cluster_similarity_factor,
            "cluster/selected_k": len(server.cluster_global_lora_states),
            "artifacts/checkpoints_dir": checkpoints_dir,
            "artifacts/cluster_assignments_json": cluster_assignments_path,
        }
    )
    for cluster_id, path in sorted(cluster_checkpoint_paths.items()):
        wandb_logger.summary_update({f"artifacts/cluster_{cluster_id}_checkpoint": path})
    wandb_logger.log_analysis_artifacts(clustering_artifacts)
    wandb_logger.finish()
