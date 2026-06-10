import json
import os
from statistics import mean
from typing import Any, Dict

import torch

from models.dual_lora_model import setup_model_with_dual_lora
from utils.analysis import analyze_upload_payloads, extract_global_lora_from_uploads
from utils.training.clustering import (
    aggregate_cluster_global_lora,
    build_personalized_cluster_broadcast,
    compute_client_similarity_matrix,
    orthogonalize_cluster_deltaw_states,
    save_clustering_artifacts,
    select_cluster_assignments,
)
from utils.training.common import checkpoint_suffix, logger, set_seed
from utils.training.local_round import run_local_personalization_round, run_client_round
from utils.training.wandb import wandb_logger


def initialize_broadcast_state(server) -> Dict[str, torch.Tensor]:
    """서버가 공통 초기 dual LoRA state를 한 번 만들고 모든 client에 broadcast할 준비를 한다."""
    set_seed(server.seed)
    model = setup_model_with_dual_lora(
        server.model_id,
        server.hf_token,
        rank=server.rank,
        alpha=server.lora_alpha,
        dropout=server.lora_dropout,
        target_modules=server.lora_target_modules,
    )
    init_state = {
        key: value.detach().cpu().clone()
        for key, value in model.dual_lora_adapter.state_dict().items()
    }
    del model
    torch.cuda.empty_cache()
    return init_state


def _phase_banner(phase_idx: int, title: str) -> None:
    logger.info("-" * 70)
    logger.info(f"[Phase {phase_idx}] {title}")
    logger.info("-" * 70)


def aggregate_global_lora_fedavg(
    uploads: Dict[str, Dict],
) -> Dict[str, torch.Tensor]:
    """Fallback single-global FedAvg helper."""
    if not uploads:
        raise ValueError("No client uploads were provided for aggregation.")

    reference_client = sorted(uploads.keys())[0]
    module_names = sorted(uploads[reference_client]["global_lora"].keys())
    aggregated_global_state: Dict[str, torch.Tensor] = {}

    for module_name in module_names:
        a_matrices = [
            payload["global_lora"][module_name]["A"].float()
            for payload in uploads.values()
            if module_name in payload.get("global_lora", {})
        ]
        b_matrices = [
            payload["global_lora"][module_name]["B"].float()
            for payload in uploads.values()
            if module_name in payload.get("global_lora", {})
        ]
        if not a_matrices or not b_matrices:
            continue

        aggregated_global_state[f"{module_name}.global_lora.lora_A.weight"] = (
            torch.stack(a_matrices, dim=0).mean(dim=0).cpu()
        )
        aggregated_global_state[f"{module_name}.global_lora.lora_B.weight"] = (
            torch.stack(b_matrices, dim=0).mean(dim=0).cpu()
        )

    return aggregated_global_state


def _build_analysis_output_dir(server, round_idx: int) -> str:
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    )
    timestamp = os.environ.get("RUN_TIMESTAMP")
    if timestamp:
        return os.path.join(
            project_root,
            "outputs",
            timestamp,
            "global_lora_analysis",
            f"{server.dataset_name}_round_{round_idx}_clients_{len(server.client_ids)}",
        )
    return os.path.join(
        project_root,
        "outputs",
        "global_lora_analysis",
        f"{server.dataset_name}_round_{round_idx}_clients_{len(server.client_ids)}",
    )


def _build_checkpoint_output_dir() -> str:
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    )
    timestamp = os.environ.get("RUN_TIMESTAMP")
    if timestamp:
        return os.path.join(project_root, "src", "checkpoints", timestamp)
    return os.path.join(project_root, "src", "checkpoints")


def _build_epoch_analysis_root_dir(server) -> str:
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    )
    timestamp = os.environ.get("RUN_TIMESTAMP")
    if timestamp:
        return os.path.join(
            project_root,
            "outputs",
            timestamp,
            "epoch_global_lora_analysis",
            server.dataset_name,
        )
    return os.path.join(
        project_root,
        "outputs",
        "epoch_global_lora_analysis",
        f"{server.dataset_name}_manual",
    )


def _build_clustering_output_dir(server, warmup_round_idx: int) -> str:
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    )
    timestamp = os.environ.get("RUN_TIMESTAMP")
    if timestamp:
        return os.path.join(
            project_root,
            "outputs",
            timestamp,
            "cluster_training",
            f"warmup_round_{warmup_round_idx:02d}",
        )
    return os.path.join(
        project_root,
        "outputs",
        "cluster_training",
        f"{server.dataset_name}_manual",
        f"warmup_round_{warmup_round_idx:02d}",
    )


def _save_cluster_assignments_json(
    output_dir: str,
    assignments: Dict[str, int],
    cluster_members: dict[int, list[str]],
) -> str:
    path = os.path.join(output_dir, "cluster_assignments.json")
    payload = {
        "assignments": assignments,
        "cluster_members": {str(k): v for k, v in sorted(cluster_members.items())},
    }
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)
    return path


def _save_cluster_global_checkpoints(
    cluster_states: Dict[int, Dict[str, torch.Tensor]],
    checkpoints_dir: str,
) -> dict[int, str]:
    cluster_dir = os.path.join(checkpoints_dir, "cluster_global")
    os.makedirs(cluster_dir, exist_ok=True)
    saved_paths: dict[int, str] = {}
    for cluster_id, state in sorted(cluster_states.items()):
        save_path = os.path.join(cluster_dir, f"cluster_global_lora_cluster_{cluster_id}.pth")
        torch.save(state, save_path)
        saved_paths[cluster_id] = save_path
        logger.info("[Checkpoint] Saved cluster-global LoRA for cluster %d -> %s", cluster_id, save_path)
    return saved_paths


def save_phase1_client_checkpoints_from_uploads(
    clients: Dict[str, Any],
    final_uploads: Dict[str, Dict],
    checkpoints_dir: str,
    *,
    subdir: str | None = None,
) -> None:
    """
    마지막 phase-1 round에서 각 client가 업로드한 global LoRA를 그대로 checkpoint에 보존한다.
    """
    output_dir = os.path.join(checkpoints_dir, subdir) if subdir else checkpoints_dir
    os.makedirs(output_dir, exist_ok=True)
    for client_id, client in clients.items():
        if client.local_lora_state is None or client.global_lora_state is None:
            raise ValueError(f"{client_id} has no local/global LoRA state to save.")

        if client_id not in final_uploads:
            raise ValueError(f"Missing final upload payload for {client_id}")

        checkpoint_state = client.get_adapter_state()
        for module_name, module_payload in final_uploads[client_id]["global_lora"].items():
            checkpoint_state[f"{module_name}.global_lora.lora_A.weight"] = (
                module_payload["A"].detach().cpu().clone()
            )
            checkpoint_state[f"{module_name}.global_lora.lora_B.weight"] = (
                module_payload["B"].detach().cpu().clone()
            )

        save_path = os.path.join(output_dir, f"dual_lora_adapter_client_{checkpoint_suffix(client_id)}.pth")
        torch.save(checkpoint_state, save_path)
        logger.info("[Checkpoint] Saved phase-1 client upload for %s -> %s", client_id, save_path)


def _save_personalized_checkpoints(
    server,
    checkpoints_dir: str,
) -> dict[str, str]:
    saved_paths: dict[str, str] = {}
    for client_id, client in server.clients.items():
        checkpoint_state = {
            **{
                key: value.detach().cpu().clone()
                for key, value in client.local_lora_state.items()
            },
            **{
                key: value.detach().cpu().clone()
                for key, value in client.global_lora_state.items()
            },
        }
        save_path = os.path.join(checkpoints_dir, f"dual_lora_adapter_client_{checkpoint_suffix(client_id)}.pth")
        torch.save(checkpoint_state, save_path)
        saved_paths[client_id] = save_path
        logger.info("[Checkpoint] Saved personalized checkpoint for %s -> %s", client_id, save_path)
    return saved_paths


def _analyze_epoch_snapshots(
    server,
    round_idx: int,
    epoch_uploads_by_client: Dict[str, list[Dict[str, Any]]],
) -> Dict[str, str]:
    if not epoch_uploads_by_client:
        return {}

    epoch_counts = {client_id: len(entries) for client_id, entries in epoch_uploads_by_client.items()}
    expected_epochs = sorted(set(epoch_counts.values()))
    if len(expected_epochs) != 1:
        raise ValueError(f"Epoch snapshot counts are inconsistent across clients: {epoch_counts}")

    output_root = _build_epoch_analysis_root_dir(server)
    artifact_paths: Dict[str, str] = {}
    final_epoch_idx = expected_epochs[0]
    uploads = {}
    for client_id, entries in epoch_uploads_by_client.items():
        snapshot = entries[final_epoch_idx - 1]
        uploads[client_id] = {
            "global_lora": snapshot["global_lora"],
            "metrics": {
                "global_epoch_loss": snapshot.get("avg_loss"),
            },
        }

    epoch_dir = os.path.join(output_root, f"round_{round_idx:02d}", f"epoch_{final_epoch_idx:02d}")
    epoch_artifacts = analyze_upload_payloads(
        uploads=uploads,
        output_dir=epoch_dir,
        source_name=(
            f"{server.dataset_name} | round {round_idx} epoch {final_epoch_idx} "
            "phase-1 global LoRA layer-wise similarity"
        ),
        methods=[],
    )
    artifact_paths[f"round_{round_idx:02d}_epoch_{final_epoch_idx:02d}"] = epoch_dir
    logger.info(
        "[Epoch Analysis] Round %d final epoch %d artifacts saved under %s",
        round_idx,
        final_epoch_idx,
        epoch_dir,
    )
    for name, path in sorted(epoch_artifacts.items()):
        logger.info("  - %s: %s", name, path)

    return artifact_paths


def collect_client_uploads(
    server,
    round_idx: int,
    *,
    epochs_override: int | None = None,
    phase_label: str = "Phase 1",
) -> Dict[str, Dict]:
    """
    각 client가 local training 후 global LoRA (B^m, A^m)를 서버로 업로드한다.
    """
    effective_epochs = epochs_override or server.local_epochs
    _phase_banner(
        1,
        f"{phase_label} | Client -> Server upload of global LoRA (round={round_idx}, epochs={effective_epochs})",
    )
    uploads: Dict[str, Dict] = {}
    epoch_uploads_by_client: Dict[str, list[Dict[str, Any]]] = {}
    for client_id, client in server.clients.items():
        result = run_client_round(
            client=client,
            model_id=server.model_id,
            round_idx=round_idx,
            dataset_name=server.dataset_name,
            epochs=effective_epochs,
            micro_batch_size=server.micro_batch_size,
            gradient_accumulation_steps=server.gradient_accumulation_steps,
            num_samples=server.num_samples,
            max_length=server.max_length,
            seed=server.seed,
            rank=server.rank,
            lora_alpha=server.lora_alpha,
            lora_dropout=server.lora_dropout,
            learning_rate=server.learning_rate,
            train_on_inputs=server.train_on_inputs,
            lora_target_modules=server.lora_target_modules,
            stage1_only=True,
        )
        uploads[client_id] = result["upload_payload"]
        uploads[client_id]["metrics"] = result.get("metrics", {})
        epoch_uploads_by_client[client_id] = result.get("epoch_global_uploads", [])
        logger.info("[Phase 1] Collected upload from %s", client_id)
        metrics = result.get("metrics", {})
        if result["upload_payload"].get("had_non_finite"):
            logger.warning(
                "[Phase 1] %s produced non-finite global LoRA values during round %d. "
                "Saved payloads were sanitized before aggregation/analysis.",
                client_id,
                round_idx,
            )
        wandb_logger.log(
            {
                "train/round": round_idx,
                f"train/{client_id}/global_final_loss": metrics.get("global_final_loss"),
                f"train/{client_id}/global_epoch_losses": metrics.get("global_epoch_losses"),
                f"train/{client_id}/global_had_non_finite": result["upload_payload"].get("had_non_finite"),
            },
            step=round_idx,
        )

    if server.save_epoch_snapshots:
        _phase_banner(1, f"{phase_label} | Epoch-wise client global LoRA snapshot analysis")
        _analyze_epoch_snapshots(
            server=server,
            round_idx=round_idx,
            epoch_uploads_by_client=epoch_uploads_by_client,
        )
    return uploads


def build_single_global_broadcast_state(
    server,
    uploads: Dict[str, Dict],
) -> Dict[str, Dict[str, torch.Tensor]]:
    _phase_banner(2, "Server aggregates uploaded global LoRA with single-global FedAvg")
    shared_global_lora = aggregate_global_lora_fedavg(uploads)
    return {
        client_id: {
            key: value.detach().cpu().clone()
            for key, value in shared_global_lora.items()
        }
        for client_id in server.client_ids
    }


def _apply_cluster_orthogonalization(
    server,
    cluster_states: Dict[int, Dict[str, torch.Tensor]],
    *,
    phase_label: str,
) -> tuple[Dict[int, Dict[str, torch.Tensor]], dict[str, object]]:
    orthogonalization = {"applied": False, "space": "BA", "phase": phase_label}
    if not server.cluster_orthogonalization:
        return cluster_states, orthogonalization

    cluster_states, orthogonalization = orthogonalize_cluster_deltaw_states(cluster_states)
    orthogonalization["phase"] = phase_label

    if orthogonalization.get("degenerate_modules"):
        logger.warning(
            "[%s] Orthogonalization encountered degenerate modules: %s",
            phase_label,
            orthogonalization["degenerate_modules"][:5],
        )

    approximation_summary = orthogonalization.get("approximation_summary", {})
    if approximation_summary:
        logger.info(
            "[%s] BA orthogonalization rank-r refactorization | mean_relative_error=%.6f | max_relative_error=%.6f | count=%s",
            phase_label,
            approximation_summary.get("mean_relative_error", 0.0),
            approximation_summary.get("max_relative_error", 0.0),
            approximation_summary.get("count", 0),
        )

    return cluster_states, orthogonalization


def initialize_clustered_training(
    server,
    warmup_uploads: Dict[str, Dict],
    *,
    round_idx: int,
) -> dict[str, str]:
    _phase_banner(2, "Server-side warm-up clustering and cluster-global LoRA construction")
    client_global_lora = extract_global_lora_from_uploads(warmup_uploads)
    client_ids, similarity, pair_details = compute_client_similarity_matrix(
        client_global_lora,
        factor=server.cluster_similarity_factor,
    )
    assignments, diagnostics = select_cluster_assignments(client_ids, similarity)
    cluster_states, cluster_members = aggregate_cluster_global_lora(warmup_uploads, assignments)

    cluster_states, orthogonalization = _apply_cluster_orthogonalization(
        server,
        cluster_states,
        phase_label="Warm-up Orthogonalization",
    )

    server.cluster_assignments = assignments
    server.cluster_global_lora_states = cluster_states
    server.global_lora_states = build_personalized_cluster_broadcast(cluster_states, assignments)
    for client_id, cluster_id in assignments.items():
        server.clients[client_id].set_cluster_id(cluster_id)

    clustering_output_dir = _build_clustering_output_dir(server, round_idx)
    artifact_paths = save_clustering_artifacts(
        output_dir=clustering_output_dir,
        client_ids=client_ids,
        similarity=similarity,
        pair_details=pair_details,
        assignments=assignments,
        diagnostics=diagnostics,
        factor=server.cluster_similarity_factor,
        orthogonalization=orthogonalization,
        source_name=f"{server.dataset_name} | warm-up round {round_idx}",
    )
    assignments_json = _save_cluster_assignments_json(clustering_output_dir, assignments, cluster_members)
    artifact_paths["cluster_assignments_json"] = assignments_json

    logger.info(
        "[Warm-up Clustering] factor=%s | selected_k=%s | gap_k=%s | elbow_k=%s | %s silhouette=%.4f | method=%s",
        server.cluster_similarity_factor,
        diagnostics["selected_k"],
        diagnostics.get("gap_selected_k"),
        diagnostics.get("elbow_selected_k"),
        diagnostics.get("silhouette_variant", "singleton_safe"),
        diagnostics["selected_silhouette"],
        diagnostics.get("selection_method", "gap_elbow_agreement"),
    )
    for cluster_id, members in sorted(cluster_members.items()):
        logger.info("[Warm-up Clustering] cluster %d -> %s", cluster_id, ", ".join(members))

    wandb_logger.summary_update(
        {
            "cluster/factor": server.cluster_similarity_factor,
            "cluster/selected_k": diagnostics["selected_k"],
            "cluster/selected_silhouette": diagnostics["selected_silhouette"],
            "cluster/gap_selected_k": diagnostics.get("gap_selected_k"),
            "cluster/elbow_selected_k": diagnostics.get("elbow_selected_k"),
            "cluster/selection_method": diagnostics.get("selection_method"),
            "artifacts/clustering_output_dir": clustering_output_dir,
        }
    )
    return artifact_paths


def build_clusterwise_broadcast_state(
    server,
    uploads: Dict[str, Dict],
) -> Dict[str, Dict[str, torch.Tensor]]:
    _phase_banner(2, "Server aggregates uploaded global LoRA within each cluster")
    if not server.cluster_assignments:
        raise ValueError("Cluster assignments are not initialized. Warm-up clustering must run first.")

    cluster_states, cluster_members = aggregate_cluster_global_lora(uploads, server.cluster_assignments)
    server.cluster_global_lora_states = cluster_states
    for cluster_id, members in sorted(cluster_members.items()):
        logger.info(
            "[Cluster Aggregation] cluster %d | members=%s",
            cluster_id,
            ", ".join(members),
        )

    next_global_lora_states = build_personalized_cluster_broadcast(
        cluster_states,
        server.cluster_assignments,
    )
    logger.info(
        "[Cluster Aggregation] Broadcast-ready cluster-global LoRA prepared for %d clients",
        len(next_global_lora_states),
    )
    return next_global_lora_states


def broadcast_personalized_global_lora(
    server,
    global_lora_states: Dict[str, Dict[str, torch.Tensor]],
    *,
    phase_title: str = "Server -> Client broadcast of aggregated global LoRA",
) -> None:
    """
    서버가 round aggregation 결과 global LoRA를 각 client에 배포한다.
    """
    _phase_banner(3, phase_title)
    for client_id, client in server.clients.items():
        client.set_global_lora_state(global_lora_states[client_id])
        if client.cluster_id is None:
            logger.info("[Broadcast] Sent global LoRA to %s", client_id)
        else:
            logger.info("[Broadcast] Sent cluster-global LoRA to %s | cluster=%d", client_id, client.cluster_id)


def run_stage2_local_personalization(
    server,
    checkpoints_dir: str,
) -> dict[str, str]:
    _phase_banner(4, "Local alignment and personalization (FedDPA-F Stage 2)")
    metrics_by_client: dict[str, float | None] = {}
    for client_id, client in server.clients.items():
        if client.global_lora_state is None:
            raise ValueError(f"{client_id} is missing assigned cluster-global LoRA for personalization.")

        result = run_local_personalization_round(
            model_id=server.model_id,
            client_id=client_id,
            dataset_name=server.dataset_name,
            aligned_global_state=client.global_lora_state,
            epochs=server.personalization_epochs,
            micro_batch_size=server.micro_batch_size,
            gradient_accumulation_steps=server.gradient_accumulation_steps,
            num_samples=server.num_samples,
            max_length=server.max_length,
            seed=server.seed + 10_000 + int(checkpoint_suffix(client_id)),
            rank=server.rank,
            lora_alpha=server.lora_alpha,
            lora_dropout=server.lora_dropout,
            learning_rate=server.learning_rate,
            train_on_inputs=server.train_on_inputs,
            lora_target_modules=server.lora_target_modules,
        )
        client.local_lora_state = result["local_state"]
        metrics_by_client[client_id] = result["metrics"]["personalization_final_loss"]
        wandb_logger.log(
            {
                f"personalization/{client_id}/final_loss": result["metrics"]["personalization_final_loss"],
                f"personalization/{client_id}/epoch_losses": result["metrics"]["personalization_epoch_losses"],
            },
            step=server.num_rounds,
        )

    saved_paths = _save_personalized_checkpoints(server, checkpoints_dir)
    mean_personalization_loss = mean(
        [value for value in metrics_by_client.values() if value is not None]
    ) if metrics_by_client else None
    wandb_logger.summary_update(
        {
            "personalization/mean_final_loss": mean_personalization_loss,
        }
    )
    return saved_paths


def run_federated_training(server) -> None:
    """Phase 0 warm-up -> phase 1 cluster-wise FL -> phase 2 local personalization orchestration.

    의도한 실험 설계:
    - Phase 0: round 밖에서 warm-up 후 clustering + BA orthogonalization 1회
    - Phase 1: 실제 federated round 1..num_rounds 를 cluster-wise aggregation으로 수행
    - Phase 2: 마지막 round 후 BA orthogonalization 1회, 이후 local LoRA align/personalization
    """
    wandb_logger.init(
        enabled=server.use_wandb,
        project=server.wandb_project,
        run_name=server.wandb_run_name,
        entity=server.wandb_entity,
        config={
            "model_id": server.model_id,
            "dataset_name": server.dataset_name,
            "num_clients": len(server.client_ids),
            "num_rounds": server.num_rounds,
            "local_epochs": server.local_epochs,
            "warmup_epochs": server.warmup_epochs,
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
            "stage1_only": server.stage1_only,
            "cluster_similarity_factor": server.cluster_similarity_factor,
            "cluster_orthogonalization": server.cluster_orthogonalization,
        },
    )

    logger.info("\n" + "=" * 70)
    logger.info("[FL] Initializing common dual LoRA weights and broadcasting to all clients")
    logger.info("=" * 70)
    logger.info("[FL] stage1_only=%s", server.stage1_only)
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
    clustering_artifacts: dict[str, str] = {}

    logger.info("\n" + "=" * 70)
    logger.info("[FL] Phase 0 warm-up (not counted in num_rounds)")
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

    for round_idx in range(1, server.num_rounds + 1):
        logger.info("\n" + "=" * 70)
        logger.info("[FL] Phase 1 cluster-wise round %d/%d", round_idx, server.num_rounds)
        logger.info("=" * 70)

        broadcast_personalized_global_lora(
            server,
            server.global_lora_states,
            phase_title="Cluster-wise broadcast of assigned cluster-global LoRA",
        )
        uploads = collect_client_uploads(
            server,
            round_idx,
            epochs_override=server.local_epochs,
            phase_label="Phase 1 Cluster-wise Training",
        )
        final_round_uploads = uploads
        server.global_lora_states = build_clusterwise_broadcast_state(server, uploads)

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

        _phase_banner(
            4,
            "Client starts next round from the newly aggregated cluster-global LoRA",
        )
        logger.info(
            "[Next Round] Cluster-global LoRA built in this round will be used as the next-round initialization."
        )

    logger.info("\n" + "=" * 70)
    logger.info("[FL] Final phase-1 checkpoint save and analysis")
    logger.info("=" * 70)

    if final_round_uploads is None:
        raise ValueError("No phase-1 round upload payload was collected.")

    final_orthogonalization = {"applied": False, "space": "BA", "phase": "Final Orthogonalization"}
    server.cluster_global_lora_states, final_orthogonalization = _apply_cluster_orthogonalization(
        server,
        server.cluster_global_lora_states,
        phase_label="Final Orthogonalization",
    )
    server.global_lora_states = build_personalized_cluster_broadcast(
        server.cluster_global_lora_states,
        server.cluster_assignments,
    )
    if not server.stage1_only:
        broadcast_personalized_global_lora(
            server,
            server.global_lora_states,
            phase_title="Final orthogonalized cluster-global LoRA broadcast before personalization",
        )

    checkpoints_dir = _build_checkpoint_output_dir()
    os.makedirs(checkpoints_dir, exist_ok=True)
    save_phase1_client_checkpoints_from_uploads(
        clients=server.clients,
        final_uploads=final_round_uploads,
        checkpoints_dir=checkpoints_dir,
        subdir="phase1_client_uploads" if not server.stage1_only else None,
    )

    cluster_checkpoint_paths = _save_cluster_global_checkpoints(
        server.cluster_global_lora_states,
        checkpoints_dir,
    )
    cluster_assignments_path = _save_cluster_assignments_json(
        checkpoints_dir,
        server.cluster_assignments,
        {
            cluster_id: sorted(
                [client_id for client_id, assigned_cluster in server.cluster_assignments.items() if assigned_cluster == cluster_id]
            )
            for cluster_id in sorted(server.cluster_global_lora_states.keys())
        },
    )

    analysis_output_dir = _build_analysis_output_dir(server, server.num_rounds)
    artifact_paths = analyze_upload_payloads(
        uploads=final_round_uploads,
        output_dir=analysis_output_dir,
        source_name=f"{server.dataset_name} | round {server.num_rounds} phase-1 global LoRA",
        methods=[],
    )
    logger.info("[Global LoRA Analysis] Saved artifacts:")
    for name, path in sorted(artifact_paths.items()):
        logger.info("  - %s: %s", name, path)

    final_checkpoint_paths: dict[str, str] = {}
    if server.stage1_only:
        final_checkpoint_paths = {
            client_id: os.path.join(checkpoints_dir, f"dual_lora_adapter_client_{checkpoint_suffix(client_id)}.pth")
            for client_id in server.client_ids
        }
    else:
        final_checkpoint_paths = run_stage2_local_personalization(server, checkpoints_dir)

    wandb_logger.summary_update(
        {
            "artifacts/checkpoints_dir": checkpoints_dir,
            "artifacts/analysis_output_dir": analysis_output_dir,
            "artifacts/cluster_assignments_json": cluster_assignments_path,
            "artifacts/final_checkpoint_count": len(final_checkpoint_paths),
            "cluster/final_orthogonalization_applied": final_orthogonalization.get("applied", False),
            "cluster/final_orthogonalization_space": final_orthogonalization.get("space"),
            "cluster/final_orthogonalization_mean_relative_error": (
                final_orthogonalization.get("approximation_summary", {}).get("mean_relative_error")
            ),
            "cluster/final_orthogonalization_max_relative_error": (
                final_orthogonalization.get("approximation_summary", {}).get("max_relative_error")
            ),
        }
    )
    for cluster_id, path in sorted(cluster_checkpoint_paths.items()):
        wandb_logger.summary_update({f"artifacts/cluster_{cluster_id}_checkpoint": path})
    wandb_logger.log_analysis_artifacts(artifact_paths | clustering_artifacts)
    wandb_logger.finish()
