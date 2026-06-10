from __future__ import annotations

import csv
import json
import logging
import os
import re
from collections import defaultdict
from itertools import combinations
from typing import Dict

import numpy as np
import torch
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform


logger = logging.getLogger(__name__)

ClientGlobalLoRA = Dict[str, Dict[str, Dict[str, torch.Tensor]]]
ClusterAssignments = Dict[str, int]
ClusterGlobalStates = Dict[int, Dict[str, torch.Tensor]]


def _ensure_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for clustering diagnostic figures."
        ) from exc
    return plt


def _extract_layer_index(module_name: str) -> int | None:
    matched = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", module_name)
    if matched:
        return int(matched.group(1))

    matched = re.search(r"(?:^|\.)h\.(\d+)(?:\.|$)", module_name)
    if matched:
        return int(matched.group(1))

    candidates = re.findall(r"(?:^|\.)(\d+)(?:\.|$)", module_name)
    if candidates:
        return int(candidates[0])
    return None


def _safe_cosine(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    flat_a = torch.nan_to_num(vec_a.float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    flat_b = torch.nan_to_num(vec_b.float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    denom = torch.norm(flat_a) * torch.norm(flat_b)
    if float(denom.item()) < 1e-12:
        return 0.0
    cosine = torch.dot(flat_a, flat_b) / denom
    return float(torch.clamp(cosine, min=-1.0, max=1.0).item())


def _factor_matrix(module_payload: Dict[str, torch.Tensor], factor: str) -> torch.Tensor:
    if factor == "B":
        return module_payload["B"]
    if factor == "BA":
        return module_payload["B"].float() @ module_payload["A"].float()
    raise ValueError(f"Unsupported clustering factor: {factor}")


def compute_client_similarity_matrix(
    client_global_lora: ClientGlobalLoRA,
    *,
    factor: str = "B",
) -> tuple[list[str], np.ndarray, list[dict[str, object]]]:
    client_ids = sorted(client_global_lora.keys())
    similarity = np.eye(len(client_ids), dtype=np.float64)
    pair_details: list[dict[str, object]] = []

    for row_idx, client_a in enumerate(client_ids):
        for col_idx, client_b in enumerate(client_ids[row_idx + 1 :], start=row_idx + 1):
            modules_a = client_global_lora[client_a]
            modules_b = client_global_lora[client_b]
            common_modules = sorted(set(modules_a.keys()) & set(modules_b.keys()))
            if not common_modules:
                raise ValueError(f"No common modules found between {client_a} and {client_b}")

            layer_bucket: dict[int, list[float]] = defaultdict(list)
            for module_name in common_modules:
                layer_idx = _extract_layer_index(module_name)
                if layer_idx is None:
                    continue
                cosine = _safe_cosine(
                    _factor_matrix(modules_a[module_name], factor),
                    _factor_matrix(modules_b[module_name], factor),
                )
                layer_bucket[layer_idx].append(cosine)

            if not layer_bucket:
                raise ValueError(f"No valid layer buckets found between {client_a} and {client_b}")

            layers = sorted(layer_bucket.keys())
            layer_scores = [float(np.mean(layer_bucket[layer_idx])) for layer_idx in layers]
            pair_similarity = float(np.mean(layer_scores))
            similarity[row_idx, col_idx] = pair_similarity
            similarity[col_idx, row_idx] = pair_similarity
            pair_details.append(
                {
                    "pair": [client_a, client_b],
                    "layers": layers,
                    "layer_scores": layer_scores,
                    "mean_similarity": pair_similarity,
                    "distance": float(1.0 - pair_similarity),
                }
            )

    return client_ids, similarity, pair_details


def _silhouette_score_precomputed(distance: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = sorted(set(int(label) for label in labels.tolist()))
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return -1.0

    scores: list[float] = []
    for idx in range(len(labels)):
        same_cluster = [j for j in range(len(labels)) if labels[j] == labels[idx] and j != idx]
        if same_cluster:
            a_i = float(np.mean([distance[idx, j] for j in same_cluster]))
        else:
            a_i = 0.0

        other_cluster_means = []
        for other_label in unique_labels:
            if other_label == labels[idx]:
                continue
            members = [j for j in range(len(labels)) if labels[j] == other_label]
            if not members:
                continue
            other_cluster_means.append(float(np.mean([distance[idx, j] for j in members])))

        if not other_cluster_means:
            scores.append(0.0)
            continue

        b_i = min(other_cluster_means)
        denom = max(a_i, b_i)
        scores.append(0.0 if denom < 1e-12 else float((b_i - a_i) / denom))

    return float(np.mean(scores)) if scores else -1.0


def _singleton_safe_silhouette_score_precomputed(distance: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = sorted(set(int(label) for label in labels.tolist()))
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return -1.0

    scores: list[float] = []
    for idx in range(len(labels)):
        same_cluster = [j for j in range(len(labels)) if labels[j] == labels[idx] and j != idx]
        if not same_cluster:
            # Singleton clusters have undefined intra-cluster cohesion.
            # We treat them as neutral instead of artificially rewarding them.
            scores.append(0.0)
            continue

        a_i = float(np.mean([distance[idx, j] for j in same_cluster]))

        other_cluster_means = []
        for other_label in unique_labels:
            if other_label == labels[idx]:
                continue
            members = [j for j in range(len(labels)) if labels[j] == other_label]
            if not members:
                continue
            other_cluster_means.append(float(np.mean([distance[idx, j] for j in members])))

        if not other_cluster_means:
            scores.append(0.0)
            continue

        b_i = min(other_cluster_means)
        denom = max(a_i, b_i)
        scores.append(0.0 if denom < 1e-12 else float((b_i - a_i) / denom))

    return float(np.mean(scores)) if scores else -1.0


def _normalize_cluster_labels(labels: np.ndarray) -> np.ndarray:
    mapping: dict[int, int] = {}
    normalized = []
    next_cluster_id = 1
    for label in labels.tolist():
        int_label = int(label)
        if int_label not in mapping:
            mapping[int_label] = next_cluster_id
            next_cluster_id += 1
        normalized.append(mapping[int_label])
    return np.asarray(normalized, dtype=np.int64)


def _labels_for_candidate_k(linkage_matrix: np.ndarray, num_clients: int, candidate_k: int) -> np.ndarray:
    if candidate_k < 2 or candidate_k >= num_clients:
        raise ValueError(f"candidate_k must be in [2, {num_clients - 1}], got {candidate_k}")

    merge_distances = linkage_matrix[:, 2]
    lower_idx = num_clients - candidate_k - 1
    upper_idx = num_clients - candidate_k
    lower = float(merge_distances[lower_idx])
    upper = float(merge_distances[upper_idx])

    if upper > lower:
        threshold = (lower + upper) / 2.0
        labels = fcluster(linkage_matrix, t=threshold, criterion="distance")
    else:
        labels = fcluster(linkage_matrix, t=candidate_k, criterion="maxclust")

    labels = _normalize_cluster_labels(labels)
    actual_cluster_count = len(set(labels.tolist()))
    if actual_cluster_count != candidate_k:
        logger.warning(
            "[Warm-up Clustering] Requested k=%d but exact distance cut produced %d clusters. "
            "Continuing with the realized partition.",
            candidate_k,
            actual_cluster_count,
        )
    return labels


def _cluster_member_sizes(labels: np.ndarray) -> dict[int, int]:
    sizes: dict[int, int] = defaultdict(int)
    for label in labels.tolist():
        sizes[int(label)] += 1
    return dict(sorted(sizes.items()))


def select_cluster_assignments(
    client_ids: list[str],
    similarity: np.ndarray,
) -> tuple[ClusterAssignments, dict[str, object]]:
    if len(client_ids) < 3:
        raise ValueError("At least 3 clients are required for hierarchical clustering.")

    distance = np.clip(1.0 - similarity, a_min=0.0, a_max=2.0)
    condensed = squareform(distance, checks=False)
    linkage_matrix = linkage(condensed, method="average")

    max_candidate_k = len(client_ids) - 1
    merge_distances = linkage_matrix[:, 2]
    silhouette_by_k: list[dict[str, float]] = []
    gap_by_k: list[dict[str, float]] = []
    elbow_by_k: list[dict[str, float]] = []
    labels_by_k: dict[int, np.ndarray] = {}
    candidate_stats_by_k: list[dict[str, object]] = []
    feasible_candidate_ks: list[int] = []

    for candidate_k in range(2, max_candidate_k + 1):
        labels = _labels_for_candidate_k(linkage_matrix, len(client_ids), candidate_k)
        labels_by_k[candidate_k] = labels
        member_sizes = _cluster_member_sizes(labels)
        min_cluster_size = min(member_sizes.values()) if member_sizes else 0
        has_singleton = min_cluster_size <= 1
        score = _singleton_safe_silhouette_score_precomputed(distance, labels)
        silhouette_by_k.append(
            {
                "k": candidate_k,
                "silhouette": float(score),
                "cluster_count": int(len(set(labels.tolist()))),
                "min_cluster_size": int(min_cluster_size),
                "has_singleton": bool(has_singleton),
            }
        )
        candidate_stats_by_k.append(
            {
                "k": candidate_k,
                "cluster_sizes": member_sizes,
                "min_cluster_size": int(min_cluster_size),
                "has_singleton": bool(has_singleton),
            }
        )
        if not has_singleton:
            feasible_candidate_ks.append(candidate_k)

        gap_score = float(merge_distances[len(client_ids) - candidate_k] - merge_distances[len(client_ids) - candidate_k - 1])
        gap_by_k.append(
            {
                "k": candidate_k,
                "gap": gap_score,
                "min_cluster_size": int(min_cluster_size),
                "has_singleton": bool(has_singleton),
            }
        )

    for candidate_k in range(3, max_candidate_k):
        cut_prev = float(merge_distances[len(client_ids) - (candidate_k - 1) - 1])
        cut_curr = float(merge_distances[len(client_ids) - candidate_k - 1])
        cut_next = float(merge_distances[len(client_ids) - (candidate_k + 1) - 1])
        elbow_score = cut_prev - 2.0 * cut_curr + cut_next
        labels = labels_by_k[candidate_k]
        member_sizes = _cluster_member_sizes(labels)
        min_cluster_size = min(member_sizes.values()) if member_sizes else 0
        elbow_by_k.append(
            {
                "k": candidate_k,
                "elbow": float(elbow_score),
                "min_cluster_size": int(min_cluster_size),
                "has_singleton": bool(min_cluster_size <= 1),
            }
        )

    constrained_gap_by_k = [entry for entry in gap_by_k if not entry["has_singleton"]]
    constrained_elbow_by_k = [entry for entry in elbow_by_k if not entry["has_singleton"]]
    singleton_constraint_applied = bool(constrained_gap_by_k)
    silhouette_lookup = {int(entry["k"]): float(entry["silhouette"]) for entry in silhouette_by_k}

    if not constrained_gap_by_k:
        logger.warning(
            "[Warm-up Clustering] No natural cut satisfied min cluster size >= 2. Falling back to K=1."
        )
        selected_labels = np.ones(len(client_ids), dtype=np.int64)
        selected_k = 1
        selected_silhouette = -1.0
        gap_selected_k = None
        elbow_selected_k = None
        selection_method = "no_valid_non_singleton_cut_fallback_k1"
    else:
        gap_selected = max(constrained_gap_by_k, key=lambda item: (item["gap"], -item["k"]))
        gap_selected_k = int(gap_selected["k"])

        if constrained_elbow_by_k:
            elbow_selected = max(constrained_elbow_by_k, key=lambda item: (item["elbow"], -item["k"]))
            elbow_selected_k = int(elbow_selected["k"])
        else:
            elbow_selected_k = gap_selected_k

        selection_method = "gap_elbow_agreement"
        if gap_selected_k == elbow_selected_k:
            selected_k = gap_selected_k
        else:
            selection_method = "singleton_safe_silhouette_tiebreak"
            candidate_pair = sorted({gap_selected_k, elbow_selected_k})
            selected_k = max(
                candidate_pair,
                key=lambda k: (silhouette_lookup[k], -k),
            )

        selected_labels = labels_by_k[selected_k]
        selected_silhouette = silhouette_lookup[selected_k]

    assignments = {
        client_id: int(label)
        for client_id, label in zip(client_ids, selected_labels.tolist())
    }
    members: dict[int, list[str]] = defaultdict(list)
    for client_id, cluster_id in assignments.items():
        members[cluster_id].append(client_id)

    diagnostics = {
        "linkage_matrix": linkage_matrix,
        "distance_matrix": distance,
        "silhouette_by_k": silhouette_by_k,
        "silhouette_variant": "singleton_safe",
        "gap_by_k": gap_by_k,
        "elbow_by_k": elbow_by_k,
        "candidate_stats_by_k": candidate_stats_by_k,
        "singleton_constraint": {
            "enabled": True,
            "min_cluster_size": 2,
            "feasible_candidate_ks": feasible_candidate_ks,
            "applied_without_fallback": singleton_constraint_applied,
        },
        "gap_selected_k": gap_selected_k,
        "elbow_selected_k": elbow_selected_k,
        "selection_method": selection_method,
        "selected_k": selected_k,
        "selected_silhouette": float(selected_silhouette),
        "cluster_members": {int(k): sorted(v) for k, v in sorted(members.items())},
    }
    return assignments, diagnostics


def aggregate_cluster_global_lora(
    uploads: Dict[str, Dict],
    assignments: ClusterAssignments,
) -> tuple[ClusterGlobalStates, dict[int, list[str]]]:
    members: dict[int, list[str]] = defaultdict(list)
    for client_id, cluster_id in assignments.items():
        members[cluster_id].append(client_id)

    cluster_states: ClusterGlobalStates = {}
    for cluster_id, client_ids in sorted(members.items()):
        reference_client = sorted(client_ids)[0]
        module_names = sorted(uploads[reference_client]["global_lora"].keys())
        state: Dict[str, torch.Tensor] = {}
        for module_name in module_names:
            a_matrices = [
                uploads[client_id]["global_lora"][module_name]["A"].float()
                for client_id in client_ids
                if module_name in uploads[client_id].get("global_lora", {})
            ]
            b_matrices = [
                uploads[client_id]["global_lora"][module_name]["B"].float()
                for client_id in client_ids
                if module_name in uploads[client_id].get("global_lora", {})
            ]
            if not a_matrices or not b_matrices:
                continue
            state[f"{module_name}.global_lora.lora_A.weight"] = torch.stack(a_matrices, dim=0).mean(dim=0).cpu()
            state[f"{module_name}.global_lora.lora_B.weight"] = torch.stack(b_matrices, dim=0).mean(dim=0).cpu()
        cluster_states[cluster_id] = state
    return cluster_states, {int(k): sorted(v) for k, v in sorted(members.items())}


def _factorize_delta_to_lora(
    delta_matrix: torch.Tensor,
    *,
    original_b: torch.Tensor,
    original_a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    """
    Orthogonalized Delta W를 기존 LoRA rank에 맞는 B, A로 다시 근사한다.
    """
    if delta_matrix.ndim != 2:
        raise ValueError(f"delta_matrix must be 2D, got shape={tuple(delta_matrix.shape)}")

    target_rank = int(original_b.shape[1])
    if target_rank != int(original_a.shape[0]):
        raise ValueError(
            "LoRA A/B rank mismatch: "
            f"B shape={tuple(original_b.shape)}, A shape={tuple(original_a.shape)}"
        )

    u_matrix, singular_values, vh_matrix = torch.linalg.svd(delta_matrix, full_matrices=False)
    effective_rank = min(target_rank, int(singular_values.numel()))

    ref_b = original_b.float()
    ref_a = original_a.float()
    factorized_b = torch.zeros_like(ref_b)
    factorized_a = torch.zeros_like(ref_a)

    if effective_rank > 0:
        scaled_left = u_matrix[:, :effective_rank] * singular_values[:effective_rank].unsqueeze(0)
        factorized_b[:, :effective_rank] = scaled_left
        factorized_a[:effective_rank, :] = vh_matrix[:effective_rank, :]

    reconstructed = factorized_b @ factorized_a
    delta_norm = torch.norm(delta_matrix)
    relative_error = 0.0
    if float(delta_norm.item()) >= 1e-12:
        relative_error = float((torch.norm(delta_matrix - reconstructed) / delta_norm).item())

    return factorized_b.cpu(), factorized_a.cpu(), relative_error, effective_rank


def orthogonalize_cluster_deltaw_states(
    cluster_states: ClusterGlobalStates,
) -> tuple[ClusterGlobalStates, dict[str, object]]:
    if len(cluster_states) <= 1:
        return cluster_states, {"applied": False, "reason": "single_cluster", "space": "BA"}

    orthogonalized = {
        cluster_id: {
            key: value.detach().cpu().clone()
            for key, value in state.items()
        }
        for cluster_id, state in cluster_states.items()
    }

    cluster_ids = sorted(orthogonalized.keys())
    reference_state = orthogonalized[cluster_ids[0]]
    module_names = sorted(
        key.split(".global_lora.lora_A.weight")[0]
        for key in reference_state.keys()
        if key.endswith(".global_lora.lora_A.weight")
    )

    degenerate_pairs: list[dict[str, object]] = []
    approximation_by_module: list[dict[str, object]] = []

    for module_name in module_names:
        basis: list[torch.Tensor] = []
        for cluster_id in cluster_ids:
            a_key = f"{module_name}.global_lora.lora_A.weight"
            b_key = f"{module_name}.global_lora.lora_B.weight"
            if a_key not in orthogonalized[cluster_id] or b_key not in orthogonalized[cluster_id]:
                continue

            original_a = orthogonalized[cluster_id][a_key].float()
            original_b = orthogonalized[cluster_id][b_key].float()
            delta_matrix = original_b @ original_a
            delta_vector = delta_matrix.reshape(-1)
            original_norm = torch.norm(delta_vector)
            residual = delta_vector.clone()
            for basis_vector in basis:
                residual = residual - torch.dot(residual, basis_vector) * basis_vector
            residual_norm = torch.norm(residual)

            if float(residual_norm.item()) < 1e-12:
                degenerate_pairs.append({"cluster_id": cluster_id, "module_name": module_name})
                orth_delta = delta_matrix
            else:
                orth_delta = (residual / residual_norm * torch.clamp(original_norm, min=1e-12)).reshape_as(delta_matrix)

            factorized_b, factorized_a, relative_error, effective_rank = _factorize_delta_to_lora(
                orth_delta,
                original_b=original_b,
                original_a=original_a,
            )
            orthogonalized[cluster_id][b_key] = factorized_b.to(dtype=orthogonalized[cluster_id][b_key].dtype).cpu()
            orthogonalized[cluster_id][a_key] = factorized_a.to(dtype=orthogonalized[cluster_id][a_key].dtype).cpu()

            reconstructed_delta = factorized_b.float() @ factorized_a.float()
            reconstructed_vector = reconstructed_delta.reshape(-1)
            reconstructed_norm = torch.norm(reconstructed_vector)
            if float(reconstructed_norm.item()) >= 1e-12:
                basis.append((reconstructed_vector / reconstructed_norm).cpu())

            approximation_by_module.append(
                {
                    "cluster_id": cluster_id,
                    "module_name": module_name,
                    "relative_error": relative_error,
                    "effective_rank": effective_rank,
                }
            )

    relative_errors = [entry["relative_error"] for entry in approximation_by_module]
    diagnostics = {
        "applied": True,
        "space": "BA",
        "degenerate_modules": degenerate_pairs,
        "approximation_by_module": approximation_by_module,
        "approximation_summary": {
            "count": len(relative_errors),
            "mean_relative_error": float(np.mean(relative_errors)) if relative_errors else 0.0,
            "max_relative_error": float(np.max(relative_errors)) if relative_errors else 0.0,
        },
    }
    return orthogonalized, diagnostics


def build_personalized_cluster_broadcast(
    cluster_states: ClusterGlobalStates,
    assignments: ClusterAssignments,
) -> Dict[str, Dict[str, torch.Tensor]]:
    return {
        client_id: {
            key: value.detach().cpu().clone()
            for key, value in cluster_states[cluster_id].items()
        }
        for client_id, cluster_id in assignments.items()
    }


def _save_matrix_csv(output_path: str, labels: list[str], matrix: np.ndarray) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["id", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[f"{value:.6f}" for value in row]])


def _save_dendrogram(output_path: str, labels: list[str], linkage_matrix: np.ndarray, title: str) -> None:
    plt = _ensure_matplotlib()
    fig, axis = plt.subplots(figsize=(9, 5))
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=45, ax=axis)
    axis.set_title(title)
    axis.set_ylabel("distance = 1 - similarity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_silhouette_curve(output_path: str, silhouette_by_k: list[dict[str, float]], title: str) -> None:
    plt = _ensure_matplotlib()
    fig, axis = plt.subplots(figsize=(7, 4.5))
    x_values = [entry["k"] for entry in silhouette_by_k]
    y_values = [entry["silhouette"] for entry in silhouette_by_k]
    axis.plot(x_values, y_values, marker="o", linewidth=1.8)
    axis.set_title(title)
    axis.set_xlabel("Number of clusters (k)")
    axis.set_ylabel("Silhouette score")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_merge_distance_curve(output_path: str, linkage_matrix: np.ndarray, title: str) -> None:
    plt = _ensure_matplotlib()
    fig, axis = plt.subplots(figsize=(7, 4.5))
    merge_steps = list(range(1, linkage_matrix.shape[0] + 1))
    merge_distances = linkage_matrix[:, 2].tolist()
    axis.plot(merge_steps, merge_distances, marker="o", linewidth=1.8)
    axis.set_title(title)
    axis.set_xlabel("Merge step")
    axis.set_ylabel("Average-linkage merge distance")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_clustering_artifacts(
    *,
    output_dir: str,
    client_ids: list[str],
    similarity: np.ndarray,
    pair_details: list[dict[str, object]],
    assignments: ClusterAssignments,
    diagnostics: dict[str, object],
    factor: str,
    orthogonalization: dict[str, object] | None = None,
    source_name: str = "warmup clustering",
) -> dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    artifact_paths: dict[str, str] = {}

    similarity_csv = os.path.join(output_dir, "cluster_similarity_matrix.csv")
    distance_csv = os.path.join(output_dir, "cluster_distance_matrix.csv")
    summary_json = os.path.join(output_dir, "cluster_summary.json")
    pairwise_json = os.path.join(output_dir, "cluster_pair_details.json")

    _save_matrix_csv(similarity_csv, client_ids, similarity)
    _save_matrix_csv(distance_csv, client_ids, diagnostics["distance_matrix"])

    with open(pairwise_json, "w", encoding="utf-8") as file_obj:
        json.dump(pair_details, file_obj, indent=2, ensure_ascii=False)

    summary_payload = {
        "source_name": source_name,
        "factor": factor,
        "selected_k": diagnostics["selected_k"],
        "selected_silhouette": diagnostics["selected_silhouette"],
        "silhouette_variant": diagnostics.get("silhouette_variant", "singleton_safe"),
        "gap_selected_k": diagnostics.get("gap_selected_k"),
        "elbow_selected_k": diagnostics.get("elbow_selected_k"),
        "selection_method": diagnostics.get("selection_method"),
        "singleton_constraint": diagnostics.get("singleton_constraint", {}),
        "assignments": assignments,
        "cluster_members": diagnostics["cluster_members"],
        "silhouette_by_k": diagnostics["silhouette_by_k"],
        "gap_by_k": diagnostics.get("gap_by_k", []),
        "elbow_by_k": diagnostics.get("elbow_by_k", []),
        "candidate_stats_by_k": diagnostics.get("candidate_stats_by_k", []),
        "orthogonalization": orthogonalization or {"applied": False},
    }
    with open(summary_json, "w", encoding="utf-8") as file_obj:
        json.dump(summary_payload, file_obj, indent=2, ensure_ascii=False)

    artifact_paths["cluster_similarity_csv"] = similarity_csv
    artifact_paths["cluster_distance_csv"] = distance_csv
    artifact_paths["cluster_summary_json"] = summary_json
    artifact_paths["cluster_pair_details_json"] = pairwise_json

    try:
        dendrogram_png = os.path.join(output_dir, "cluster_dendrogram.png")
        silhouette_png = os.path.join(output_dir, "cluster_silhouette_curve.png")
        elbow_png = os.path.join(output_dir, "cluster_merge_distance_curve.png")
        _save_dendrogram(
            dendrogram_png,
            client_ids,
            diagnostics["linkage_matrix"],
            title=f"{source_name} | {factor} average-linkage dendrogram",
        )
        _save_silhouette_curve(
            silhouette_png,
            diagnostics["silhouette_by_k"],
            title=f"{source_name} | silhouette by k",
        )
        _save_merge_distance_curve(
            elbow_png,
            diagnostics["linkage_matrix"],
            title=f"{source_name} | merge distance curve",
        )
        artifact_paths["cluster_dendrogram_png"] = dendrogram_png
        artifact_paths["cluster_silhouette_curve_png"] = silhouette_png
        artifact_paths["cluster_merge_distance_curve_png"] = elbow_png
    except ModuleNotFoundError as exc:
        logger.warning("[Warm-up Clustering] %s", exc)

    return artifact_paths
