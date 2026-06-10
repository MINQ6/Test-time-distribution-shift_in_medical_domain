from __future__ import annotations

import csv
import json
import logging
import os
import re
from itertools import combinations
from typing import Dict

import numpy as np
import torch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


ClientGlobalLoRA = Dict[str, Dict[str, Dict[str, torch.Tensor]]]


def _sanitize_factor_tensor(
    tensor: torch.Tensor,
    *,
    client_id: str,
    module_name: str,
    factor_name: str,
) -> tuple[torch.Tensor, bool]:
    sanitized = tensor.detach().cpu().clone()
    finite_mask = torch.isfinite(sanitized)
    if bool(finite_mask.all()):
        return sanitized, False

    invalid_count = int((~finite_mask).sum().item())
    logger.warning(
        "[Global LoRA Analysis] %s | %s | %s had %d non-finite values. Replacing them with 0.0.",
        client_id,
        module_name,
        factor_name,
        invalid_count,
    )
    sanitized = torch.nan_to_num(sanitized, nan=0.0, posinf=0.0, neginf=0.0)
    return sanitized, True


def _sanitize_client_global_lora(client_global_lora: ClientGlobalLoRA) -> ClientGlobalLoRA:
    sanitized_clients: ClientGlobalLoRA = {}
    for client_id, modules in client_global_lora.items():
        sanitized_modules: Dict[str, Dict[str, torch.Tensor]] = {}
        for module_name, payload in modules.items():
            if "A" not in payload or "B" not in payload:
                logger.warning(
                    "[Global LoRA Analysis] %s | %s is missing A/B factors and will be skipped.",
                    client_id,
                    module_name,
                )
                continue
            sanitized_a, _ = _sanitize_factor_tensor(
                payload["A"],
                client_id=client_id,
                module_name=module_name,
                factor_name="A",
            )
            sanitized_b, _ = _sanitize_factor_tensor(
                payload["B"],
                client_id=client_id,
                module_name=module_name,
                factor_name="B",
            )
            sanitized_modules[module_name] = {"A": sanitized_a, "B": sanitized_b}
        sanitized_clients[client_id] = sanitized_modules
    return sanitized_clients


def extract_global_lora_from_uploads(uploads: Dict[str, Dict]) -> ClientGlobalLoRA:
    client_global_lora: ClientGlobalLoRA = {}
    for client_id, payload in uploads.items():
        global_lora = payload.get("global_lora", {})
        client_global_lora[client_id] = {
            module_name: {
                "A": module_payload["A"].detach().cpu().clone(),
                "B": module_payload["B"].detach().cpu().clone(),
            }
            for module_name, module_payload in global_lora.items()
        }
    return _sanitize_client_global_lora(client_global_lora)


def extract_global_lora_from_checkpoints(checkpoint_dir: str) -> ClientGlobalLoRA:
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    logger.info("[Global LoRA Analysis] Loading checkpoints from %s", checkpoint_dir)
    client_global_lora: ClientGlobalLoRA = {}
    for filename in sorted(os.listdir(checkpoint_dir)):
        if not filename.startswith("dual_lora_adapter_client_") or not filename.endswith(".pth"):
            continue

        client_suffix = filename[len("dual_lora_adapter_client_") : -len(".pth")]
        client_id = f"client_{client_suffix}"
        state_dict = torch.load(
            os.path.join(checkpoint_dir, filename),
            map_location="cpu",
            weights_only=False,
        )

        modules: Dict[str, Dict[str, torch.Tensor]] = {}
        for key, value in state_dict.items():
            if ".global_lora.lora_A.weight" in key:
                module_name = key.split(".global_lora.lora_A.weight")[0]
                modules.setdefault(module_name, {})["A"] = value.detach().cpu().clone()
            elif ".global_lora.lora_B.weight" in key:
                module_name = key.split(".global_lora.lora_B.weight")[0]
                modules.setdefault(module_name, {})["B"] = value.detach().cpu().clone()

        client_global_lora[client_id] = modules
        logger.info(
            "[Global LoRA Analysis] Loaded %s with %d global LoRA modules",
            client_id,
            len(modules),
        )

    if not client_global_lora:
        raise ValueError(f"No dual LoRA checkpoints found in {checkpoint_dir}")

    return _sanitize_client_global_lora(client_global_lora)


def _module_delta_w(module_payload: Dict[str, torch.Tensor]) -> torch.Tensor:
    delta_w = module_payload["B"].float() @ module_payload["A"].float()
    if not bool(torch.isfinite(delta_w).all()):
        delta_w = torch.nan_to_num(delta_w, nan=0.0, posinf=0.0, neginf=0.0)
    return delta_w


def _top1_vector_from_delta(module_payload: Dict[str, torch.Tensor]) -> torch.Tensor:
    delta_w = _module_delta_w(module_payload)
    left_vectors, _, _ = torch.linalg.svd(delta_w, full_matrices=False)
    top1 = left_vectors[:, 0]
    return top1 / torch.clamp(torch.norm(top1), min=1e-12)


def _top1_vector_from_qr(module_payload: Dict[str, torch.Tensor]) -> torch.Tensor:
    basis, _ = torch.linalg.qr(torch.nan_to_num(module_payload["B"].float(), nan=0.0, posinf=0.0, neginf=0.0), mode="reduced")
    top1 = basis[:, 0]
    return top1 / torch.clamp(torch.norm(top1), min=1e-12)


def _build_client_module_vectors(
    client_global_lora: ClientGlobalLoRA,
    method: str,
) -> Dict[str, Dict[str, torch.Tensor]]:
    if method not in {"svd", "qr"}:
        raise ValueError(f"Unsupported method: {method}")

    vector_fn = _top1_vector_from_delta if method == "svd" else _top1_vector_from_qr
    client_vectors: Dict[str, Dict[str, torch.Tensor]] = {}
    for client_id, modules in client_global_lora.items():
        logger.info(
            "[Global LoRA Analysis] %s | building top-1 vectors for %s (%d modules)",
            method.upper(),
            client_id,
            len(modules),
        )
        client_vectors[client_id] = {}
        for idx, (module_name, module_payload) in enumerate(modules.items(), start=1):
            client_vectors[client_id][module_name] = vector_fn(module_payload)
            if idx == 1 or idx % 25 == 0 or idx == len(modules):
                logger.info(
                    "[Global LoRA Analysis] %s | %s | module %d/%d | %s",
                    method.upper(),
                    client_id,
                    idx,
                    len(modules),
                    module_name,
                )
    return client_vectors


def _normalize_methods(methods: list[str] | tuple[str, ...] | None) -> list[str]:
    if methods is None:
        return ["svd", "qr"]
    if len(methods) == 0:
        return []

    normalized: list[str] = []
    for method in methods:
        lowered = method.lower().strip()
        if lowered == "both":
            for candidate in ("svd", "qr"):
                if candidate not in normalized:
                    normalized.append(candidate)
            continue
        if lowered not in {"svd", "qr"}:
            raise ValueError(f"Unsupported analysis method: {method}")
        if lowered not in normalized:
            normalized.append(lowered)

    if not normalized:
        raise ValueError("At least one analysis method must be provided.")
    return normalized


def _safe_abs_cosine(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    if vec_a.shape != vec_b.shape:
        raise ValueError(f"Vector shape mismatch: {tuple(vec_a.shape)} vs {tuple(vec_b.shape)}")
    cosine = torch.dot(vec_a, vec_b) / torch.clamp(torch.norm(vec_a) * torch.norm(vec_b), min=1e-12)
    return float(torch.clamp(torch.abs(cosine), min=0.0, max=1.0).item())


def _pairwise_similarity_from_module_vectors(
    client_vectors: Dict[str, Dict[str, torch.Tensor]],
) -> tuple[list[str], np.ndarray, dict[tuple[str, str], dict[str, float]]]:
    client_ids = sorted(client_vectors.keys())
    logger.info(
        "[Global LoRA Analysis] Computing pairwise cosine similarity across %d clients",
        len(client_ids),
    )
    similarity = np.eye(len(client_ids), dtype=np.float64)
    pair_stats: dict[tuple[str, str], dict[str, float]] = {}

    for row_idx, client_a in enumerate(client_ids):
        for col_idx, client_b in enumerate(client_ids[row_idx + 1 :], start=row_idx + 1):
            common_modules = sorted(set(client_vectors[client_a]) & set(client_vectors[client_b]))
            if not common_modules:
                raise ValueError(f"No common modules found between {client_a} and {client_b}")

            module_scores = [
                _safe_abs_cosine(client_vectors[client_a][module_name], client_vectors[client_b][module_name])
                for module_name in common_modules
            ]
            mean_score = float(np.mean(module_scores))
            similarity[row_idx, col_idx] = mean_score
            similarity[col_idx, row_idx] = mean_score
            pair_stats[(client_a, client_b)] = {
                "mean": mean_score,
                "min": float(np.min(module_scores)),
                "max": float(np.max(module_scores)),
                "std": float(np.std(module_scores)),
                "num_modules": len(common_modules),
            }
            logger.info(
                "[Global LoRA Analysis] pair %s vs %s | modules=%d | mean=%.4f",
                client_a,
                client_b,
                len(common_modules),
                mean_score,
            )

    return client_ids, similarity, pair_stats


def _ensure_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for heatmap/dendrogram generation. "
            "Install it from requirements.txt before running the analysis."
        ) from exc
    return plt


def _save_similarity_csv(output_path: str, client_ids: list[str], similarity: np.ndarray) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["client_id", *client_ids])
        for client_id, row in zip(client_ids, similarity):
            writer.writerow([client_id, *[f"{value:.6f}" for value in row]])


def _save_heatmap(output_path: str, client_ids: list[str], similarity: np.ndarray, title: str) -> None:
    plt = _ensure_matplotlib()
    fig, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(similarity, cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(client_ids)))
    axis.set_xticklabels(client_ids, rotation=45, ha="right")
    axis.set_yticks(range(len(client_ids)))
    axis.set_yticklabels(client_ids)
    axis.set_title(title)

    for row_idx in range(len(client_ids)):
        for col_idx in range(len(client_ids)):
            axis.text(
                col_idx,
                row_idx,
                f"{similarity[row_idx, col_idx]:.2f}",
                ha="center",
                va="center",
                color="white" if similarity[row_idx, col_idx] < 0.6 else "black",
                fontsize=8,
            )

    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="absolute cosine similarity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_dendrogram(output_path: str, client_ids: list[str], similarity: np.ndarray, title: str) -> None:
    plt = _ensure_matplotlib()
    distance = np.clip(1.0 - similarity, a_min=0.0, a_max=1.0)
    condensed = squareform(distance, checks=False)
    linkage_matrix = linkage(condensed, method="average")

    fig, axis = plt.subplots(figsize=(9, 5))
    dendrogram(linkage_matrix, labels=client_ids, leaf_rotation=45, ax=axis)
    axis.set_title(title)
    axis.set_ylabel("distance = 1 - similarity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_summary(
    output_path: str,
    client_ids: list[str],
    similarity: np.ndarray,
    pair_stats: dict[tuple[str, str], dict[str, float]],
    method: str,
) -> None:
    off_diagonal = [
        {
            "pair": [client_a, client_b],
            **stats,
        }
        for (client_a, client_b), stats in sorted(pair_stats.items())
    ]

    closest_to_zero = min(off_diagonal, key=lambda item: item["mean"]) if off_diagonal else None
    highest_similarity = max(off_diagonal, key=lambda item: item["mean"]) if off_diagonal else None

    summary = {
        "method": method,
        "client_ids": client_ids,
        "mean_off_diagonal_similarity": float(
            np.mean([item["mean"] for item in off_diagonal]) if off_diagonal else 1.0
        ),
        "min_off_diagonal_similarity": float(
            np.min([item["mean"] for item in off_diagonal]) if off_diagonal else 1.0
        ),
        "max_off_diagonal_similarity": float(
            np.max([item["mean"] for item in off_diagonal]) if off_diagonal else 1.0
        ),
        "closest_to_zero_pair": closest_to_zero,
        "highest_similarity_pair": highest_similarity,
        "pair_statistics": off_diagonal,
    }

    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, ensure_ascii=False)


def _save_client_vectors(
    output_path: str,
    client_vectors: Dict[str, Dict[str, torch.Tensor]],
) -> None:
    serializable = {
        client_id: {
            module_name: vector.detach().cpu()
            for module_name, vector in modules.items()
        }
        for client_id, modules in client_vectors.items()
    }
    torch.save(serializable, output_path)


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


def _display_client_name(raw_name: str) -> str:
    matched = re.fullmatch(r"client_(\d+)", raw_name)
    if matched:
        return f"client{matched.group(1)}"
    return raw_name


def _safe_abs_cosine_matrix(matrix_a: torch.Tensor, matrix_b: torch.Tensor) -> float:
    flat_a = matrix_a.float().reshape(-1)
    flat_b = matrix_b.float().reshape(-1)
    return _safe_abs_cosine(flat_a, flat_b)


def _flatten_vector_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(matrix.float(), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)


def _top1_left_singular_vector_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    safe_matrix = torch.nan_to_num(matrix.float(), nan=0.0, posinf=0.0, neginf=0.0)
    left_vectors, _, _ = torch.linalg.svd(safe_matrix, full_matrices=False)
    top1 = left_vectors[:, 0]
    return top1 / torch.clamp(torch.norm(top1), min=1e-12)


def _top1_q_vector_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    safe_matrix = torch.nan_to_num(matrix.float(), nan=0.0, posinf=0.0, neginf=0.0)
    q_basis, _ = torch.linalg.qr(safe_matrix, mode="reduced")
    top1 = q_basis[:, 0]
    return top1 / torch.clamp(torch.norm(top1), min=1e-12)


def _representation_vector_from_matrix(matrix: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "flatten":
        return _flatten_vector_from_matrix(matrix)
    if mode == "svd":
        return _top1_left_singular_vector_from_matrix(matrix)
    if mode == "qr":
        return _top1_q_vector_from_matrix(matrix)
    raise ValueError(f"Unsupported layerwise similarity mode: {mode}")


def _compute_pairwise_layerwise_factor_similarity(
    client_global_lora: ClientGlobalLoRA,
    *,
    mode: str,
) -> dict[str, object]:
    client_ids = sorted(client_global_lora.keys())
    pair_entries: list[dict[str, object]] = []
    all_layers: set[int] = set()
    client_pairs = list(combinations(client_ids, 2))

    logger.info(
        "[Global LoRA Analysis] %s layer-wise similarity start | clients=%d | pairs=%d",
        mode.upper(),
        len(client_ids),
        len(client_pairs),
    )

    for pair_idx, (client_a, client_b) in enumerate(client_pairs, start=1):
        modules_a = client_global_lora[client_a]
        modules_b = client_global_lora[client_b]
        common_modules = sorted(set(modules_a) & set(modules_b))
        if not common_modules:
            logger.warning(
                "[Global LoRA Analysis] %s pair %d/%d | %s vs %s has no common modules. Skipping.",
                mode.upper(),
                pair_idx,
                len(client_pairs),
                client_a,
                client_b,
            )
            continue

        if pair_idx == 1 or pair_idx == len(client_pairs) or pair_idx % 5 == 0:
            logger.info(
                "[Global LoRA Analysis] %s pair %d/%d | %s vs %s | common_modules=%d",
                mode.upper(),
                pair_idx,
                len(client_pairs),
                client_a,
                client_b,
                len(common_modules),
            )

        bucket: dict[int, dict[str, list[float]]] = {}
        for module_name in common_modules:
            layer_idx = _extract_layer_index(module_name)
            if layer_idx is None:
                continue

            payload_a = modules_a[module_name]
            payload_b = modules_b[module_name]

            matrix_a_a = payload_a["A"]
            matrix_a_b = payload_a["B"]
            matrix_a_ba = payload_a["B"].float() @ payload_a["A"].float()
            matrix_b_a = payload_b["A"]
            matrix_b_b = payload_b["B"]
            matrix_b_ba = payload_b["B"].float() @ payload_b["A"].float()

            sim_a = _safe_abs_cosine(
                _representation_vector_from_matrix(matrix_a_a, mode),
                _representation_vector_from_matrix(matrix_b_a, mode),
            )
            sim_b = _safe_abs_cosine(
                _representation_vector_from_matrix(matrix_a_b, mode),
                _representation_vector_from_matrix(matrix_b_b, mode),
            )
            sim_ba = _safe_abs_cosine(
                _representation_vector_from_matrix(matrix_a_ba, mode),
                _representation_vector_from_matrix(matrix_b_ba, mode),
            )

            layer_bucket = bucket.setdefault(layer_idx, {"A": [], "B": [], "BA": []})
            layer_bucket["A"].append(sim_a)
            layer_bucket["B"].append(sim_b)
            layer_bucket["BA"].append(sim_ba)

        if not bucket:
            logger.warning(
                "[Global LoRA Analysis] %s pair %d/%d | %s vs %s produced no valid layer buckets.",
                mode.upper(),
                pair_idx,
                len(client_pairs),
                client_a,
                client_b,
            )
            continue

        layers = sorted(bucket.keys())
        all_layers.update(layers)
        a_scores = [float(np.mean(bucket[layer]["A"])) for layer in layers]
        b_scores = [float(np.mean(bucket[layer]["B"])) for layer in layers]
        ba_scores = [float(np.mean(bucket[layer]["BA"])) for layer in layers]

        pair_entries.append(
            {
                "pair": [client_a, client_b],
                "pair_label": f"{_display_client_name(client_a)} vs {_display_client_name(client_b)}",
                "layers": layers,
                "A": a_scores,
                "B": b_scores,
                "BA": ba_scores,
                "avg_A": float(np.mean(a_scores)),
                "avg_B": float(np.mean(b_scores)),
                "avg_BA": float(np.mean(ba_scores)),
            }
        )

    logger.info(
        "[Global LoRA Analysis] %s layer-wise similarity done | pairs_with_stats=%d | layers=%d",
        mode.upper(),
        len(pair_entries),
        len(all_layers),
    )

    return {
        "mode": mode,
        "client_ids": client_ids,
        "layers": sorted(all_layers),
        "pairs": pair_entries,
    }


def _save_layerwise_similarity_csv(output_path: str, stats: dict[str, object]) -> None:
    pairs = stats.get("pairs", [])
    with open(output_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["pair", "layer", "sim_A", "sim_B", "sim_BA"])
        for entry in pairs:
            pair_label = entry["pair_label"]
            layers = entry["layers"]
            a_scores = entry["A"]
            b_scores = entry["B"]
            ba_scores = entry["BA"]
            for layer, score_a, score_b, score_ba in zip(layers, a_scores, b_scores, ba_scores):
                writer.writerow([pair_label, layer, f"{score_a:.6f}", f"{score_b:.6f}", f"{score_ba:.6f}"])


def _draw_panel(
    axis,
    pairs: list[dict[str, object]],
    matrix_key: str,
    title: str,
    colors,
) -> None:
    axis.set_facecolor("#f0f0f0")
    axis.grid(True, color="#b9b9b9", alpha=0.35, linewidth=0.8)

    for idx, entry in enumerate(pairs):
        color = colors[idx % len(colors)]
        label = entry["pair_label"]
        axis.plot(
            entry["layers"],
            entry[matrix_key],
            label=label,
            color=color,
            linewidth=1.2,
            marker="o",
            markersize=2.8,
        )

    axis.set_title(title, fontsize=12, fontweight="bold")
    axis.set_xlabel("Layer")
    axis.set_ylabel("Cosine Similarity")
    axis.set_ylim(0.0, 1.02)


def _legend_labels(pairs: list[dict[str, object]]) -> list[str]:
    return [str(entry["pair_label"]) for entry in pairs]


def _avg_table_lines(pairs: list[dict[str, object]]) -> list[str]:
    lines = []
    for entry in pairs:
        lines.append(
            f"{entry['pair_label']:<28}  "
            f"{entry['avg_A']:>7.4f}  "
            f"{entry['avg_B']:>7.4f}  "
            f"{entry['avg_BA']:>7.4f}"
        )
    return lines


def _save_legend_only_png(output_path: str, pairs: list[dict[str, object]], colors) -> None:
    plt = _ensure_matplotlib()
    fig, axis = plt.subplots(figsize=(4.8, max(6.0, 0.32 * len(pairs) + 0.8)))
    axis.axis("off")
    handles = [
        plt.Line2D([0], [0], color=colors[idx % len(colors)], linewidth=1.8, marker="o", markersize=3.2)
        for idx in range(len(pairs))
    ]
    axis.legend(
        handles,
        _legend_labels(pairs),
        loc="upper left",
        fontsize=8.5,
        framealpha=0.95,
        ncol=1,
        borderaxespad=0.0,
        handlelength=2.0,
        handletextpad=0.6,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_avg_only_png(output_path: str, pairs: list[dict[str, object]]) -> None:
    plt = _ensure_matplotlib()
    lines = _avg_table_lines(pairs)
    fig, axis = plt.subplots(figsize=(8.6, max(6.0, 0.28 * len(lines) + 0.8)))
    axis.axis("off")
    axis.text(
        0.0,
        1.0,
        "\n".join(lines),
        fontsize=8.1,
        ha="left",
        va="top",
        fontweight="bold",
        linespacing=1.10,
        family="monospace",
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _render_global_footer(legend_axis, avg_axis, plot_axis, pairs: list[dict[str, object]]) -> None:
    legend_axis.axis("off")
    avg_axis.axis("off")

    handles, labels = plot_axis.get_legend_handles_labels()
    legend_axis.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.00, 1.00),
        fontsize=8.0,
        framealpha=0.95,
        ncol=1,
        borderaxespad=0.0,
        handlelength=1.9,
        handletextpad=0.5,
        columnspacing=0.6,
    )

    avg_axis.text(
        0.0,
        1.0,
        "\n".join(_avg_table_lines(pairs)),
        fontsize=8.0,
        ha="left",
        va="top",
        fontweight="bold",
        linespacing=1.10,
        family="monospace",
    )


def _save_layerwise_similarity_plot(output_path: str, stats: dict[str, object], source_name: str) -> None:
    pairs = stats.get("pairs", [])
    if not pairs:
        return

    plt = _ensure_matplotlib()
    footer_height = max(6.8, 1.1 + 0.28 * len(pairs))
    panel_height = 4.5 + footer_height
    fig = plt.figure(figsize=(16.5, panel_height))
    grid = fig.add_gridspec(2, 1, height_ratios=[4.5, footer_height], hspace=0.18)
    plot_grid = grid[0].subgridspec(1, 3, wspace=0.28)
    footer_grid = grid[1].subgridspec(1, 2, width_ratios=[1.05, 1.45], wspace=0.12)
    axes = [fig.add_subplot(plot_grid[0, idx]) for idx in range(3)]
    legend_axis = fig.add_subplot(footer_grid[0, 0])
    avg_axis = fig.add_subplot(footer_grid[0, 1])
    for axis in axes[1:]:
        axis.sharey(axes[0])
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(pairs), 1)))

    _draw_panel(axes[0], pairs, "A", "(a) LoRA A matrices.", colors)
    _draw_panel(axes[1], pairs, "B", "(b) LoRA B matrices.", colors)
    _draw_panel(axes[2], pairs, "BA", "(c) LoRA BA matrices.", colors)
    _render_global_footer(legend_axis, avg_axis, axes[0], pairs)

    mode = str(stats.get("mode", "flatten")).upper()
    fig.suptitle(f"{source_name} | {mode} layer-wise similarity", fontsize=11)
    fig.subplots_adjust(top=0.93, bottom=0.05)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    if mode == "FLATTEN":
        legend_only_path = output_path.replace(".png", "_legend_only.png")
        avg_only_path = output_path.replace(".png", "_avg_only.png")
    else:
        legend_only_path = output_path.replace("_plot.png", "_legend_only.png")
        avg_only_path = output_path.replace("_plot.png", "_avg_only.png")
    _save_legend_only_png(legend_only_path, pairs, colors)
    _save_avg_only_png(avg_only_path, pairs)


def _analyze_layerwise_factor_similarity(
    client_global_lora: ClientGlobalLoRA,
    output_dir: str,
    source_name: str,
) -> dict[str, str]:
    artifact_paths: dict[str, str] = {}
    # Diagnostic-only similarity analysis. svd/qr modes reconstruct ΔW = B @ A
    # (d×d) per module × pair and run a full d×d SVD on it — 6,272 such SVDs ≈
    # ~30 min per mode at d=4096. Not part of the FedDPA-F method. Keep only the
    # cheap flatten mode so the artifact still gets written.
    for mode in ("flatten",):
        logger.info("[Global LoRA Analysis] Layer-wise mode start: %s", mode.upper())
        stats = _compute_pairwise_layerwise_factor_similarity(client_global_lora, mode=mode)
        if not stats.get("pairs"):
            logger.warning(
                "[Global LoRA Analysis] No pairwise layer-wise stats were generated for mode=%s.",
                mode,
            )
            continue

        if mode == "flatten":
            json_path = os.path.join(output_dir, "layerwise_similarity.json")
            csv_path = os.path.join(output_dir, "layerwise_similarity.csv")
            plot_path = os.path.join(output_dir, "layerwise_similarity_plot.png")
        else:
            json_path = os.path.join(output_dir, f"layerwise_similarity_{mode}.json")
            csv_path = os.path.join(output_dir, f"layerwise_similarity_{mode}.csv")
            plot_path = os.path.join(output_dir, f"layerwise_similarity_{mode}_plot.png")

        # with open(json_path, "w", encoding="utf-8") as file_obj:
        #     json.dump(stats, file_obj, ensure_ascii=False, indent=2)
        #
        # _save_layerwise_similarity_csv(csv_path, stats)
        # logger.info(
        #     "[Global LoRA Analysis] %s layer-wise tables saved | json=%s | csv=%s",
        #     mode.upper(),
        #     json_path,
        #     csv_path,
        # )

        plot_artifacts = {}
        try:
            _save_layerwise_similarity_plot(plot_path, stats, source_name)
            plot_artifacts["plot"] = plot_path
            logger.info(
                "[Global LoRA Analysis] %s layer-wise plot saved | plot=%s",
                mode.upper(),
                plot_path,
            )
        except ModuleNotFoundError as exc:
            logger.warning("[Global LoRA Analysis] %s", exc)

        if mode == "flatten":
            # artifact_paths["layerwise_similarity_json"] = json_path
            # artifact_paths["layerwise_similarity_csv"] = csv_path
            if "plot" in plot_artifacts:
                artifact_paths["layerwise_similarity_plot"] = plot_path
        else:
            # artifact_paths[f"layerwise_similarity_{mode}_json"] = json_path
            # artifact_paths[f"layerwise_similarity_{mode}_csv"] = csv_path
            if "plot" in plot_artifacts:
                artifact_paths[f"layerwise_similarity_{mode}_plot"] = plot_path

        logger.info("[Global LoRA Analysis] Layer-wise mode done: %s", mode.upper())

    return artifact_paths


def analyze_global_lora_sources(
    client_global_lora: ClientGlobalLoRA,
    output_dir: str,
    source_name: str,
    methods: list[str] | tuple[str, ...] | None = None,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    selected_methods = _normalize_methods(methods)
    logger.info(
        "[Global LoRA Analysis] Starting analysis | output_dir=%s | methods=%s",
        output_dir,
        ",".join(selected_methods),
    )

    torch.save(client_global_lora, os.path.join(output_dir, "client_global_lora.pt"))
    artifact_paths: Dict[str, str] = {
        "client_global_lora": os.path.join(output_dir, "client_global_lora.pt"),
    }
    logger.info("[Global LoRA Analysis] Saved client_global_lora.pt")

    layerwise_artifacts = _analyze_layerwise_factor_similarity(
        client_global_lora=client_global_lora,
        output_dir=output_dir,
        source_name=source_name,
    )
    artifact_paths.update(layerwise_artifacts)
    if layerwise_artifacts:
        logger.info(
            "[Global LoRA Analysis] Layer-wise A/B/BA similarity artifacts saved: %s",
            ", ".join(sorted(layerwise_artifacts.keys())),
        )

    for method in selected_methods:
        logger.info("[Global LoRA Analysis] Method start: %s", method.upper())
        client_vectors = _build_client_module_vectors(client_global_lora, method=method)
        client_ids, similarity, pair_stats = _pairwise_similarity_from_module_vectors(client_vectors)

        matrix_path = os.path.join(output_dir, f"{method}_pairwise_similarity.csv")
        summary_path = os.path.join(output_dir, f"{method}_summary.json")
        vector_path = os.path.join(output_dir, f"{method}_top1_vectors.pt")
        heatmap_path = os.path.join(output_dir, f"{method}_heatmap.png")
        dendrogram_path = os.path.join(output_dir, f"{method}_dendrogram.png")

        _save_similarity_csv(matrix_path, client_ids, similarity)
        _write_summary(summary_path, client_ids, similarity, pair_stats, method)
        _save_client_vectors(vector_path, client_vectors)
        logger.info(
            "[Global LoRA Analysis] %s artifacts saved | matrix=%s | summary=%s | vectors=%s",
            method.upper(),
            matrix_path,
            summary_path,
            vector_path,
        )
        try:
            _save_heatmap(
                heatmap_path,
                client_ids,
                similarity,
                title=f"{source_name} | {method.upper()} top-1 similarity",
            )
            _save_dendrogram(
                dendrogram_path,
                client_ids,
                similarity,
                title=f"{source_name} | {method.upper()} hierarchical clustering",
            )
            artifact_paths[f"{method}_heatmap"] = heatmap_path
            artifact_paths[f"{method}_dendrogram"] = dendrogram_path
            logger.info(
                "[Global LoRA Analysis] %s figures saved | heatmap=%s | dendrogram=%s",
                method.upper(),
                heatmap_path,
                dendrogram_path,
            )
        except ModuleNotFoundError as exc:
            logger.warning("[Global LoRA Analysis] %s", exc)

        artifact_paths[f"{method}_matrix"] = matrix_path
        artifact_paths[f"{method}_summary"] = summary_path
        artifact_paths[f"{method}_vectors"] = vector_path

        logger.info(
            "[Global LoRA Analysis] %s | min off-diag=%.4f | max off-diag=%.4f",
            method.upper(),
            float(np.min(similarity[np.triu_indices_from(similarity, k=1)])) if len(client_ids) > 1 else 1.0,
            float(np.max(similarity[np.triu_indices_from(similarity, k=1)])) if len(client_ids) > 1 else 1.0,
        )

    return artifact_paths


def analyze_upload_payloads(
    uploads: Dict[str, Dict],
    output_dir: str,
    source_name: str = "final_phase1_uploads",
    methods: list[str] | tuple[str, ...] | None = None,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    raw_upload_path = os.path.join(output_dir, "raw_uploads.pt")
    torch.save(uploads, raw_upload_path)

    artifact_paths = analyze_global_lora_sources(
        client_global_lora=extract_global_lora_from_uploads(uploads),
        output_dir=output_dir,
        source_name=source_name,
        methods=methods,
    )
    artifact_paths["raw_uploads"] = raw_upload_path
    return artifact_paths
