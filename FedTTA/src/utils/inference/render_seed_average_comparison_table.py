from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_SRC = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PROJECT_SRC.parent
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))


TASK_ORDER = [
    "paraphrase",
    "entailment",
    "structure_to_text",
    "text_formatting",
    "linguistic_acceptability",
    "word_disambiguation",
    "coreference",
    "question_classification",
]
TASK_HEADERS = [
    "Para\n-phrase",
    "Entail\n-ment",
    "Structure\nto Text",
    "Text For\n-matting",
    "Linguistic\nAcc",
    "Word\nDis",
    "Core\n-ference",
    "Question\nCLS",
]
SCOPE_LABELS = {
    "personalized_subset": "Personalization",
    "full_test_file": "Test-Time Personalization",
}
METHOD_LABELS = {
    "feddpa": "FedDPA-F",
    "entropy_min": "Entropy-Min",
}
DEFAULT_RUN_TAG = "greedy_seed_average_table"


def _build_run_id(run_tag: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_tag}"


def _setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("render_seed_average_comparison_table")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(log_dir / "render_seed_average_comparison_table.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _metric_values(rows: list[dict[str, Any]], scope: str, metric: str) -> list[float | None]:
    values: list[float | None] = []
    scoped_rows = [row for row in rows if row.get("sample_scope") == scope]
    for task in TASK_ORDER:
        task_values = []
        for row in scoped_rows:
            task_payload = (row.get("task_metrics") or {}).get(task)
            if task_payload and metric in task_payload:
                task_values.append(float(task_payload[metric]))
        values.append(sum(task_values) / len(task_values) if task_values else None)
    return values


def _population_variance(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _stats(values: list[float | None]) -> dict[str, Any]:
    valid_values = [value for value in values if value is not None]
    variance = _population_variance(valid_values)
    mean = sum(valid_values) / len(valid_values) if valid_values else None
    return {
        "values": valid_values,
        "mean": mean,
        "variance": variance,
        "std": math.sqrt(variance) if variance is not None else None,
        "count": len(valid_values),
    }


def _format_cell(payload: dict[str, Any] | None) -> str:
    if not payload or payload.get("mean") is None:
        return "-"
    variance = payload.get("variance")
    if variance is None:
        return f"{payload['mean']:.2f}"
    return f"{payload['mean']:.2f}\nvar {variance:.3f}"


def _build_method_scope_stats(
    summaries: list[dict[str, Any]],
    scope: str,
    metric: str,
) -> dict[str, Any]:
    per_seed_task_values = [_metric_values(summary.get("rows") or [], scope, metric) for summary in summaries]
    task_stats = {}
    for task_index, task in enumerate(TASK_ORDER):
        task_stats[task] = _stats([values[task_index] for values in per_seed_task_values])

    seed_averages: list[float] = []
    for values in per_seed_task_values:
        valid_values = [value for value in values if value is not None]
        if valid_values:
            seed_averages.append(sum(valid_values) / len(valid_values))
    return {
        "tasks": task_stats,
        "average": _stats(seed_averages),
    }


def _build_table_rows(
    method_summaries: dict[str, list[dict[str, Any]]],
    metric: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table_rows: list[dict[str, Any]] = []
    summary_payload: dict[str, Any] = {"metric": metric, "methods": {}}
    for method_key, summaries in method_summaries.items():
        summary_payload["methods"][method_key] = {
            "label": METHOD_LABELS[method_key],
            "seeds": [summary.get("settings", {}).get("seed") for summary in summaries],
            "scopes": {},
        }

    for scope in ("personalized_subset", "full_test_file"):
        table_rows.append({"section": SCOPE_LABELS[scope]})
        for method_key in ("feddpa", "entropy_min"):
            scoped_stats = _build_method_scope_stats(method_summaries[method_key], scope, metric)
            summary_payload["methods"][method_key]["scopes"][scope] = scoped_stats
            table_rows.append(
                {
                    "method": METHOD_LABELS[method_key],
                    "scope": scope,
                    "task_stats": scoped_stats["tasks"],
                    "average": scoped_stats["average"],
                }
            )
    return table_rows, summary_payload


def _render_png(path: Path, table_rows: list[dict[str, Any]], metric: str, dataset_name: str) -> None:
    mpl_config_dir = PROJECT_ROOT / "outputs" / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    columns = ["Methods", *TASK_HEADERS, "Average"]
    widths = [2.35, 1.25, 1.25, 1.55, 1.55, 1.55, 1.25, 1.35, 1.55, 1.35]
    x_edges = [0.0]
    for width in widths:
        x_edges.append(x_edges[-1] + width)
    total_width = x_edges[-1]

    row_heights = [1.25] + [0.46 if "section" in row else 0.76 for row in table_rows]
    y_edges = [sum(row_heights)]
    for height in row_heights:
        y_edges.append(y_edges[-1] - height)
    total_height = sum(row_heights)

    fig, ax = plt.subplots(figsize=(15.8, 5.4), dpi=220)
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, total_height)
    ax.axis("off")

    section_fill = "#eeeeee"
    method_fill = "#f4f9ff"
    entropy_fill = "#fff8ec"
    line_color = "#111111"
    thin_color = "#555555"

    for row_idx, row in enumerate(table_rows, start=1):
        y_top = y_edges[row_idx]
        height = row_heights[row_idx]
        if "section" in row:
            fill = section_fill
        else:
            fill = entropy_fill if row.get("method") == "Entropy-Min" else method_fill
        ax.add_patch(Rectangle((0, y_top - height), total_width, height, facecolor=fill, edgecolor="none"))

    ax.plot([0, total_width], [total_height - 0.05, total_height - 0.05], color=line_color, lw=2.3)
    ax.plot([0, total_width], [0.05, 0.05], color=line_color, lw=2.3)
    ax.plot([0, total_width], [y_edges[1], y_edges[1]], color=thin_color, lw=0.8)
    ax.plot([x_edges[1], x_edges[1]], [0.18, total_height - 0.18], color=line_color, lw=0.8)
    ax.plot([x_edges[-2], x_edges[-2]], [0.18, y_edges[1]], color=thin_color, lw=0.8)

    for edge_idx in range(2, len(y_edges) - 1):
        line_width = 0.8 if edge_idx == 4 else 0.45
        line_col = line_color if edge_idx == 4 else thin_color
        ax.plot([0, total_width], [y_edges[edge_idx], y_edges[edge_idx]], color=line_col, lw=line_width)

    ax.text(
        (x_edges[1] + x_edges[-1]) / 2,
        total_height - 0.25,
        dataset_name,
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
        family="serif",
    )
    for index, column in enumerate(columns):
        x_center = (x_edges[index] + x_edges[index + 1]) / 2
        if index == 0:
            ax.text(x_edges[index] + 0.12, y_edges[1] + 0.52, column, ha="left", va="center", fontsize=14, family="serif")
        else:
            ax.text(x_center, y_edges[1] + 0.36, column, ha="center", va="center", fontsize=12.5, family="serif", linespacing=1.1)

    for row_idx, row in enumerate(table_rows, start=1):
        y_center = (y_edges[row_idx] + y_edges[row_idx + 1]) / 2
        if "section" in row:
            ax.text(0.14, y_center, row["section"], ha="left", va="center", fontsize=13.5, family="serif", style="italic")
            continue

        ax.text(0.14, y_center, row["method"], ha="left", va="center", fontsize=13.1, family="serif")
        payloads = [row["task_stats"].get(task) for task in TASK_ORDER] + [row["average"]]
        for idx, payload in enumerate(payloads, start=1):
            x_center = (x_edges[idx] + x_edges[idx + 1]) / 2
            ax.text(
                x_center,
                y_center,
                _format_cell(payload),
                ha="center",
                va="center",
                fontsize=10.9,
                family="serif",
                linespacing=1.15,
            )

    ax.text(total_width, 0.08, f"Metric: {metric}, cell: mean + variance over seeds", ha="right", va="bottom", fontsize=7.5, color="#555555")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _load_many(paths: list[Path], method_label: str) -> list[dict[str, Any]]:
    summaries = []
    for path in paths:
        summary = _load_summary(path)
        if summary.get("method") != method_label:
            summary["method"] = method_label
        summaries.append(summary)
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render mean/variance table for greedy seed-average comparison.")
    parser.add_argument("--feddpa_summary_jsons", type=Path, nargs="+", required=True)
    parser.add_argument("--entropy_summary_jsons", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--log_dir", type=Path, default=None)
    parser.add_argument("--run_tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--metric", default="ROUGE-1")
    parser.add_argument("--dataset_title", default="Federated Dataset 1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if len(args.feddpa_summary_jsons) != len(args.entropy_summary_jsons):
        raise ValueError("FedDPA-F and Entropy-Min summary counts must match.")

    run_id = _build_run_id(args.run_tag)
    output_dir = args.output_dir or (PROJECT_ROOT / "outputs" / run_id)
    log_dir = args.log_dir or (PROJECT_ROOT / "logs" / run_id)
    logger = _setup_logging(log_dir)

    method_summaries = {
        "feddpa": _load_many(args.feddpa_summary_jsons, "FedDPA-F"),
        "entropy_min": _load_many(args.entropy_summary_jsons, "Entropy-Min"),
    }
    table_rows, summary_payload = _build_table_rows(method_summaries, args.metric)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "fedDPA_vs_entropy_minimization_seed_average_table.png"
    json_path = output_dir / "seed_average_summary.json"
    _render_png(png_path, table_rows, args.metric, args.dataset_title)
    with json_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary_payload, file_obj, indent=2, ensure_ascii=False)

    for row in table_rows:
        if "section" in row:
            logger.info("[%s]", row["section"])
        else:
            logger.info("%s average=%s", row["method"], _format_cell(row["average"]).replace("\n", " "))
    logger.info("Saved PNG: %s", png_path)
    logger.info("Saved JSON: %s", json_path)


if __name__ == "__main__":
    main()
