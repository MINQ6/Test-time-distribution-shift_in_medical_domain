from __future__ import annotations

import argparse
import json
import logging
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
DEFAULT_RUN_TAG = "entropy_comparison_table"


def _build_run_id(run_tag: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_tag}"


def _setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("render_entropy_comparison_table")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(log_dir / "render_entropy_comparison_table.log", encoding="utf-8")
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


def _format_value(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _build_table_rows(summaries: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    table_rows: list[dict[str, Any]] = []
    for scope in ("personalized_subset", "full_test_file"):
        table_rows.append({"section": SCOPE_LABELS[scope]})
        for summary in summaries:
            values = _metric_values(summary.get("rows") or [], scope, metric)
            valid_values = [value for value in values if value is not None]
            table_rows.append(
                {
                    "method": summary.get("method", "Unknown"),
                    "scope": scope,
                    "values": values,
                    "average": sum(valid_values) / len(valid_values) if valid_values else None,
                }
            )
    return table_rows


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

    row_heights = [1.25] + [0.46 if "section" in row else 0.58 for row in table_rows]
    y_edges = [sum(row_heights)]
    for height in row_heights:
        y_edges.append(y_edges[-1] - height)
    total_height = sum(row_heights)

    fig, ax = plt.subplots(figsize=(14.2, 4.9), dpi=220)
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
            ax.add_patch(Rectangle((0, y_top - height), total_width, height, facecolor=section_fill, edgecolor="none"))
        else:
            fill = entropy_fill if row.get("method") == "Entropy-Min" else method_fill
            ax.add_patch(Rectangle((0, y_top - height), total_width, height, facecolor=fill, edgecolor="none"))

    ax.plot([0, total_width], [total_height - 0.05, total_height - 0.05], color=line_color, lw=2.3)
    ax.plot([0, total_width], [0.05, 0.05], color=line_color, lw=2.3)
    ax.plot([0, total_width], [y_edges[1], y_edges[1]], color=thin_color, lw=0.8)
    ax.plot([x_edges[1], x_edges[1]], [0.18, total_height - 0.18], color=line_color, lw=0.8)
    ax.plot([x_edges[-2], x_edges[-2]], [0.18, y_edges[1]], color=thin_color, lw=0.8)

    for edge_idx in range(2, len(y_edges) - 1):
        if edge_idx in {4}:
            ax.plot([0, total_width], [y_edges[edge_idx], y_edges[edge_idx]], color=line_color, lw=0.8)
        else:
            ax.plot([0, total_width], [y_edges[edge_idx], y_edges[edge_idx]], color=thin_color, lw=0.45)

    ax.text((x_edges[1] + x_edges[-1]) / 2, total_height - 0.25, dataset_name, ha="center", va="top", fontsize=16, fontweight="bold", family="serif")
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
        all_values = [*row["values"], row["average"]]
        for idx, value in enumerate(all_values, start=1):
            x_center = (x_edges[idx] + x_edges[idx + 1]) / 2
            ax.text(x_center, y_center, _format_value(value), ha="center", va="center", fontsize=12.8, family="serif")

    ax.text(total_width, 0.08, f"Metric: {metric}", ha="right", va="bottom", fontsize=7.5, color="#555555")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render FedDPA-F vs entropy-minimization comparison table.")
    parser.add_argument("--feddpa_summary_json", type=Path, required=True)
    parser.add_argument("--entropy_summary_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--log_dir", type=Path, default=None)
    parser.add_argument("--run_tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--metric", default="ROUGE-1")
    parser.add_argument("--dataset_title", default="Federated Dataset 1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_id = _build_run_id(args.run_tag)
    output_dir = args.output_dir or (PROJECT_ROOT / "outputs" / run_id)
    log_dir = args.log_dir or (PROJECT_ROOT / "logs" / run_id)
    logger = _setup_logging(log_dir)

    feddpa_summary = _load_summary(args.feddpa_summary_json)
    entropy_summary = _load_summary(args.entropy_summary_json)
    table_rows = _build_table_rows([feddpa_summary, entropy_summary], args.metric)
    png_path = output_dir / "fedDPA_vs_entropy_minimization_table.png"
    _render_png(png_path, table_rows, args.metric, args.dataset_title)

    for row in table_rows:
        if "section" in row:
            logger.info("[%s]", row["section"])
        else:
            logger.info("%s average=%s", row["method"], _format_value(row["average"]))
    logger.info("Saved PNG: %s", png_path)


if __name__ == "__main__":
    main()
