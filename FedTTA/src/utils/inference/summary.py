import argparse
import json
from pathlib import Path


TASK_DISPLAY_ORDER = {
    "dataset1": [
        ("client_4", "Paraphrase"),
        ("client_2", "Entailment"),
        ("client_6", "Structure to Text"),
        ("client_7", "Text Formatting"),
        ("client_3", "Linguistic Acc"),
        ("client_8", "Word Dis"),
        ("client_1", "Coreference"),
        ("client_5", "Question CLS"),
    ],
    "dataset2": [
        ("client_1", "Client 1"),
        ("client_2", "Client 2"),
        ("client_3", "Client 3"),
        ("client_4", "Client 4"),
        ("client_5", "Client 5"),
        ("client_6", "Client 6"),
        ("client_7", "Client 7"),
        ("client_8", "Client 8"),
    ],
}

PERSONALIZED_METHODS = ["DP-LoRA", "Ours"]
TTP_METHODS = ["DP-LoRA", "Ours-TTP"]
METRIC_KEYS = ["F1", "BLEU", "ROUGE-1", "ROUGE-L", "METEOR"]
METRIC_TITLES = {
    "F1": "F1 Summary",
    "BLEU": "BLEU-4 Summary",
    "ROUGE-1": "ROUGE-1 Summary",
    "ROUGE-L": "ROUGE-L Summary",
    "METEOR": "METEOR Summary",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge per-client inference JSON files into summary tables.")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--summary_md", required=True)
    parser.add_argument("input_files", nargs="+")
    return parser


def append_block(lines, methods, metric, ordered_tasks, by_method_client_scope, sample_scope):
    headers = ["Methods"] + [name for _, name in ordered_tasks] + ["Average"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for method in methods:
        values = []
        for client_id, _display_name in ordered_tasks:
            row = by_method_client_scope.get((method, client_id, sample_scope))
            values.append(None if row is None else row[metric])
        if all(v is None for v in values):
            continue
        valid_values = [v for v in values if v is not None]
        avg = sum(valid_values) / len(valid_values) if valid_values else None
        formatted_values = ["-" if v is None else f"{v:.2f}" for v in values]
        avg_text = "-" if avg is None else f"{avg:.2f}"
        lines.append("| " + " | ".join([method] + formatted_values + [avg_text]) + " |")


def main() -> None:
    args = build_parser().parse_args()
    summary_json = Path(args.summary_json)
    summary_md = Path(args.summary_md)
    input_files = [Path(p) for p in args.input_files]

    rows = []
    for path in input_files:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        rows.extend(payload.get("rows", []))

    summary_json.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    ordered_tasks = TASK_DISPLAY_ORDER.get(args.dataset_name, TASK_DISPLAY_ORDER["dataset2"])
    by_method_client_scope = {}
    for row in rows:
        by_method_client_scope[(row["model"], row["client"], row.get("sample_scope", "personalized_subset"))] = row

    md_lines = ["# Inference Summary", ""]
    for metric in METRIC_KEYS:
        md_lines.append(f"## {METRIC_TITLES[metric]}")
        md_lines.append("")
        md_lines.append("| Personalized |")
        append_block(md_lines, PERSONALIZED_METHODS, metric, ordered_tasks, by_method_client_scope, "personalized_subset")
        md_lines.append("|---|---|---|---|---|---|---|---|---|---|")
        md_lines.append("| Test-Time-Personalization |")
        append_block(md_lines, TTP_METHODS, metric, ordered_tasks, by_method_client_scope, "full_test_file")
        md_lines.append("")

    summary_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"summary json: {summary_json}")
    print(f"summary md: {summary_md}")


if __name__ == "__main__":
    main()
