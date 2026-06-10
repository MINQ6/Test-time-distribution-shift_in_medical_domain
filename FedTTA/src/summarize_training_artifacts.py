import argparse
import csv
import json
import os
from datetime import datetime
from statistics import mean
from typing import Any


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if text == "" or text.lower() in {"nan", "n/a", "not supported"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _read_json_if_exists(path: str) -> dict[str, Any] | None:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def summarize_gpu_csv(gpu_csv_path: str) -> dict[str, Any]:
    if not os.path.isfile(gpu_csv_path):
        return {
            "status": "missing",
            "path": gpu_csv_path,
            "message": "GPU CSV file not found.",
        }

    rows: list[dict[str, Any]] = []
    with open(gpu_csv_path, "r", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            ts = row.get("timestamp", "").strip()
            try:
                parsed_ts = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                parsed_ts = None

            rows.append(
                {
                    "timestamp": parsed_ts,
                    "index": row.get("index", "").strip(),
                    "utilization_gpu": _to_float(row.get("utilization_gpu")),
                    "utilization_memory": _to_float(row.get("utilization_memory")),
                    "memory_used_mb": _to_float(row.get("memory_used_mb")),
                    "memory_total_mb": _to_float(row.get("memory_total_mb")),
                    "power_draw_w": _to_float(row.get("power_draw_w")),
                    "temperature_c": _to_float(row.get("temperature_c")),
                }
            )

    if not rows:
        return {
            "status": "empty",
            "path": gpu_csv_path,
            "message": "GPU CSV exists but contains no data rows.",
        }

    def collect(metric: str) -> list[float]:
        return [row[metric] for row in rows if row.get(metric) is not None]

    gpu_utils = collect("utilization_gpu")
    mem_utils = collect("utilization_memory")
    mem_used = collect("memory_used_mb")
    power = collect("power_draw_w")
    temp = collect("temperature_c")

    timestamps = [row["timestamp"] for row in rows if row["timestamp"] is not None]
    start_time = min(timestamps) if timestamps else None
    end_time = max(timestamps) if timestamps else None

    return {
        "status": "ok",
        "path": gpu_csv_path,
        "num_rows": len(rows),
        "num_unique_gpus": len({row["index"] for row in rows if row.get("index") != ""}),
        "time": {
            "start": start_time.isoformat() if start_time else None,
            "end": end_time.isoformat() if end_time else None,
            "duration_sec": int((end_time - start_time).total_seconds()) if start_time and end_time else None,
        },
        "metrics": {
            "gpu_util_mean": mean(gpu_utils) if gpu_utils else None,
            "gpu_util_max": max(gpu_utils) if gpu_utils else None,
            "gpu_util_min": min(gpu_utils) if gpu_utils else None,
            "mem_util_mean": mean(mem_utils) if mem_utils else None,
            "mem_util_max": max(mem_utils) if mem_utils else None,
            "vram_used_mb_mean": mean(mem_used) if mem_used else None,
            "vram_used_mb_peak": max(mem_used) if mem_used else None,
            "power_draw_w_mean": mean(power) if power else None,
            "power_draw_w_peak": max(power) if power else None,
            "temperature_c_mean": mean(temp) if temp else None,
            "temperature_c_peak": max(temp) if temp else None,
        },
    }


def summarize_lora_analysis(analysis_dir: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "analysis_dir": analysis_dir,
        "exists": os.path.isdir(analysis_dir),
        "svd": None,
        "qr": None,
        "layerwise": None,
    }
    if not summary["exists"]:
        summary["message"] = "Analysis directory does not exist."
        return summary

    svd_summary = _read_json_if_exists(os.path.join(analysis_dir, "svd_summary.json"))
    qr_summary = _read_json_if_exists(os.path.join(analysis_dir, "qr_summary.json"))
    layerwise = _read_json_if_exists(os.path.join(analysis_dir, "layerwise_similarity.json"))

    if svd_summary is not None:
        summary["svd"] = {
            "mean_off_diagonal_similarity": svd_summary.get("mean_off_diagonal_similarity"),
            "min_off_diagonal_similarity": svd_summary.get("min_off_diagonal_similarity"),
            "max_off_diagonal_similarity": svd_summary.get("max_off_diagonal_similarity"),
            "closest_to_zero_pair": svd_summary.get("closest_to_zero_pair"),
            "highest_similarity_pair": svd_summary.get("highest_similarity_pair"),
        }

    if qr_summary is not None:
        summary["qr"] = {
            "mean_off_diagonal_similarity": qr_summary.get("mean_off_diagonal_similarity"),
            "min_off_diagonal_similarity": qr_summary.get("min_off_diagonal_similarity"),
            "max_off_diagonal_similarity": qr_summary.get("max_off_diagonal_similarity"),
            "closest_to_zero_pair": qr_summary.get("closest_to_zero_pair"),
            "highest_similarity_pair": qr_summary.get("highest_similarity_pair"),
        }

    if layerwise is not None:
        pairs = layerwise.get("pairs", [])
        avg_a = [pair.get("avg_A") for pair in pairs if pair.get("avg_A") is not None]
        avg_b = [pair.get("avg_B") for pair in pairs if pair.get("avg_B") is not None]
        avg_ba = [pair.get("avg_BA") for pair in pairs if pair.get("avg_BA") is not None]

        summary["layerwise"] = {
            "num_pairs": len(pairs),
            "num_layers": len(layerwise.get("layers", [])),
            "mean_pair_avg_A": mean(avg_a) if avg_a else None,
            "mean_pair_avg_B": mean(avg_b) if avg_b else None,
            "mean_pair_avg_BA": mean(avg_ba) if avg_ba else None,
            "lowest_avg_BA_pair": min(pairs, key=lambda x: x.get("avg_BA", 1.0)) if pairs else None,
            "highest_avg_BA_pair": max(pairs, key=lambda x: x.get("avg_BA", 0.0)) if pairs else None,
        }

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize GPU monitoring CSV and LoRA analysis artifacts into one JSON report."
    )
    parser.add_argument("--gpu_csv", required=True, type=str)
    parser.add_argument("--analysis_dir", required=True, type=str)
    parser.add_argument("--output_json", required=True, type=str)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    report = {
        "gpu": summarize_gpu_csv(args.gpu_csv),
        "lora_analysis": summarize_lora_analysis(args.analysis_dir),
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, ensure_ascii=False, indent=2)

    print("Saved post-training summary:")
    print(args.output_json)

    gpu_peak = report["gpu"].get("metrics", {}).get("vram_used_mb_peak") if isinstance(report.get("gpu"), dict) else None
    ba_mean = report["lora_analysis"].get("layerwise", {}).get("mean_pair_avg_BA") if isinstance(report.get("lora_analysis"), dict) else None
    print(f"GPU peak VRAM (MB): {gpu_peak}")
    print(f"Layer-wise mean pair avg BA similarity: {ba_mean}")


if __name__ == "__main__":
    main()
