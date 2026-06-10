from __future__ import annotations

import argparse
import logging
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from transformers import AutoTokenizer


PROJECT_SRC = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PROJECT_SRC.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    path for path in sys.path
    if path and Path(path).resolve() != SCRIPT_DIR
]
sys.path.insert(0, str(PROJECT_SRC))

from data.loader import _load_local_feddpa_examples
from utils.inference.common import MODEL_ID, PROMPT_INPUT, PROMPT_NO_INPUT, checkpoint_suffix, compute_f1, logger, set_seed
from utils.inference.data import load_full_test_dataset, load_personalized_samples
from utils.inference.evaluator import generate_answers
from utils.inference.model_loader import load_adapter_model


DEFAULT_CHECKPOINT_ROOT = PROJECT_SRC / "checkpoints"
DEFAULT_OUTPUT_PARENT = PROJECT_ROOT / "outputs"
DEFAULT_LOG_PARENT = PROJECT_ROOT / "logs"
DEFAULT_RUN_TAG = "inference_fedDPA"
DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_PARENT / DEFAULT_RUN_TAG


def _build_run_id(run_tag: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{run_tag}"


def _add_file_logger(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "inference_fedDPA.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(filename)s - %(message)s"))
    root_logger = logging.getLogger()
    if not any(isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path for handler in root_logger.handlers):
        root_logger.addHandler(file_handler)
    logger.info("[FedDPA-F Inference] Log file: %s", log_path)


def _build_prompt(sample: dict[str, str]) -> str:
    instruction = sample["question"].strip()
    input_text = (sample.get("input") or "").strip()
    if input_text:
        return PROMPT_INPUT.format(instruction=instruction, input=input_text)
    return PROMPT_NO_INPUT.format(instruction=instruction)


def _load_hf_token() -> str:
    load_dotenv(PROJECT_SRC / ".env")
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN is not set. Put it in src/.env or export it.")
    return hf_token


def _client_sort_key(client_id: str) -> int:
    return int(checkpoint_suffix(client_id))


def _normalize_client_id(value: str | int) -> str:
    text = str(value)
    return text if text.startswith("client_") else f"client_{text}"


def _discover_latest_checkpoint_dir(checkpoint_root: Path) -> Path:
    candidates = [
        path
        for path in checkpoint_root.iterdir()
        if path.is_dir() and list(path.glob("dual_lora_adapter_client_*.pth"))
    ]
    if not candidates:
        if list(checkpoint_root.glob("dual_lora_adapter_client_*.pth")):
            return checkpoint_root
        raise FileNotFoundError(f"No FedDPA-F checkpoints found under {checkpoint_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _discover_clients(checkpoint_dir: Path) -> list[str]:
    clients = []
    for checkpoint_path in checkpoint_dir.glob("dual_lora_adapter_client_*.pth"):
        suffix = checkpoint_path.stem.removeprefix("dual_lora_adapter_client_")
        clients.append(_normalize_client_id(suffix))
    if not clients:
        raise FileNotFoundError(f"No dual_lora_adapter_client_*.pth files in {checkpoint_dir}")
    return sorted(clients, key=_client_sort_key)


def _checkpoint_path(checkpoint_dir: Path, client_id: str) -> Path:
    path = checkpoint_dir / f"dual_lora_adapter_client_{checkpoint_suffix(client_id)}.pth"
    if not path.exists():
        raise FileNotFoundError(f"Missing FedDPA-F checkpoint for {client_id}: {path}")
    return path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def _set_adapter_weights(model: Any, global_weight: float | torch.Tensor, local_weight: float | torch.Tensor) -> None:
    model.dual_lora_adapter.set_adapter_weights(
        global_weight=global_weight,
        local_weight=local_weight,
    )


def _encode_prompts(
    *,
    model: Any,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    device: torch.device,
    emb_type: str = "last",
) -> torch.Tensor:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden = outputs.hidden_states[-1]
    if emb_type == "avg":
        return torch.mean(hidden, dim=1)
    return hidden[:, -1, :]


def _load_local_reference_samples(
    *,
    client_id: str,
    dataset_name: str,
    num_instances: int,
) -> list[dict[str, str]]:
    examples = _load_local_feddpa_examples(
        dataset_name=dataset_name,
        client_id=client_id,
        split="train",
        limit=num_instances,
    )
    return [
        {
            "question": (example.get("instruction") or example.get("inputs") or "").strip(),
            "input": (example.get("input") or "").strip(),
            "reference": (example.get("output") or "").strip(),
            "task_name": example.get("task", ""),
            "task_type": example.get("category", ""),
        }
        for example in examples
    ]


def _sample_local_references(
    local_reference_pool: list[dict[str, str]],
    *,
    num_instances: int,
) -> list[dict[str, str]]:
    if len(local_reference_pool) <= num_instances:
        return list(local_reference_pool)
    return random.sample(local_reference_pool, num_instances)


def _compute_auto_global_weight(
    *,
    model: Any,
    tokenizer: AutoTokenizer,
    candidate_sample: dict[str, str],
    local_reference_samples: list[dict[str, str]],
    device: torch.device,
    scale: float,
    emb_type: str,
) -> float:
    previous_global_weight = next(iter(model.dual_lora_adapter._wrapped_modules.values())).global_adapter_weight
    previous_local_weight = next(iter(model.dual_lora_adapter._wrapped_modules.values())).local_adapter_weight
    _set_adapter_weights(model, global_weight=0.0, local_weight=1.0)
    prompts = [_build_prompt(sample) for sample in local_reference_samples] + [_build_prompt(candidate_sample)]
    embeddings = _encode_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
        emb_type=emb_type,
    )
    candidate = embeddings[-1]
    references = embeddings[:-1]
    similarity = F.cosine_similarity(candidate.unsqueeze(0), references, dim=1).mean().item()
    _set_adapter_weights(model, global_weight=previous_global_weight, local_weight=previous_local_weight)
    return max(0.0, min(scale, similarity * scale))


def _compute_auto_global_weights_batch(
    *,
    model: Any,
    tokenizer: AutoTokenizer,
    candidate_samples: list[dict[str, str]],
    reference_embeddings: torch.Tensor,
    device: torch.device,
    scale: float,
    emb_type: str,
) -> torch.Tensor:
    previous_global_weight = next(iter(model.dual_lora_adapter._wrapped_modules.values())).global_adapter_weight
    previous_local_weight = next(iter(model.dual_lora_adapter._wrapped_modules.values())).local_adapter_weight
    _set_adapter_weights(model, global_weight=0.0, local_weight=1.0)
    candidate_embeddings = _encode_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=[_build_prompt(sample) for sample in candidate_samples],
        device=device,
        emb_type=emb_type,
    )
    similarities = F.cosine_similarity(
        candidate_embeddings.unsqueeze(1),
        reference_embeddings.unsqueeze(0),
        dim=2,
    ).mean(dim=1)
    _set_adapter_weights(model, global_weight=previous_global_weight, local_weight=previous_local_weight)
    return (similarities * scale).clamp(min=0.0, max=scale).detach()


def _compute_generation_metrics(
    samples: list[dict[str, str]],
    predictions: list[str],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    rouge_sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    smoothie = SmoothingFunction().method1
    total = {"F1": 0.0, "BLEU": 0.0, "ROUGE-1": 0.0, "ROUGE-L": 0.0, "ExactMatch": 0.0}
    by_task: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}

    for sample, prediction in zip(samples, predictions):
        reference = sample["reference"].strip()
        rouge_scores = rouge_sc.score(reference, prediction)
        values = {
            "F1": compute_f1(prediction, reference),
            "BLEU": sentence_bleu([reference.split()], prediction.split(), smoothing_function=smoothie),
            "ROUGE-1": rouge_scores["rouge1"].fmeasure,
            "ROUGE-L": rouge_scores["rougeL"].fmeasure,
            "ExactMatch": float(prediction.strip().lower() == reference.strip().lower()),
        }
        task_key = sample.get("task_type") or sample.get("task_name") or "unknown"
        by_task.setdefault(task_key, {key: 0.0 for key in values})
        counts[task_key] = counts.get(task_key, 0) + 1
        for key, value in values.items():
            total[key] += value
            by_task[task_key][key] += value

    if samples:
        for key in total:
            total[key] = total[key] / len(samples) * 100
    for task_key, values in by_task.items():
        for key in values:
            values[key] = values[key] / counts[task_key] * 100
        values["num_samples"] = counts[task_key]
    total["METEOR"] = 0.0
    for values in by_task.values():
        values["METEOR"] = 0.0
    return total, by_task


def _generate_with_feddpa_weights(
    *,
    model: Any,
    tokenizer: AutoTokenizer,
    samples: list[dict[str, str]],
    device: torch.device,
    batch_size: int,
    adapter_mode: str,
    auto_reference_samples: list[dict[str, str]] | None,
    auto_num_instances: int,
    auto_weight_scale: float,
    auto_emb_type: str,
    auto_reference_strategy: str,
    static_global_weight: float,
    max_new_tokens: int,
    num_beams: int,
) -> tuple[list[str], list[dict[str, float]]]:
    predictions: list[str] = []
    weight_rows: list[dict[str, float]] = []

    if adapter_mode == "local_only":
        _set_adapter_weights(model, global_weight=0.0, local_weight=1.0)
        return generate_answers(model, tokenizer, samples, device, batch_size, max_new_tokens=max_new_tokens, num_beams=num_beams), []

    if adapter_mode == "global_only":
        _set_adapter_weights(model, global_weight=1.0, local_weight=0.0)
        return generate_answers(model, tokenizer, samples, device, batch_size, max_new_tokens=max_new_tokens, num_beams=num_beams), []

    if adapter_mode == "static":
        static_global_weight = max(0.0, min(1.0, static_global_weight))
        _set_adapter_weights(model, global_weight=static_global_weight, local_weight=1.0 - static_global_weight)
        return generate_answers(model, tokenizer, samples, device, batch_size, max_new_tokens=max_new_tokens, num_beams=num_beams), []

    if adapter_mode != "auto":
        raise ValueError(f"Unsupported adapter_mode: {adapter_mode}")
    if not auto_reference_samples:
        raise ValueError("auto adapter mode requires local reference samples.")

    reference_embeddings = None
    if auto_reference_strategy == "fixed":
        fixed_references = auto_reference_samples[:auto_num_instances]
        _set_adapter_weights(model, global_weight=0.0, local_weight=1.0)
        reference_embeddings = _encode_prompts(
            model=model,
            tokenizer=tokenizer,
            prompts=[_build_prompt(sample) for sample in fixed_references],
            device=device,
            emb_type=auto_emb_type,
        ).detach()
    elif auto_reference_strategy != "random_per_sample":
        raise ValueError(f"Unsupported auto_reference_strategy: {auto_reference_strategy}")

    for start_idx in range(0, len(samples), batch_size):
        batch_samples = samples[start_idx:start_idx + batch_size]
        if auto_reference_strategy == "fixed":
            global_weights = _compute_auto_global_weights_batch(
                model=model,
                tokenizer=tokenizer,
                candidate_samples=batch_samples,
                reference_embeddings=reference_embeddings,
                device=device,
                scale=auto_weight_scale,
                emb_type=auto_emb_type,
            )
        else:
            global_weights = torch.tensor(
                [
                    _compute_auto_global_weight(
                        model=model,
                        tokenizer=tokenizer,
                        candidate_sample=sample,
                        local_reference_samples=_sample_local_references(
                            auto_reference_samples,
                            num_instances=auto_num_instances,
                        ),
                        device=device,
                        scale=auto_weight_scale,
                        emb_type=auto_emb_type,
                    )
                    for sample in batch_samples
                ],
                device=device,
                dtype=torch.float32,
            )
        local_weights = 1.0 - global_weights
        _set_adapter_weights(model, global_weight=global_weights, local_weight=local_weights)
        predictions.extend(
            generate_answers(
                model,
                tokenizer,
                batch_samples,
                device,
                len(batch_samples),
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
        )
        weight_rows.extend(
            {
                "global_weight": float(global_weight.item()),
                "local_weight": float(local_weight.item()),
            }
            for global_weight, local_weight in zip(global_weights, local_weights)
        )
    return predictions, weight_rows


def _evaluate_scope(
    *,
    model: Any,
    tokenizer: AutoTokenizer,
    device: torch.device,
    client_id: str,
    dataset_name: str,
    scope_name: str,
    samples: list[dict[str, str]],
    output_dir: Path,
    inference_batch_size: int,
    save_predictions: bool,
    adapter_mode: str,
    auto_reference_samples: list[dict[str, str]] | None,
    auto_num_instances: int,
    auto_weight_scale: float,
    auto_emb_type: str,
    auto_reference_strategy: str,
    static_global_weight: float,
    max_new_tokens: int,
    num_beams: int,
) -> dict[str, Any]:
    desc = f"FedDPA-F {client_id} | {scope_name}"
    logger.info("Evaluating: %s | adapter_mode=%s | samples=%d", desc, adapter_mode, len(samples))
    predictions, weight_rows = _generate_with_feddpa_weights(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        device=device,
        batch_size=inference_batch_size,
        adapter_mode=adapter_mode,
        auto_reference_samples=auto_reference_samples,
        auto_num_instances=auto_num_instances,
        auto_weight_scale=auto_weight_scale,
        auto_emb_type=auto_emb_type,
        auto_reference_strategy=auto_reference_strategy,
        static_global_weight=static_global_weight,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    metrics, task_metrics = _compute_generation_metrics(samples, predictions)

    row = {
        "client": client_id,
        "dataset_name": dataset_name,
        "model": "FedDPA-F",
        "sample_scope": scope_name,
        "num_samples": len(samples),
        **metrics,
        "task_metrics": task_metrics,
    }

    if save_predictions:
        prediction_rows = [
            {
                "client": client_id,
                "dataset_name": dataset_name,
                "sample_scope": scope_name,
                "question": sample.get("question", ""),
                "input": sample.get("input", ""),
                "reference": sample.get("reference", ""),
                "prediction": prediction,
                **(weight_rows[idx] if idx < len(weight_rows) else {}),
                "task_name": sample.get("task_name", ""),
                "task_type": sample.get("task_type", ""),
            }
            for idx, (sample, prediction) in enumerate(zip(samples, predictions))
        ]
        _write_jsonl(output_dir / "predictions" / f"{client_id}_{scope_name}.jsonl", prediction_rows)

    return row


def run_feddpa_f_inference(
    *,
    checkpoint_dir: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_ROOT,
    model_id: str = MODEL_ID,
    dataset_name: str = "dataset1",
    clients: list[str] | None = None,
    num_samples: int = 200,
    ttp_num_samples: int | None = None,
    disable_ttp: bool = False,
    inference_batch_size: int = 8,
    save_predictions: bool = False,
    personalization_adapter_mode: str = "static",
    ttp_adapter_mode: str = "auto",
    auto_num_instances: int = 1,
    auto_weight_scale: float = 1.0,
    auto_emb_type: str = "last",
    auto_reference_strategy: str = "random_per_sample",
    static_global_weight: float = 0.5,
    max_new_tokens: int = 80,
    num_beams: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    set_seed(seed)
    checkpoint_dir = checkpoint_dir or _discover_latest_checkpoint_dir(DEFAULT_CHECKPOINT_ROOT)
    clients = [_normalize_client_id(client) for client in clients] if clients else _discover_clients(checkpoint_dir)

    hf_token = _load_hf_token()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    logger.info("\n" + "=" * 90)
    logger.info("[FedDPA-F Inference] checkpoint_dir=%s", checkpoint_dir)
    logger.info("[FedDPA-F Inference] output_dir=%s", output_dir)
    logger.info("[FedDPA-F Inference] dataset=%s | clients=%s", dataset_name, ",".join(clients))
    logger.info("[FedDPA-F Inference] personalized_samples=%d | batch_size=%d", num_samples, inference_batch_size)
    logger.info("[FedDPA-F Inference] decoding max_new_tokens=%d | num_beams=%d", max_new_tokens, num_beams)
    logger.info("=" * 90)

    ttp_samples = [] if disable_ttp else load_full_test_dataset(
        dataset_name=dataset_name,
        num_samples=ttp_num_samples,
    )

    for client_id in clients:
        checkpoint_path = _checkpoint_path(checkpoint_dir, client_id)
        personalized_samples = load_personalized_samples(
            target_client=client_id,
            dataset_name=dataset_name,
            num_samples=num_samples,
        )
        auto_reference_samples = _load_local_reference_samples(
            client_id=client_id,
            dataset_name=dataset_name,
            num_instances=num_samples,
        ) if ttp_adapter_mode == "auto" or personalization_adapter_mode == "auto" else None

        logger.info("[FedDPA-F Inference] Loading %s from %s", client_id, checkpoint_path)
        model = load_adapter_model(
            hf_token=hf_token,
            checkpoint_path=checkpoint_path,
            device=device,
            model_id=model_id,
        )

        client_rows = [
            _evaluate_scope(
                model=model,
                tokenizer=tokenizer,
                device=device,
                client_id=client_id,
                dataset_name=dataset_name,
                scope_name="personalized_subset",
                samples=personalized_samples,
                output_dir=output_dir,
                inference_batch_size=inference_batch_size,
                save_predictions=save_predictions,
                adapter_mode=personalization_adapter_mode,
                auto_reference_samples=auto_reference_samples,
                auto_num_instances=auto_num_instances,
                auto_weight_scale=auto_weight_scale,
                auto_emb_type=auto_emb_type,
                auto_reference_strategy=auto_reference_strategy,
                static_global_weight=static_global_weight,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
        ]

        if not disable_ttp:
            client_rows.append(
                _evaluate_scope(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    client_id=client_id,
                    dataset_name=dataset_name,
                    scope_name="full_test_file",
                    samples=ttp_samples,
                    output_dir=output_dir,
                    inference_batch_size=inference_batch_size,
                    save_predictions=save_predictions,
                    adapter_mode=ttp_adapter_mode,
                    auto_reference_samples=auto_reference_samples,
                    auto_num_instances=auto_num_instances,
                    auto_weight_scale=auto_weight_scale,
                    auto_emb_type=auto_emb_type,
                    auto_reference_strategy=auto_reference_strategy,
                    static_global_weight=static_global_weight,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                )
            )

        _write_json(
            output_dir / "clients" / f"{client_id}.json",
            {
                "client": client_id,
                "dataset_name": dataset_name,
                "checkpoint": str(checkpoint_path),
                "rows": client_rows,
            },
        )
        summary_rows.extend(client_rows)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "method": "FedDPA-F",
        "model_id": model_id,
        "dataset_name": dataset_name,
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "clients": clients,
        "rows": summary_rows,
        "settings": {
            "personalization_adapter_mode": personalization_adapter_mode,
            "ttp_adapter_mode": ttp_adapter_mode,
            "auto_num_instances": auto_num_instances,
            "auto_weight_scale": auto_weight_scale,
            "auto_emb_type": auto_emb_type,
            "auto_reference_strategy": auto_reference_strategy,
            "static_global_weight": static_global_weight,
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "seed": seed,
        },
    }
    _write_json(output_dir / "summary.json", summary)

    logger.info("\nFINAL FEDDPA-F INFERENCE REPORT")
    logger.info("=" * 90)
    header = f"{'Client':<10} | {'Scope':<20} | {'F1':>8} | {'BLEU-4':>8} | {'ROUGE-1':>8} | {'ROUGE-L':>8} | {'METEOR':>8}"
    logger.info(header)
    logger.info("-" * 90)
    for row in summary_rows:
        logger.info(
            f"{row['client']:<10} | {row['sample_scope']:<20} | {row['F1']:>8.2f} | "
            f"{row['BLEU']:>8.2f} | {row['ROUGE-1']:>8.2f} | {row['ROUGE-L']:>8.2f} | {row['METEOR']:>8.2f}"
        )
    logger.info("=" * 90)
    logger.info("[FedDPA-F Inference] Saved summary: %s", output_dir / "summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FedDPA-F inference for personalized dual-LoRA checkpoints.")
    parser.add_argument("--checkpoint_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--log_dir", type=Path, default=None)
    parser.add_argument("--run_tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--dataset_name", default="dataset1")
    parser.add_argument("--clients", nargs="*", default=None, help="Client ids, e.g. client_1 client_2 or 1 2.")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--ttp_num_samples", type=int, default=None)
    parser.add_argument("--disable_ttp", action="store_true")
    parser.add_argument("--inference_batch_size", type=int, default=8)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--personalization_adapter_mode", choices=["local_only", "global_only", "static", "auto"], default="static")
    parser.add_argument("--ttp_adapter_mode", choices=["local_only", "global_only", "static", "auto"], default="auto")
    parser.add_argument("--auto_num_instances", type=int, default=1)
    parser.add_argument("--auto_weight_scale", "--auto_lambda", dest="auto_weight_scale", type=float, default=1.0)
    parser.add_argument("--auto_emb_type", choices=["last", "avg"], default="last")
    parser.add_argument("--auto_reference_strategy", choices=["random_per_sample", "fixed"], default="random_per_sample")
    parser.add_argument("--static_global_weight", type=float, default=0.5)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = _build_run_id(args.run_tag)
    output_dir = args.output_dir or (DEFAULT_OUTPUT_PARENT / run_id)
    log_dir = args.log_dir or (DEFAULT_LOG_PARENT / run_id)
    _add_file_logger(log_dir)
    run_feddpa_f_inference(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=output_dir,
        model_id=args.model_id,
        dataset_name=args.dataset_name,
        clients=args.clients,
        num_samples=args.num_samples,
        ttp_num_samples=args.ttp_num_samples,
        disable_ttp=args.disable_ttp,
        inference_batch_size=args.inference_batch_size,
        save_predictions=args.save_predictions,
        personalization_adapter_mode=args.personalization_adapter_mode,
        ttp_adapter_mode=args.ttp_adapter_mode,
        auto_num_instances=args.auto_num_instances,
        auto_weight_scale=args.auto_weight_scale,
        auto_emb_type=args.auto_emb_type,
        auto_reference_strategy=args.auto_reference_strategy,
        static_global_weight=args.static_global_weight,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
