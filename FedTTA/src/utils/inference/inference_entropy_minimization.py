from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import AutoTokenizer


PROJECT_SRC = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PROJECT_SRC.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    path for path in sys.path
    if path and Path(path).resolve() != SCRIPT_DIR
]
sys.path.insert(0, str(PROJECT_SRC))

from utils.inference.common import MODEL_ID, PROMPT_INPUT, PROMPT_NO_INPUT, checkpoint_suffix, logger, set_seed
from utils.inference.data import load_full_test_dataset, load_personalized_samples
from utils.inference.inference_fedDPA import _compute_generation_metrics, _discover_clients, _discover_latest_checkpoint_dir
from utils.inference.model_loader import load_adapter_model


DEFAULT_CHECKPOINT_ROOT = PROJECT_SRC / "checkpoints"
DEFAULT_OUTPUT_PARENT = PROJECT_ROOT / "outputs"
DEFAULT_LOG_PARENT = PROJECT_ROOT / "logs"
DEFAULT_RUN_TAG = "inference_entropy_minimization"


def _build_run_id(run_tag: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_tag}"


def _add_file_logger(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "inference_entropy_minimization.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(filename)s - %(message)s"))
    root_logger = logging.getLogger()
    if not any(isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path for handler in root_logger.handlers):
        root_logger.addHandler(file_handler)
    logger.info("[Entropy-Min Inference] Log file: %s", log_path)


def _load_hf_token() -> str:
    load_dotenv(PROJECT_SRC / ".env")
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN is not set. Put it in src/.env or export it.")
    return hf_token


def _normalize_client_id(value: str | int) -> str:
    text = str(value)
    return text if text.startswith("client_") else f"client_{text}"


def _checkpoint_path(checkpoint_dir: Path, client_id: str) -> Path:
    path = checkpoint_dir / f"dual_lora_adapter_client_{checkpoint_suffix(client_id)}.pth"
    if not path.exists():
        raise FileNotFoundError(f"Missing FedDPA-F checkpoint for {client_id}: {path}")
    return path


def _build_prompt(sample: dict[str, str]) -> str:
    instruction = sample["question"].strip()
    input_text = (sample.get("input") or "").strip()
    if input_text:
        return PROMPT_INPUT.format(instruction=instruction, input=input_text)
    return PROMPT_NO_INPUT.format(instruction=instruction)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def _set_adapter_weights(model: Any, *, global_weight: float, local_weight: float) -> None:
    model.dual_lora_adapter.set_adapter_weights(
        global_weight=global_weight,
        local_weight=local_weight,
    )


def _last_token_logits(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor, *, global_weight: float, local_weight: float) -> torch.Tensor:
    _set_adapter_weights(model, global_weight=global_weight, local_weight=local_weight)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    return outputs.logits[:, -1, :].float()


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1).mean()


def _optimize_entropy_weight(
    global_logits: torch.Tensor,
    local_logits: torch.Tensor,
    *,
    steps: int,
    lr: float,
    init_e: float,
) -> tuple[float, float]:
    e = torch.tensor(float(init_e), device=global_logits.device, dtype=torch.float32, requires_grad=True)
    best_e = float(e.detach().item())
    best_entropy = float("inf")

    for _ in range(steps):
        mixed_logits = e * global_logits + (1.0 - e) * local_logits
        entropy = _entropy_from_logits(mixed_logits)
        entropy_value = float(entropy.detach().item())
        if entropy_value < best_entropy:
            best_entropy = entropy_value
            best_e = float(e.detach().item())

        entropy.backward()
        with torch.no_grad():
            if e.grad is not None:
                e -= lr * e.grad
            e.clamp_(0.0, 1.0)
            e.grad = None

    return best_e, best_entropy


def _entropy_min_generate_one(
    *,
    model: Any,
    tokenizer: AutoTokenizer,
    sample: dict[str, str],
    device: torch.device,
    entropy_steps: int,
    entropy_lr: float,
    init_e: float,
    max_new_tokens: int,
    decoding_strategy: str,
    num_beams: int,
) -> tuple[str, dict[str, float]]:
    prompt = _build_prompt(sample)
    inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    global_logits = _last_token_logits(model, input_ids, attention_mask, global_weight=1.0, local_weight=0.0)
    local_logits = _last_token_logits(model, input_ids, attention_mask, global_weight=0.0, local_weight=1.0)
    best_e, best_entropy = _optimize_entropy_weight(
        global_logits,
        local_logits,
        steps=entropy_steps,
        lr=entropy_lr,
        init_e=init_e,
    )

    first_step_logits = best_e * global_logits + (1.0 - best_e) * local_logits
    if decoding_strategy == "greedy":
        generated_tokens = _greedy_decode_with_fixed_weight(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            first_step_logits=first_step_logits,
            best_e=best_e,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
    elif decoding_strategy == "beam":
        generated_tokens = _beam_decode_with_fixed_weight(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            first_step_logits=first_step_logits,
            best_e=best_e,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            eos_token_id=tokenizer.eos_token_id,
        )
    else:
        raise ValueError(f"Unsupported decoding_strategy: {decoding_strategy}")

    text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return text, {
        "entropy_e": best_e,
        "entropy": best_entropy,
        "global_weight": best_e,
        "local_weight": 1.0 - best_e,
    }


def _greedy_decode_with_fixed_weight(
    *,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    first_step_logits: torch.Tensor,
    best_e: float,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> list[int]:
    generated_tokens: list[int] = []
    current_input_ids = input_ids
    current_attention_mask = attention_mask
    for token_idx in range(max_new_tokens):
        if token_idx == 0:
            mixed_logits = first_step_logits
        else:
            global_logits = _last_token_logits(model, current_input_ids, current_attention_mask, global_weight=1.0, local_weight=0.0)
            local_logits = _last_token_logits(model, current_input_ids, current_attention_mask, global_weight=0.0, local_weight=1.0)
            mixed_logits = best_e * global_logits + (1.0 - best_e) * local_logits

        next_token = torch.argmax(mixed_logits, dim=-1)
        token_id = int(next_token.item())
        if token_id == eos_token_id:
            break

        generated_tokens.append(token_id)
        current_input_ids = torch.cat([current_input_ids, next_token.view(1, 1)], dim=1)
        current_attention_mask = torch.cat(
            [
                current_attention_mask,
                torch.ones((1, 1), device=current_input_ids.device, dtype=current_attention_mask.dtype),
            ],
            dim=1,
        )

    return generated_tokens


def _beam_decode_with_fixed_weight(
    *,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    first_step_logits: torch.Tensor,
    best_e: float,
    max_new_tokens: int,
    num_beams: int,
    eos_token_id: int | None,
) -> list[int]:
    if num_beams <= 1:
        return _greedy_decode_with_fixed_weight(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            first_step_logits=first_step_logits,
            best_e=best_e,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )

    beams: list[tuple[list[int], float, bool]] = [([], 0.0, False)]
    device = input_ids.device

    for token_idx in range(max_new_tokens):
        candidates: list[tuple[list[int], float, bool]] = []
        for tokens, score, ended in beams:
            if ended:
                candidates.append((tokens, score, True))
                continue

            if tokens:
                token_tensor = torch.tensor([tokens], device=device, dtype=input_ids.dtype)
                beam_input_ids = torch.cat([input_ids, token_tensor], dim=1)
                beam_attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones((1, len(tokens)), device=device, dtype=attention_mask.dtype),
                    ],
                    dim=1,
                )
                global_logits = _last_token_logits(model, beam_input_ids, beam_attention_mask, global_weight=1.0, local_weight=0.0)
                local_logits = _last_token_logits(model, beam_input_ids, beam_attention_mask, global_weight=0.0, local_weight=1.0)
                mixed_logits = best_e * global_logits + (1.0 - best_e) * local_logits
            else:
                mixed_logits = first_step_logits

            log_probs = F.log_softmax(mixed_logits[0], dim=-1)
            top_values, top_indices = torch.topk(log_probs, k=num_beams)
            for log_prob, token_id_tensor in zip(top_values, top_indices):
                token_id = int(token_id_tensor.item())
                next_tokens = tokens + [token_id]
                next_score = score + float(log_prob.item())
                candidates.append((next_tokens, next_score, token_id == eos_token_id))

        candidates.sort(key=lambda item: item[1] / max(1, len(item[0])), reverse=True)
        beams = candidates[:num_beams]
        if all(ended for _, _, ended in beams):
            break

    best_tokens, _score, _ended = max(beams, key=lambda item: item[1] / max(1, len(item[0])))
    if eos_token_id is not None and eos_token_id in best_tokens:
        best_tokens = best_tokens[:best_tokens.index(eos_token_id)]
    return best_tokens


def _entropy_min_generate(
    *,
    model: Any,
    tokenizer: AutoTokenizer,
    samples: list[dict[str, str]],
    device: torch.device,
    entropy_steps: int,
    entropy_lr: float,
    init_e: float,
    max_new_tokens: int,
    decoding_strategy: str,
    num_beams: int,
) -> tuple[list[str], list[dict[str, float]]]:
    predictions: list[str] = []
    weight_rows: list[dict[str, float]] = []
    for sample_idx, sample in enumerate(samples, start=1):
        prediction, weights = _entropy_min_generate_one(
            model=model,
            tokenizer=tokenizer,
            sample=sample,
            device=device,
            entropy_steps=entropy_steps,
            entropy_lr=entropy_lr,
            init_e=init_e,
            max_new_tokens=max_new_tokens,
            decoding_strategy=decoding_strategy,
            num_beams=num_beams,
        )
        predictions.append(prediction)
        weight_rows.append(weights)
        if sample_idx % 25 == 0:
            logger.info("[Entropy-Min] generated %d/%d samples", sample_idx, len(samples))
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
    save_predictions: bool,
    entropy_steps: int,
    entropy_lr: float,
    init_e: float,
    max_new_tokens: int,
    decoding_strategy: str,
    num_beams: int,
) -> dict[str, Any]:
    logger.info("Evaluating entropy-min: %s | %s | samples=%d", client_id, scope_name, len(samples))
    predictions, weight_rows = _entropy_min_generate(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        device=device,
        entropy_steps=entropy_steps,
        entropy_lr=entropy_lr,
        init_e=init_e,
        max_new_tokens=max_new_tokens,
        decoding_strategy=decoding_strategy,
        num_beams=num_beams,
    )
    metrics, task_metrics = _compute_generation_metrics(samples, predictions)
    row = {
        "client": client_id,
        "dataset_name": dataset_name,
        "model": "Entropy-Min",
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


def run_entropy_minimization_inference(
    *,
    checkpoint_dir: Path | None = None,
    output_dir: Path,
    model_id: str = MODEL_ID,
    dataset_name: str = "dataset1",
    clients: list[str] | None = None,
    num_samples: int = 200,
    ttp_num_samples: int | None = None,
    disable_ttp: bool = False,
    save_predictions: bool = False,
    entropy_steps: int = 20,
    entropy_lr: float = 0.1,
    init_e: float = 0.5,
    max_new_tokens: int = 80,
    decoding_strategy: str = "greedy",
    num_beams: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    set_seed(seed)
    checkpoint_dir = checkpoint_dir or _discover_latest_checkpoint_dir(DEFAULT_CHECKPOINT_ROOT)
    clients = [_normalize_client_id(client) for client in clients] if clients else _discover_clients(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_token = _load_hf_token()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ttp_samples = [] if disable_ttp else load_full_test_dataset(dataset_name=dataset_name, num_samples=ttp_num_samples)
    summary_rows: list[dict[str, Any]] = []

    logger.info("[Entropy-Min Inference] checkpoint_dir=%s", checkpoint_dir)
    logger.info("[Entropy-Min Inference] output_dir=%s", output_dir)
    logger.info(
        "[Entropy-Min Inference] entropy_steps=%d | entropy_lr=%.4f | init_e=%.2f | decoding=%s | num_beams=%d",
        entropy_steps,
        entropy_lr,
        init_e,
        decoding_strategy,
        num_beams,
    )

    for client_id in clients:
        checkpoint_path = _checkpoint_path(checkpoint_dir, client_id)
        personalized_samples = load_personalized_samples(
            target_client=client_id,
            dataset_name=dataset_name,
            num_samples=num_samples,
        )

        logger.info("[Entropy-Min Inference] Loading %s from %s", client_id, checkpoint_path)
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
                save_predictions=save_predictions,
                entropy_steps=entropy_steps,
                entropy_lr=entropy_lr,
                init_e=init_e,
                max_new_tokens=max_new_tokens,
                decoding_strategy=decoding_strategy,
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
                    save_predictions=save_predictions,
                    entropy_steps=entropy_steps,
                    entropy_lr=entropy_lr,
                    init_e=init_e,
                    max_new_tokens=max_new_tokens,
                    decoding_strategy=decoding_strategy,
                    num_beams=num_beams,
                )
            )

        _write_json(output_dir / "clients" / f"{client_id}.json", {"client": client_id, "dataset_name": dataset_name, "checkpoint": str(checkpoint_path), "rows": client_rows})
        summary_rows.extend(client_rows)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "method": "Entropy-Min",
        "model_id": model_id,
        "dataset_name": dataset_name,
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "clients": clients,
        "rows": summary_rows,
        "settings": {
            "entropy_steps": entropy_steps,
            "entropy_lr": entropy_lr,
            "init_e": init_e,
            "max_new_tokens": max_new_tokens,
            "decoding_strategy": decoding_strategy,
            "num_beams": num_beams,
            "seed": seed,
        },
    }
    _write_json(output_dir / "summary.json", summary)
    logger.info("[Entropy-Min Inference] Saved summary: %s", output_dir / "summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run entropy-minimization test-time adaptation inference.")
    parser.add_argument("--checkpoint_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--log_dir", type=Path, default=None)
    parser.add_argument("--run_tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--dataset_name", default="dataset1")
    parser.add_argument("--clients", nargs="*", default=None)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--ttp_num_samples", type=int, default=None)
    parser.add_argument("--disable_ttp", action="store_true")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--entropy_steps", type=int, default=20)
    parser.add_argument("--entropy_lr", type=float, default=0.1)
    parser.add_argument("--init_e", type=float, default=0.5)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--decoding_strategy", choices=["greedy", "beam"], default="greedy")
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = _build_run_id(args.run_tag)
    output_dir = args.output_dir or (DEFAULT_OUTPUT_PARENT / run_id)
    log_dir = args.log_dir or (DEFAULT_LOG_PARENT / run_id)
    _add_file_logger(log_dir)
    run_entropy_minimization_inference(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=output_dir,
        model_id=args.model_id,
        dataset_name=args.dataset_name,
        clients=args.clients,
        num_samples=args.num_samples,
        ttp_num_samples=args.ttp_num_samples,
        disable_ttp=args.disable_ttp,
        save_predictions=args.save_predictions,
        entropy_steps=args.entropy_steps,
        entropy_lr=args.entropy_lr,
        init_e=args.init_e,
        max_new_tokens=args.max_new_tokens,
        decoding_strategy=args.decoding_strategy,
        num_beams=args.num_beams,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
