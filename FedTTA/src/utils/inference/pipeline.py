import json
import os
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer

from utils.inference.common import CHECKPOINT_DIR, MODEL_ID, checkpoint_suffix, logger, set_seed
from utils.inference.data import load_full_test_dataset, load_personalized_samples
from utils.inference.evaluator import evaluate_samples
from methods.lora.inference import load_lora_model
from utils.inference.model_loader import load_adapter_model, load_base_model


def write_result_json(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_inference(
    target_client: str,
    model_id: str = MODEL_ID,
    dataset_name: str = "dataset1",
    num_samples: int = 200,
    compare_base: bool = True,
    compare_local_finetuned: bool = True,
    compare_lora: bool = False,
    run_ttp: bool = True,
    ttp_num_samples: int | None = None,
    result_json: str | None = None,
    inference_batch_size: int = 8,
    lora_checkpoint_prefix: str = "lora_adapter",
):
    set_seed(42)
    project_src_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    )
    load_dotenv(os.path.join(project_src_dir, ".env"))
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN is not set.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    personalized_samples = load_personalized_samples(
        target_client=target_client,
        dataset_name=dataset_name,
        num_samples=num_samples,
    )
    ttp_samples = load_full_test_dataset(
        dataset_name=dataset_name,
        num_samples=ttp_num_samples,
    ) if run_ttp else []

    suffix = checkpoint_suffix(target_client)
    ours_ckpt = CHECKPOINT_DIR / f"dual_lora_adapter_client_{suffix}.pth"
    local_ckpt = CHECKPOINT_DIR / f"local_finetuned_dual_lora_client_{suffix}.pth"
    lora_ckpt = CHECKPOINT_DIR / f"{lora_checkpoint_prefix}_client_{suffix}.pth"
    configs: list[dict[str, Any]] = []

    if compare_base:
        configs.append(
            {
                "name": f"{target_client} | Base Frozen LLM",
                "short_name": "Base Frozen LLM",
                "mode": "base",
                "checkpoint": None,
                "samples": personalized_samples,
                "sample_scope": "personalized_subset",
            }
        )

    if compare_local_finetuned:
        configs.append(
            {
                "name": f"{target_client} | Local-finetuned",
                "short_name": "Local-finetuned",
                "mode": "adapter",
                "checkpoint": local_ckpt,
                "samples": personalized_samples,
                "sample_scope": "personalized_subset",
            }
        )

    if compare_lora:
        configs.append(
            {
                "name": f"{target_client} | DP-LoRA",
                "short_name": "DP-LoRA",
                "mode": "lora",
                "checkpoint": lora_ckpt,
                "samples": personalized_samples,
                "sample_scope": "personalized_subset",
            }
        )

    configs.append(
        {
            "name": f"{target_client} | Ours",
            "short_name": "Ours",
            "mode": "adapter",
            "checkpoint": ours_ckpt,
            "samples": personalized_samples,
            "sample_scope": "personalized_subset",
        }
    )

    if run_ttp:
        if compare_base:
            configs.append(
                {
                    "name": f"{target_client} | Base Frozen LLM | TTP",
                    "short_name": "Base Frozen LLM",
                    "mode": "base",
                    "checkpoint": None,
                    "samples": ttp_samples,
                    "sample_scope": "full_test_file",
                }
            )

        if compare_local_finetuned:
            configs.append(
                {
                    "name": f"{target_client} | Local-finetuned | TTP",
                    "short_name": "Local-finetuned",
                    "mode": "adapter",
                    "checkpoint": local_ckpt,
                    "samples": ttp_samples,
                    "sample_scope": "full_test_file",
                }
            )

        if compare_lora:
            configs.append(
                {
                    "name": f"{target_client} | DP-LoRA | TTP",
                    "short_name": "DP-LoRA",
                    "mode": "lora",
                    "checkpoint": lora_ckpt,
                    "samples": ttp_samples,
                    "sample_scope": "full_test_file",
                }
            )

        configs.append(
            {
                "name": f"{target_client} | Ours-TTP",
                "short_name": "Ours-TTP",
                "mode": "adapter",
                "checkpoint": ours_ckpt,
                "samples": ttp_samples,
                "sample_scope": "full_test_file",
            }
        )

    results = {config["name"]: {} for config in configs}
    json_rows: list[dict[str, Any]] = []

    logger.info("\n" + "=" * 90)
    logger.info(f"Inference target={target_client} | dataset={dataset_name}")
    logger.info(f"Personalized samples={len(personalized_samples)}")
    logger.info(f"Inference batch size={inference_batch_size}")
    if run_ttp:
        logger.info(f"TTP full-test samples={len(ttp_samples)}")
    logger.info("=" * 90)

    for config in configs:
        ckpt = config["checkpoint"]
        eval_samples = config["samples"]

        if ckpt is not None and not ckpt.exists():
            logger.warning(f"Skipping missing checkpoint: {ckpt}")
            continue
        if not eval_samples:
            logger.warning(f"Skipping empty eval set for {config['name']}")
            continue

        logger.info(f"Evaluating: {config['name']}")
        if config["mode"] == "base":
            model = load_base_model(hf_token=hf_token, model_id=model_id)
        elif config["mode"] == "lora":
            model = load_lora_model(model_id=model_id, checkpoint_path=ckpt, device=device)
        else:
            model = load_adapter_model(
                hf_token=hf_token,
                checkpoint_path=ckpt,
                device=device,
                model_id=model_id,
            )

        metric_dict = evaluate_samples(
            model=model,
            tokenizer=tokenizer,
            eval_samples=eval_samples,
            device=device,
            desc=config["name"],
            inference_batch_size=inference_batch_size,
        )
        results[config["name"]] = metric_dict
        json_rows.append(
            {
                "client": target_client,
                "dataset_name": dataset_name,
                "model": config["short_name"],
                "sample_scope": config["sample_scope"],
                "num_samples": len(eval_samples),
                **metric_dict,
            }
        )

        del model
        torch.cuda.empty_cache()

    logger.info("\nFINAL REPORT")
    logger.info("=" * 90)
    header = f"{'Model':<32} | {'F1':>8} | {'BLEU-4':>8} | {'ROUGE-1':>8} | {'ROUGE-L':>8} | {'METEOR':>8}"
    logger.info(header)
    logger.info("-" * 90)
    for name, metric_dict in results.items():
        if not metric_dict:
            continue
        row = (
            f"{name:<32} | {metric_dict['F1']:>8.2f} | {metric_dict['BLEU']:>8.2f} | "
            f"{metric_dict['ROUGE-1']:>8.2f} | {metric_dict['ROUGE-L']:>8.2f} | {metric_dict['METEOR']:>8.2f}"
        )
        logger.info(row)
    logger.info("=" * 90)

    write_result_json(
        result_json,
        {
            "target_client": target_client,
            "dataset_name": dataset_name,
            "rows": json_rows,
        },
    )
