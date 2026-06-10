from typing import Any

import nltk
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.data import find
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.inference.common import PROMPT_INPUT, PROMPT_NO_INPUT, compute_f1

def _has_nltk_resource(resource_path: str) -> bool:
    try:
        find(resource_path)
        return True
    except LookupError:
        return False


_METEOR_AVAILABLE = _has_nltk_resource("corpora/wordnet")


def _build_prompts(samples: list[dict[str, str]]) -> list[str]:
    prompts = []
    for sample in samples:
        instruction = sample["question"].strip()
        input_text = (sample.get("input") or "").strip()
        if input_text:
            prompts.append(PROMPT_INPUT.format(instruction=instruction, input=input_text))
        else:
            prompts.append(PROMPT_NO_INPUT.format(instruction=instruction))
    return prompts


def generate_answers(
    model: Any,
    tokenizer: AutoTokenizer,
    samples: list[dict[str, str]],
    device: torch.device,
    batch_size: int = 32,
    max_new_tokens: int = 80,
    num_beams: int = 4,
) -> list[str]:
    generated_texts: list[str] = []

    for start_idx in range(0, len(samples), batch_size):
        batch_samples = samples[start_idx:start_idx + batch_size]
        batch_prompts = _build_prompts(batch_samples)
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        prompt_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        for idx in range(len(batch_samples)):
            generated_tokens = outputs[idx][prompt_length:]
            generated_texts.append(tokenizer.decode(generated_tokens, skip_special_tokens=True).strip())

    return generated_texts


def evaluate_samples(
    model: Any,
    tokenizer: AutoTokenizer,
    eval_samples: list[dict[str, str]],
    device: torch.device,
    desc: str,
    inference_batch_size: int = 32,
) -> dict[str, float]:
    rouge_sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    smoothie = SmoothingFunction().method1
    metrics = {"F1": 0.0, "BLEU": 0.0, "ROUGE-1": 0.0, "ROUGE-L": 0.0, "METEOR": 0.0}

    for start_idx in tqdm(range(0, len(eval_samples), inference_batch_size), desc=desc, leave=False):
        batch_samples = eval_samples[start_idx:start_idx + inference_batch_size]
        batch_questions = [sample["question"] for sample in batch_samples]
        batch_generated = generate_answers(
            model=model,
            tokenizer=tokenizer,
            samples=batch_samples,
            device=device,
            batch_size=inference_batch_size,
        )

        for sample, generated in zip(batch_samples, batch_generated):
            reference = sample["reference"].strip()
            rouge_scores = rouge_sc.score(reference, generated)

            metrics["F1"] += compute_f1(generated, reference)
            metrics["BLEU"] += sentence_bleu([reference.split()], generated.split(), smoothing_function=smoothie)
            metrics["ROUGE-1"] += rouge_scores["rouge1"].fmeasure
            metrics["ROUGE-L"] += rouge_scores["rougeL"].fmeasure
            if _METEOR_AVAILABLE:
                metrics["METEOR"] += meteor_score([reference.split()], generated.split())

    for key in metrics:
        metrics[key] = (metrics[key] / len(eval_samples)) * 100
    return metrics
