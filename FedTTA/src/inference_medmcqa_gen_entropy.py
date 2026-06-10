"""MedMCQA 생성(generation) 기반 entropy-min 추론 — sequence per-token TTA.

cloze(보기 likelihood 채점)와 달리, 모델이 정답 보기 본문을 **토큰별로 생성**한다.
entropy_min은 **매 토큰마다** global/local 로짓을 따로 구해 entropy 최소화로 e_t를 새로
정하고 그 혼합 분포로 다음 토큰을 고른다(첫 토큰 고정 X = sequence 방식).

생성된 문자열을 4개 보기 중 가장 가까운 것에 매칭(정규화 exact → 토큰겹침 F1)하여
정답 letter로 환산, accuracy 측정.

methods:
  - fedit       : global LoRA 고정 생성 (gw=1)
  - local_only  : local LoRA 고정 생성 (lw=1)
  - equal       : gw=lw=0.5 고정 생성
  - entropy_min : 매 토큰 e_t* (sequence TTA) 생성  ← 핵심

사용: python src/inference_medmcqa_gen_entropy.py --checkpoint_dir src/checkpoints/exp3_4b_cloze
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

PROJECT_SRC = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_SRC.parent
sys.path.insert(0, str(PROJECT_SRC))

from data.loader import MEDMCQA_SUBJECTS, get_client_eval_dataset, get_full_test_eval_dataset, _load_medmcqa_examples  # noqa: E402
from utils.inference.common import PROMPT_NO_INPUT, logger, set_seed  # noqa: E402
from utils.inference.model_loader import load_adapter_model  # noqa: E402
from inference_medmcqa import optimize_e, embed_prompts, adapter_forward  # noqa: E402

LETTERS = ["A", "B", "C", "D"]
GEN_METHODS = ["base", "fedit", "local_only", "equal", "FedDPA", "entropy_min"]
DEFAULT_MODEL_ID = "/scratch2/0630rb/models/Qwen3-4B-Base"
DEFAULT_CHECKPOINT_DIR = PROJECT_SRC / "checkpoints"
DEFAULT_OUTPUT_PARENT = PROJECT_ROOT / "outputs"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def _tok_f1(a: str, b: str) -> float:
    ta, tb = set(_norm(a).split()), set(_norm(b).split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    if inter == 0:
        return 0.0
    p, r = inter / len(ta), inter / len(tb)
    return 2 * p * r / (p + r)


def match_option(text: str, options: dict[str, str]) -> str:
    """생성 문자열 -> 가장 가까운 보기 letter. 정규화 exact 우선, 없으면 토큰 F1 최대."""
    nt = _norm(text)
    for L in LETTERS:
        if _norm(options[L]) == nt and nt:
            return L
    best, bestf = "A", -1.0
    for L in LETTERS:
        f = _tok_f1(text, options[L])
        if f > bestf:
            bestf, best = f, L
    return best


@torch.no_grad()
def _last_logits(model, ids, attn, *, gw, lw):
    model.dual_lora_adapter.set_adapter_weights(global_weight=gw, local_weight=lw)
    out = model(input_ids=ids, attention_mask=attn, return_dict=True)
    return out.logits[0, -1, :].float()


def generate_fixed(model, tok, ids, attn, *, gw, lw, max_new_tokens, device):
    """고정 가중치(gw,lw)로 greedy 생성."""
    gen = []
    for _ in range(max_new_tokens):
        logits = _last_logits(model, ids, attn, gw=gw, lw=lw)
        nt = int(logits.argmax())
        if nt == tok.eos_token_id:
            break
        gen.append(nt)
        ids = torch.cat([ids, torch.tensor([[nt]], device=device)], dim=1)
        attn = torch.cat([attn, torch.ones((1, 1), device=device, dtype=attn.dtype)], dim=1)
    return tok.decode(gen, skip_special_tokens=True).strip()


def generate_entropy_seq(model, tok, ids, attn, *, max_new_tokens, steps, lr, temperature, device):
    """sequence per-token entropy-min: 매 토큰 e_t* 재최적화 후 그 혼합으로 생성."""
    gen, e_hist = [], []
    for _ in range(max_new_tokens):
        zg = _last_logits(model, ids, attn, gw=1.0, lw=0.0)
        zl = _last_logits(model, ids, attn, gw=0.0, lw=1.0)
        e = optimize_e(zg, zl, domain="fullvocab", letter_id_tensor=None,
                       steps=steps, lr=lr, temperature=temperature)
        e_hist.append(e)
        mix = e * zg + (1.0 - e) * zl
        nt = int(mix.argmax())
        if nt == tok.eos_token_id:
            break
        gen.append(nt)
        ids = torch.cat([ids, torch.tensor([[nt]], device=device)], dim=1)
        attn = torch.cat([attn, torch.ones((1, 1), device=device, dtype=attn.dtype)], dim=1)
    text = tok.decode(gen, skip_special_tokens=True).strip()
    e_mean = sum(e_hist) / len(e_hist) if e_hist else 0.5
    return text, e_mean


def evaluate_samples(model, tok, samples, device, *, max_new_tokens, steps, lr, temperature,
                     ref_embs=None, fedDPA_w=0.5, fedDPA_num_ref=5, seed=42):
    import random
    rng = random.Random(seed)
    do_feddpa = ref_embs is not None and ref_embs.shape[0] > 0
    correct = {m: 0 for m in GEN_METHODS}
    by_subject = {m: defaultdict(lambda: [0, 0]) for m in GEN_METHODS}
    e_values, w_local_values = [], []
    n = 0
    for sample in samples:
        gold = sample["answer_letter"]
        subj = sample.get("subject_name", "")
        options = sample["options"]
        prompt = PROMPT_NO_INPUT.format(instruction=sample["question"].strip())
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(device)
        attn = torch.ones_like(ids)

        preds = {}
        preds["base"] = match_option(generate_fixed(model, tok, ids, attn, gw=0.0, lw=0.0,
                                                     max_new_tokens=max_new_tokens, device=device), options)
        preds["fedit"] = match_option(generate_fixed(model, tok, ids, attn, gw=1.0, lw=0.0,
                                                      max_new_tokens=max_new_tokens, device=device), options)
        preds["local_only"] = match_option(generate_fixed(model, tok, ids, attn, gw=0.0, lw=1.0,
                                                           max_new_tokens=max_new_tokens, device=device), options)
        preds["equal"] = match_option(generate_fixed(model, tok, ids, attn, gw=0.5, lw=0.5,
                                                      max_new_tokens=max_new_tokens, device=device), options)
        # FedDPA cosine: 입력 임베딩 유사도로 w_local 1회 결정 후 고정 가중치 생성
        if do_feddpa:
            _, test_emb = adapter_forward(model, ids, attn, global_weight=1.0, local_weight=0.0, emb_type="last")
            k = min(fedDPA_num_ref, ref_embs.shape[0])
            idx = rng.sample(range(ref_embs.shape[0]), k)
            w_local = float(F.cosine_similarity(test_emb.unsqueeze(0), ref_embs[idx], dim=1).mean()) * fedDPA_w
            w_local_values.append(w_local)
            preds["FedDPA"] = match_option(generate_fixed(model, tok, ids, attn, gw=1.0 - w_local, lw=w_local,
                                                          max_new_tokens=max_new_tokens, device=device), options)
        else:
            preds["FedDPA"] = preds["fedit"]
        em_text, e_mean = generate_entropy_seq(model, tok, ids, attn, max_new_tokens=max_new_tokens,
                                               steps=steps, lr=lr, temperature=temperature, device=device)
        preds["entropy_min"] = match_option(em_text, options)
        e_values.append(e_mean)

        for m in GEN_METHODS:
            ok = int(preds[m] == gold)
            correct[m] += ok
            by_subject[m][subj][0] += ok
            by_subject[m][subj][1] += 1
        n += 1
        if n % 25 == 0:
            logger.info("  gen-eval %d/%d", n, len(samples))

    acc = {m: (correct[m] / n if n else 0.0) for m in GEN_METHODS}
    subj_acc = {m: {s: (c / t if t else 0.0) for s, (c, t) in sorted(by_subject[m].items())} for m in GEN_METHODS}
    def stat(v):
        return {"mean": sum(v) / len(v) if v else None, "min": min(v) if v else None, "max": max(v) if v else None}
    return {"n": n, "accuracy": acc, "accuracy_by_subject": subj_acc,
            "e_star": {"entropy_min": stat(e_values)}, "feddpa_w_local": stat(w_local_values)}


def discover_clients(ckpt: Path) -> list[str]:
    found = [f"client_{i}" for i in range(1, len(MEDMCQA_SUBJECTS) + 1)
             if (ckpt / f"dual_lora_adapter_client_{i}.pth").exists()]
    if not found:
        raise FileNotFoundError(f"No dual_lora_adapter_client_*.pth in {ckpt}")
    return found


def main() -> None:
    p = argparse.ArgumentParser(description="MedMCQA generation-based sequence entropy-min inference.")
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--checkpoint_dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--clients", nargs="*", default=None)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--cross_num_samples", type=int, default=None)
    p.add_argument("--skip_cross", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--entropy_steps", type=int, default=20)
    p.add_argument("--entropy_lr", type=float, default=0.1)
    p.add_argument("--entropy_temperature", type=float, default=1.0)
    p.add_argument("--feddpa_w", type=float, default=0.5)
    p.add_argument("--feddpa_num_ref", type=int, default=5)
    p.add_argument("--feddpa_ref_pool", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_infer_gen_entropy"
    output_dir = args.output_dir or (DEFAULT_OUTPUT_PARENT / run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_id, token=None)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    clients = args.clients or discover_clients(args.checkpoint_dir)
    cross = None if args.skip_cross else get_full_test_eval_dataset(dataset_name="medmcqa", num_samples=args.cross_num_samples)

    ekw = dict(max_new_tokens=args.max_new_tokens, steps=args.entropy_steps,
               lr=args.entropy_lr, temperature=args.entropy_temperature)
    rows: list[dict[str, Any]] = []
    for cid in clients:
        ckpt = args.checkpoint_dir / f"dual_lora_adapter_client_{cid.split('_')[-1]}.pth"
        logger.info("=" * 70)
        logger.info("[gen-entropy] %s | %s", cid, ckpt)
        model = load_adapter_model(hf_token=None, checkpoint_path=ckpt, device=device, model_id=args.model_id)
        # FedDPA cosine ref = own-train 프롬프트 임베딩 (global adapter, last token)
        tr = _load_medmcqa_examples(cid, split="train", limit=args.feddpa_ref_pool)
        train_prompts = [PROMPT_NO_INPUT.format(instruction=(e.get("instruction") or "").strip()) for e in tr]
        ref_embs = embed_prompts(model, tok, train_prompts, device, global_weight=1.0, local_weight=0.0, emb_type="last")
        fd = dict(ref_embs=ref_embs, fedDPA_w=args.feddpa_w, fedDPA_num_ref=args.feddpa_num_ref, seed=args.seed)
        own = get_client_eval_dataset(cid, dataset_name="medmcqa", num_samples=args.num_samples)
        logger.info("[gen-entropy] %s own (n=%d)", cid, len(own))
        rows.append({"client": cid, "scope": "own", **evaluate_samples(model, tok, own, device, **ekw, **fd)})
        if cross is not None:
            logger.info("[gen-entropy] %s cross (n=%d)", cid, len(cross))
            rows.append({"client": cid, "scope": "cross", **evaluate_samples(model, tok, cross, device, **ekw, **fd)})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {"model_id": args.model_id, "checkpoint_dir": str(args.checkpoint_dir),
               "scoring": "generation_seq_entropy", "methods": GEN_METHODS, "clients": clients,
               "settings": {"max_new_tokens": args.max_new_tokens, "entropy_steps": args.entropy_steps,
                            "entropy_lr": args.entropy_lr, "entropy_temperature": args.entropy_temperature,
                            "num_samples": args.num_samples, "seed": args.seed},
               "rows": rows}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[gen-entropy] saved -> %s", output_dir / "summary.json")


if __name__ == "__main__":
    main()
