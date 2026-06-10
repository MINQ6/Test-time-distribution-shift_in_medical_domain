"""MedMCQA cloze-likelihood 추론 (letter 대신 보기 본문 likelihood).

각 보기 o의 점수 = 길이정규화 log P(o_text | prompt). adapter별로 4개 보기 점수
z = [s_A, s_B, s_C, s_D] 를 구한다 (4개 보기를 batched forward 1회).

method (정답 = 4개 점수 argmax):
  - base        : LoRA 없음 (frozen LLM)
  - local_only  : local LoRA만
  - fedit       : global LoRA만 (FedAvg)
  - equal       : 0.5/0.5
  - FedDPA      : cosine routing (test↔own-train 임베딩)
  - fedalt      : 학습된 mixer α(x) 로 Individual/RoW 결합 (mixer ckpt 있을 때)
  - entropy_min : 4-way 점수 분포의 entropy 최소화로 e* 결정 (TTA)

cloze는 보기가 멀티토큰이라 점수가 one-hot로 붕괴하지 않음 → entropy 신호가 letter보다 풍부.

사용: python src/inference_medmcqa_cloze.py --model_id <4B> --checkpoint_dir src/checkpoints/exp3_4b_cloze
"""
from __future__ import annotations

import argparse
import json
import random
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

from data.loader import MEDMCQA_SUBJECTS, get_client_eval_dataset, get_full_test_eval_dataset  # noqa: E402
from utils.inference.common import PROMPT_NO_INPUT, logger, set_seed  # noqa: E402
from utils.inference.model_loader import load_adapter_model  # noqa: E402
from inference_medmcqa import adapter_forward, embed_prompts, optimize_e  # noqa: E402

LETTERS = ["A", "B", "C", "D"]
METHODS = ["base", "local_only", "fedit", "equal", "FedDPA", "fedalt", "entropy_min"]
SCOPE_LABEL = {"own": "Personalization", "cross": "TTG (Test-time generalization)"}
DEFAULT_MODEL_ID = "/scratch2/0630rb/models/Qwen3-4B-Base"
DEFAULT_CHECKPOINT_DIR = PROJECT_SRC / "checkpoints"
DEFAULT_OUTPUT_PARENT = PROJECT_ROOT / "outputs"


@torch.no_grad()
def cloze_scores(model, tokenizer, prompt, options, device, *, global_weight, local_weight):
    """4개 보기의 길이정규화 log-likelihood [4] (순서 A,B,C,D). batched forward 1회."""
    model.dual_lora_adapter.set_adapter_weights(global_weight=global_weight, local_weight=local_weight)
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids[0].to(device)
    plen = prompt_ids.shape[0]
    seqs, optlens = [], []
    for letter in LETTERS:
        oid = tokenizer(options[letter], add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
        if oid.shape[0] == 0:  # 빈 보기 방어
            oid = tokenizer(" ", add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
        seqs.append(torch.cat([prompt_ids, oid]))
        optlens.append(oid.shape[0])
    maxlen = max(s.shape[0] for s in seqs)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((4, maxlen), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((4, maxlen), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        input_ids[i, :s.shape[0]] = s
        attn[i, :s.shape[0]] = 1
    out = model(input_ids=input_ids, attention_mask=attn, return_dict=True)
    logits = out.logits.float()  # [4, maxlen, V] — causal라 뒤쪽 pad는 옵션 토큰 logit에 영향 없음
    scores = []
    for i in range(4):
        ol = optlens[i]
        lp = F.log_softmax(logits[i, plen - 1:plen + ol - 1], dim=-1)  # [ol, V]
        tgt = seqs[i][plen:plen + ol]
        scores.append(lp[range(ol), tgt].sum() / max(1, ol))  # 길이정규화
    return torch.stack(scores)  # [4]


@torch.no_grad()
def cloze_scores_mixer(model, tokenizer, prompt, options, device):
    """FedALT: 학습된 per-token mixer α(x) 로 Individual/RoW 결합한 4점수 [4].

    set_use_mixer(True) 상태에서 forward → mixer가 layer 내부에서 결합. 점수 계산은
    cloze_scores와 동일(길이정규화 log-likelihood). 호출 후 use_mixer는 복구한다.
    """
    model.dual_lora_adapter.set_use_mixer(True)
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids[0].to(device)
    plen = prompt_ids.shape[0]
    seqs, optlens = [], []
    for letter in LETTERS:
        oid = tokenizer(options[letter], add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
        if oid.shape[0] == 0:
            oid = tokenizer(" ", add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
        seqs.append(torch.cat([prompt_ids, oid]))
        optlens.append(oid.shape[0])
    maxlen = max(s.shape[0] for s in seqs)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((4, maxlen), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((4, maxlen), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        input_ids[i, :s.shape[0]] = s
        attn[i, :s.shape[0]] = 1
    out = model(input_ids=input_ids, attention_mask=attn, return_dict=True)
    logits = out.logits.float()
    scores = []
    for i in range(4):
        ol = optlens[i]
        lp = F.log_softmax(logits[i, plen - 1:plen + ol - 1], dim=-1)
        tgt = seqs[i][plen:plen + ol]
        scores.append(lp[range(ol), tgt].sum() / max(1, ol))
    alpha_mean = model.dual_lora_adapter.alpha_indiv_mean()
    model.dual_lora_adapter.set_use_mixer(False)
    return torch.stack(scores), alpha_mean  # [4], float|None


def argmax_letter(scores) -> str:
    return LETTERS[int(scores.argmax())]


def evaluate_samples(model, tokenizer, samples, device, *, entropy_steps, entropy_lr, entropy_temperature,
                     ref_embs=None, fedDPA_emb="global", fedDPA_emb_type="last", fedDPA_w=0.5,
                     fedDPA_num_ref=5, seed=42, has_mixer=False) -> dict[str, Any]:
    correct = {m: 0 for m in METHODS}
    by_subject = {m: defaultdict(lambda: [0, 0]) for m in METHODS}
    e_values, w_local_values, alpha_values = [], [], []
    do_feddpa = ref_embs is not None and ref_embs.shape[0] > 0
    rng = random.Random(seed)
    n = 0
    for sample in samples:
        gold = sample["answer_letter"]; subj = sample.get("subject_name", "")
        options = sample["options"]
        prompt = PROMPT_NO_INPUT.format(instruction=sample["question"].strip())

        z_base = cloze_scores(model, tokenizer, prompt, options, device, global_weight=0.0, local_weight=0.0)
        z_g = cloze_scores(model, tokenizer, prompt, options, device, global_weight=1.0, local_weight=0.0)
        z_l = cloze_scores(model, tokenizer, prompt, options, device, global_weight=0.0, local_weight=1.0)

        preds = {}
        preds["base"] = argmax_letter(z_base)
        preds["local_only"] = argmax_letter(z_l)
        preds["fedit"] = argmax_letter(z_g)
        preds["equal"] = argmax_letter(0.5 * z_g + 0.5 * z_l)

        if do_feddpa:
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
            gw, lw = (1.0, 0.0) if fedDPA_emb == "global" else (0.0, 1.0) if fedDPA_emb == "local" else (0.0, 0.0)
            _, test_emb = adapter_forward(model, enc["input_ids"], enc["attention_mask"],
                                          global_weight=gw, local_weight=lw, emb_type=fedDPA_emb_type)
            k = min(fedDPA_num_ref, ref_embs.shape[0])
            idx = rng.sample(range(ref_embs.shape[0]), k)
            w_local = float(F.cosine_similarity(test_emb.unsqueeze(0), ref_embs[idx], dim=1).mean()) * fedDPA_w
            w_local_values.append(w_local)
            preds["FedDPA"] = argmax_letter((1.0 - w_local) * z_g + w_local * z_l)
        else:
            preds["FedDPA"] = preds["fedit"]

        if has_mixer:
            z_mix, alpha_mean = cloze_scores_mixer(model, tokenizer, prompt, options, device)
            preds["fedalt"] = argmax_letter(z_mix)
            if alpha_mean is not None:
                alpha_values.append(alpha_mean)
        else:
            preds["fedalt"] = preds["equal"]

        # entropy_min: 4-way 점수 분포 entropy 최소화 (z는 이미 [4] → domain=fullvocab = 4-way softmax)
        e_star = optimize_e(z_g, z_l, domain="fullvocab", letter_id_tensor=None,
                            steps=entropy_steps, lr=entropy_lr, temperature=entropy_temperature)
        preds["entropy_min"] = argmax_letter(e_star * z_g + (1.0 - e_star) * z_l)
        e_values.append(e_star)

        for m in METHODS:
            ok = int(preds[m] == gold)
            correct[m] += ok
            by_subject[m][subj][0] += ok
            by_subject[m][subj][1] += 1
        n += 1
        if n % 50 == 0:
            logger.info("  evaluated %d/%d", n, len(samples))

    acc = {m: (correct[m] / n if n else 0.0) for m in METHODS}
    subj_acc = {m: {s: (c / t if t else 0.0) for s, (c, t) in sorted(by_subject[m].items())} for m in METHODS}
    def stats(v): return {"mean": (sum(v) / len(v) if v else None), "min": (min(v) if v else None), "max": (max(v) if v else None)}
    return {"n": n, "accuracy": acc, "accuracy_by_subject": subj_acc,
            "e_star": {"entropy_min": stats(e_values)}, "feddpa_w_local": stats(w_local_values),
            "mixer_alpha_mean": stats(alpha_values)}


def discover_clients(checkpoint_dir: Path) -> list[str]:
    found = [f"client_{i}" for i in range(1, len(MEDMCQA_SUBJECTS) + 1)
             if (checkpoint_dir / f"dual_lora_adapter_client_{i}.pth").exists()]
    if not found:
        raise FileNotFoundError(f"No dual_lora_adapter_client_*.pth in {checkpoint_dir}")
    return found


def main() -> None:
    p = argparse.ArgumentParser(description="MedMCQA cloze-likelihood inference.")
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--checkpoint_dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--clients", nargs="*", default=None)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--cross_num_samples", type=int, default=None)
    p.add_argument("--skip_cross", action="store_true")
    p.add_argument("--entropy_steps", type=int, default=20)
    p.add_argument("--entropy_lr", type=float, default=0.1)
    p.add_argument("--entropy_temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--feddpa_emb", choices=["global", "local", "base"], default="global")
    p.add_argument("--feddpa_emb_type", choices=["last", "avg"], default="last")
    p.add_argument("--feddpa_w", type=float, default=0.5)
    p.add_argument("--feddpa_num_ref", type=int, default=5)
    p.add_argument("--feddpa_ref_pool", type=int, default=300)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_infer_cloze"
    output_dir = args.output_dir or (DEFAULT_OUTPUT_PARENT / run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=None)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # cloze 배치 스코어링은 right-pad

    clients = args.clients or discover_clients(args.checkpoint_dir)
    cross_samples = None if args.skip_cross else get_full_test_eval_dataset(dataset_name="medmcqa", num_samples=args.cross_num_samples)

    rows: list[dict[str, Any]] = []
    for client_id in clients:
        suffix = client_id.split('_')[-1]
        ckpt = args.checkpoint_dir / f"dual_lora_adapter_client_{suffix}.pth"
        mixer_ckpt = args.checkpoint_dir / f"mixer_client_{suffix}.pth"
        has_mixer = mixer_ckpt.exists()
        logger.info("=" * 70); logger.info("[cloze] %s | %s | mixer=%s", client_id, ckpt, has_mixer)
        model = load_adapter_model(hf_token=None, checkpoint_path=ckpt, device=device,
                                   model_id=args.model_id, mixer_path=mixer_ckpt if has_mixer else None)

        # FedDPA ref 임베딩 (own-train)
        gw, lw = (1.0, 0.0) if args.feddpa_emb == "global" else (0.0, 1.0) if args.feddpa_emb == "local" else (0.0, 0.0)
        # FedDPA cosine ref = own-train 프롬프트 임베딩
        from data.loader import _load_medmcqa_examples
        tr = _load_medmcqa_examples(client_id, split="train", limit=args.feddpa_ref_pool)
        train_prompts = [PROMPT_NO_INPUT.format(instruction=(e.get("instruction") or "").strip()) for e in tr]
        ref_embs = embed_prompts(model, tokenizer, train_prompts, device, global_weight=gw, local_weight=lw, emb_type=args.feddpa_emb_type)

        fd = dict(ref_embs=ref_embs, fedDPA_emb=args.feddpa_emb, fedDPA_emb_type=args.feddpa_emb_type,
                  fedDPA_w=args.feddpa_w, fedDPA_num_ref=args.feddpa_num_ref, seed=args.seed,
                  has_mixer=has_mixer)
        ekw = dict(entropy_steps=args.entropy_steps, entropy_lr=args.entropy_lr, entropy_temperature=args.entropy_temperature)

        own = get_client_eval_dataset(client_id, dataset_name="medmcqa", num_samples=args.num_samples)
        logger.info("[cloze] %s own (n=%d)", client_id, len(own))
        rows.append({"client": client_id, "scope": "own", **evaluate_samples(model, tokenizer, own, device, **ekw, **fd)})
        if cross_samples is not None:
            logger.info("[cloze] %s cross (n=%d)", client_id, len(cross_samples))
            rows.append({"client": client_id, "scope": "cross", **evaluate_samples(model, tokenizer, cross_samples, device, **ekw, **fd)})

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {"model_id": args.model_id, "checkpoint_dir": str(args.checkpoint_dir), "scoring": "cloze",
               "methods": METHODS, "clients": clients,
               "settings": {"entropy_steps": args.entropy_steps, "entropy_lr": args.entropy_lr,
                            "entropy_temperature": args.entropy_temperature, "num_samples": args.num_samples, "seed": args.seed},
               "rows": rows}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[cloze] saved -> %s", output_dir / "summary.json")

    print("\n" + "=" * 92)
    print(f"{'client':10s} {'scope':6s} | " + " ".join(f"{m[:12]:>12s}" for m in METHODS))
    print("-" * 92)
    for r in rows:
        print(f"{r['client']:10s} {SCOPE_LABEL.get(r['scope'], r['scope']):32s} | " + " ".join(f"{r['accuracy'][m]*100:>11.1f}%" for m in METHODS))
    print("=" * 92)
    for r in rows:
        em = r["e_star"]["entropy_min"]["mean"]; fw = r["feddpa_w_local"]["mean"]
        am = r.get("mixer_alpha_mean", {}).get("mean")
        line = f"  {r['client']:10s} {SCOPE_LABEL.get(r['scope'], r['scope']):32s} | entropy e*={em:.3f}"
        if fw is not None:
            line += f"  feddpa w_local={fw:.3f}"
        if am is not None:
            line += f"  mixer alpha_indiv={am:.3f}"
        print(line)


if __name__ == "__main__":
    main()
