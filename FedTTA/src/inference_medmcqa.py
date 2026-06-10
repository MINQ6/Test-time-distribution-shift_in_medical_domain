"""MedMCQA × FedDPA-F × Entropy-Min TTA 추론/평가 (letter-likelihood).

핵심: test sample 1개당 forward 2회만 수행
  z_g = global-only last-token logits, z_l = local-only last-token logits
이 둘로 아래 모든 method를 추가 forward 없이 동시에 평가한다 (공정 비교).

method (정답 결정은 전부 A/B/C/D 4-letter argmax):
  - base                : adapter 0 (frozen base LLM만; LoRA 무) — 하한 baseline
  - local_only          : e=0  (local LoRA만)
  - fedit               : e=1  (global=FedAvg LoRA만; 곧 FedIT)
  - equal               : e=0.5
  - FedDPA              : 원조 FedDPA cosine routing. test↔own-train 임베딩 cosine 유사도로
                          local 가중치 동적 결정. 임베딩은 global adapter로 추출(논문 §). e=1-w_local.
  - entropy_min         : e* = 4-letter restricted entropy 최소화 (A/B/C/D 4-way softmax)

평가 scope:
  - own   : 각 client가 자기 subject test
  - cross : 각 client 모델이 4 subject 통합 test(800) — subject별 분해 포함

사용:
  python src/inference_medmcqa.py                       # 모든 client, own+cross
  python src/inference_medmcqa.py --clients client_1    # 일부만
  python src/inference_medmcqa.py --skip_cross          # own만
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

from data.loader import (  # noqa: E402
    MEDMCQA_SUBJECTS,
    _load_medmcqa_examples,
    get_client_eval_dataset,
    get_full_test_eval_dataset,
)
from utils.inference.common import PROMPT_NO_INPUT, logger, set_seed  # noqa: E402
from utils.inference.model_loader import load_adapter_model  # noqa: E402

LETTERS = ["A", "B", "C", "D"]
DEFAULT_MODEL_ID = "/scratch2/0630rb/models/Qwen3-1.7B-Base"
DEFAULT_CHECKPOINT_DIR = PROJECT_SRC / "checkpoints"
DEFAULT_OUTPUT_PARENT = PROJECT_ROOT / "outputs"

METHODS = ["base", "local_only", "fedit", "equal", "FedDPA", "entropy_min"]
# scope 표시 라벨 (내부 키는 own/cross 유지): own=Personalization, cross=Test-time generalization
SCOPE_LABEL = {"own": "Personalization", "cross": "TTG (Test-time generalization)"}


def get_letter_ids(tokenizer) -> dict[str, int]:
    ids: dict[str, int] = {}
    for letter in LETTERS:
        enc = tokenizer.encode(letter, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"Letter {letter!r} is not a single token: {enc}")
        ids[letter] = enc[0]
    return ids


@torch.no_grad()
def adapter_forward(model, input_ids, attention_mask, *, global_weight, local_weight, emb_type=None):
    """마지막 토큰 logits 반환. emb_type 지정 시 마지막 hidden state 임베딩도 함께 반환.
    emb_type: None | 'last'(마지막 토큰 hidden) | 'avg'(mask 평균 hidden).

    임베딩은 output_hidden_states=True 대신 lm_head pre-hook으로 final hidden을 가로채 얻는다.
    (device_map='auto'/accelerate 환경에서 output_hidden_states 경로가 meta-device 에러를 내서 회피)
    """
    model.dual_lora_adapter.set_adapter_weights(global_weight=global_weight, local_weight=local_weight)
    captured: dict[str, torch.Tensor] = {}
    handle = None
    if emb_type is not None:
        def _pre_hook(_module, args):
            captured["h"] = args[0]  # lm_head 입력 = final hidden state [B, seq, H]
        handle = model.get_output_embeddings().register_forward_pre_hook(_pre_hook)
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    finally:
        if handle is not None:
            handle.remove()
    last_logits = out.logits[:, -1, :].float()[0]  # [vocab]
    emb = None
    if emb_type == "last":
        emb = captured["h"][:, -1, :].float()[0]                    # [H]
    elif emb_type == "avg":
        h = captured["h"].float()[0]                                # [seq, H]
        m = attention_mask[0].unsqueeze(-1).float()                 # [seq, 1]
        emb = (h * m).sum(0) / m.sum().clamp(min=1.0)
    return last_logits, emb


def optimize_e(z_g, z_l, *, domain, letter_id_tensor, steps, lr, init_e=0.5, temperature=1.0) -> float:
    """entropy 최소화로 e* 탐색 (best-e tracking). domain: '4letter' | 'fullvocab'.

    temperature>1 이면 entropy 계산 직전 logits를 T로 나눠 분포를 평탄화 → 과확신으로
    납작해진 ∂entropy/∂e 신호를 회복시키는 용도(재학습 없이 추론에서만). T는 entropy
    landscape(=e* 선택)만 바꾸고, 최종 정답 argmax 자체는 호출부에서 원본 logits로 한다."""
    e = torch.tensor(float(init_e), device=z_g.device, dtype=torch.float32, requires_grad=True)
    best_e, best_ent = float(init_e), float("inf")
    for _ in range(steps):
        z_mix = e * z_g + (1.0 - e) * z_l
        logits = z_mix[letter_id_tensor] if domain == "4letter" else z_mix
        logits = logits / temperature
        log_p = F.log_softmax(logits, dim=-1)
        ent = -(log_p.exp() * log_p).sum()
        ent_val = float(ent.detach())
        if ent_val < best_ent:
            best_ent, best_e = ent_val, float(e.detach())
        ent.backward()
        with torch.no_grad():
            if e.grad is not None:
                e -= lr * e.grad
            e.clamp_(0.0, 1.0)
            e.grad = None
    return best_e


def letter_argmax(z_mix, letter_id_tensor) -> str:
    sub = z_mix[letter_id_tensor]  # [4]
    return LETTERS[int(sub.argmax())]


@torch.no_grad()
def embed_prompts(model, tokenizer, prompts, device, *, global_weight, local_weight, emb_type):
    """프롬프트 리스트를 (base+지정 adapter) hidden state로 임베딩. [N, H] 반환."""
    embs = []
    for p in prompts:
        enc = tokenizer(p, return_tensors="pt", truncation=True, max_length=1024).to(device)
        _, emb = adapter_forward(model, enc["input_ids"], enc["attention_mask"],
                                 global_weight=global_weight, local_weight=local_weight, emb_type=emb_type)
        embs.append(emb)
    return torch.stack(embs)  # [N, H]


def evaluate_samples(
    model, tokenizer, samples, device, letter_ids, *, entropy_steps, entropy_lr,
    entropy_temperature=1.0,
    ref_embs=None, fedDPA_emb="global", fedDPA_emb_type="last", fedDPA_w=0.5,
    fedDPA_num_ref=5, seed=42,
) -> dict[str, Any]:
    """samples 전체를 method별로 평가. forward 2회/sample. subject별 분해도 집계.

    FedDPA(cosine routing): ref_embs(= client own-train 임베딩 [N,H])가 주어지면, test마다
    fedDPA_num_ref개 train을 랜덤 추출해 평균 cosine 유사도 → w_local = mean_cos * fedDPA_w,
    z_mix = (1-w_local)*z_g + w_local*z_l. 임베딩 adapter는 fedDPA_emb로 결정(기본 global).
    """
    letter_id_tensor = torch.tensor([letter_ids[l] for l in LETTERS], device=device)
    correct = {m: 0 for m in METHODS}
    by_subject = {m: defaultdict(lambda: [0, 0]) for m in METHODS}  # m -> subj -> [correct, total]
    e_values = {"entropy_min": []}
    w_local_values: list[float] = []
    do_feddpa = ref_embs is not None and ref_embs.shape[0] > 0
    emb_g, emb_l = (1.0, 0.0) if fedDPA_emb == "global" else (0.0, 1.0) if fedDPA_emb == "local" else (0.0, 0.0)
    rng = random.Random(seed)
    n = 0

    for sample in samples:
        gold = sample["answer_letter"]
        subj = sample.get("subject_name", "")
        prompt = PROMPT_NO_INPUT.format(instruction=sample["question"].strip())
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        ids, mask = enc["input_ids"], enc["attention_mask"]

        # z_g forward에서 (필요시) global 임베딩도 같이 얻어 test당 추가 forward 0
        z_g, emb_from_g = adapter_forward(model, ids, mask, global_weight=1.0, local_weight=0.0,
                                          emb_type=(fedDPA_emb_type if do_feddpa and fedDPA_emb == "global" else None))
        z_l, emb_from_l = adapter_forward(model, ids, mask, global_weight=0.0, local_weight=1.0,
                                          emb_type=(fedDPA_emb_type if do_feddpa and fedDPA_emb == "local" else None))
        # base(LoRA 무): frozen LLM 하한 baseline. emb=base면 임베딩도 같이.
        z_base, emb_from_base = adapter_forward(model, ids, mask, global_weight=0.0, local_weight=0.0,
                                                emb_type=(fedDPA_emb_type if do_feddpa and fedDPA_emb == "base" else None))

        preds: dict[str, str] = {}
        preds["base"] = letter_argmax(z_base, letter_id_tensor)
        preds["local_only"] = letter_argmax(z_l, letter_id_tensor)
        preds["fedit"] = letter_argmax(z_g, letter_id_tensor)
        preds["equal"] = letter_argmax(0.5 * z_g + 0.5 * z_l, letter_id_tensor)

        if do_feddpa:
            if fedDPA_emb == "global":
                test_emb = emb_from_g
            elif fedDPA_emb == "local":
                test_emb = emb_from_l
            else:  # base 임베딩
                test_emb = emb_from_base
            k = min(fedDPA_num_ref, ref_embs.shape[0])
            idx = rng.sample(range(ref_embs.shape[0]), k)
            sims = F.cosine_similarity(test_emb.unsqueeze(0), ref_embs[idx], dim=1)  # [k]
            w_local = float(sims.mean()) * fedDPA_w
            w_local_values.append(w_local)
            preds["FedDPA"] = letter_argmax((1.0 - w_local) * z_g + w_local * z_l, letter_id_tensor)
        else:
            preds["FedDPA"] = preds["fedit"]  # ref 없으면 global로 fallback (표기만)

        e_star = optimize_e(z_g, z_l, domain="4letter", letter_id_tensor=letter_id_tensor,
                            steps=entropy_steps, lr=entropy_lr, temperature=entropy_temperature)
        preds["entropy_min"] = letter_argmax(e_star * z_g + (1.0 - e_star) * z_l, letter_id_tensor)
        e_values["entropy_min"].append(e_star)

        for m in METHODS:
            ok = int(preds[m] == gold)
            correct[m] += ok
            by_subject[m][subj][0] += ok
            by_subject[m][subj][1] += 1
        n += 1
        if n % 50 == 0:
            logger.info("  evaluated %d/%d", n, len(samples))

    acc = {m: (correct[m] / n if n else 0.0) for m in METHODS}
    subj_acc = {
        m: {s: (c / t if t else 0.0) for s, (c, t) in sorted(by_subject[m].items())}
        for m in METHODS
    }
    e_summary = {
        name: {
            "mean": (sum(v) / len(v) if v else None),
            "min": (min(v) if v else None),
            "max": (max(v) if v else None),
        }
        for name, v in e_values.items()
    }
    feddpa_w_summary = {
        "mean": (sum(w_local_values) / len(w_local_values) if w_local_values else None),
        "min": (min(w_local_values) if w_local_values else None),
        "max": (max(w_local_values) if w_local_values else None),
    }
    return {"n": n, "accuracy": acc, "accuracy_by_subject": subj_acc,
            "e_star": e_summary, "feddpa_w_local": feddpa_w_summary}


def discover_clients(checkpoint_dir: Path) -> list[str]:
    found = []
    for i in range(1, len(MEDMCQA_SUBJECTS) + 1):
        if (checkpoint_dir / f"dual_lora_adapter_client_{i}.pth").exists():
            found.append(f"client_{i}")
    if not found:
        raise FileNotFoundError(f"No dual_lora_adapter_client_*.pth in {checkpoint_dir}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="MedMCQA FedDPA-F entropy-min TTA inference (letter-likelihood).")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--checkpoint_dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--clients", nargs="*", default=None)
    parser.add_argument("--num_samples", type=int, default=200, help="own-domain test 샘플 수")
    parser.add_argument("--cross_num_samples", type=int, default=None, help="cross test 샘플 수(None=전체 800)")
    parser.add_argument("--skip_cross", action="store_true")
    parser.add_argument("--entropy_steps", type=int, default=20)
    parser.add_argument("--entropy_lr", type=float, default=0.1)
    parser.add_argument("--entropy_temperature", type=float, default=1.0,
                        help="entropy-min 시 logits/T 로 분포 평탄화(과확신 완화). T=1이면 기존과 동일")
    parser.add_argument("--seed", type=int, default=42)
    # --- FedDPA cosine routing (원조 baseline) ---
    parser.add_argument("--feddpa_emb", choices=["global", "local", "base"], default="global",
                        help="test↔train 유사도 임베딩 adapter (논문: global)")
    parser.add_argument("--feddpa_emb_type", choices=["last", "avg"], default="last")
    parser.add_argument("--feddpa_w", type=float, default=0.5, help="평균 cosine에 곱하는 스케일 w")
    parser.add_argument("--feddpa_num_ref", type=int, default=5, help="test당 비교할 own-train 랜덤 샘플 수")
    parser.add_argument("--feddpa_ref_pool", type=int, default=300,
                        help="임베딩 미리 계산할 own-train 풀 크기(여기서 num_ref개 랜덤 추출)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("CUDA not available; running on CPU will be very slow.")

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_infer_medmcqa"
    output_dir = args.output_dir or (DEFAULT_OUTPUT_PARENT / run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=None)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    letter_ids = get_letter_ids(tokenizer)
    logger.info("[infer] letter token ids: %s", letter_ids)

    clients = args.clients or discover_clients(args.checkpoint_dir)
    cross_samples = None if args.skip_cross else get_full_test_eval_dataset(
        dataset_name="medmcqa", num_samples=args.cross_num_samples
    )

    rows: list[dict[str, Any]] = []
    for client_id in clients:
        ckpt = args.checkpoint_dir / f"dual_lora_adapter_client_{client_id.split('_')[-1]}.pth"
        logger.info("=" * 70)
        logger.info("[infer] %s | checkpoint=%s", client_id, ckpt)
        model = load_adapter_model(hf_token=None, checkpoint_path=ckpt, device=device, model_id=args.model_id)

        # FedDPA cosine routing용: 이 client own-train 임베딩을 1회 미리 계산 (global adapter)
        emb_g, emb_l = ((1.0, 0.0) if args.feddpa_emb == "global"
                        else (0.0, 1.0) if args.feddpa_emb == "local" else (0.0, 0.0))
        train_ex = _load_medmcqa_examples(client_id, split="train", limit=args.feddpa_ref_pool)
        train_prompts = [PROMPT_NO_INPUT.format(instruction=(e.get("instruction") or "").strip())
                         for e in train_ex]
        logger.info("[infer] %s | FedDPA ref embeddings (n=%d, adapter=%s)",
                    client_id, len(train_prompts), args.feddpa_emb)
        ref_embs = embed_prompts(model, tokenizer, train_prompts, device,
                                 global_weight=emb_g, local_weight=emb_l, emb_type=args.feddpa_emb_type)

        feddpa_kw = dict(ref_embs=ref_embs, fedDPA_emb=args.feddpa_emb,
                         fedDPA_emb_type=args.feddpa_emb_type, fedDPA_w=args.feddpa_w,
                         fedDPA_num_ref=args.feddpa_num_ref, seed=args.seed)

        own_samples = get_client_eval_dataset(client_id, dataset_name="medmcqa", num_samples=args.num_samples)
        logger.info("[infer] %s | own-domain eval (n=%d)", client_id, len(own_samples))
        own = evaluate_samples(model, tokenizer, own_samples, device, letter_ids,
                               entropy_steps=args.entropy_steps, entropy_lr=args.entropy_lr,
                               entropy_temperature=args.entropy_temperature, **feddpa_kw)
        rows.append({"client": client_id, "scope": "own", **own})

        if cross_samples is not None:
            logger.info("[infer] %s | cross-domain eval (n=%d)", client_id, len(cross_samples))
            cross = evaluate_samples(model, tokenizer, cross_samples, device, letter_ids,
                                     entropy_steps=args.entropy_steps, entropy_lr=args.entropy_lr,
                               entropy_temperature=args.entropy_temperature, **feddpa_kw)
            rows.append({"client": client_id, "scope": "cross", **cross})

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "model_id": args.model_id,
        "checkpoint_dir": str(args.checkpoint_dir),
        "methods": METHODS,
        "clients": clients,
        "settings": {"entropy_steps": args.entropy_steps, "entropy_lr": args.entropy_lr,
                     "entropy_temperature": args.entropy_temperature,
                     "num_samples": args.num_samples, "seed": args.seed,
                     "feddpa_emb": args.feddpa_emb, "feddpa_emb_type": args.feddpa_emb_type,
                     "feddpa_w": args.feddpa_w, "feddpa_num_ref": args.feddpa_num_ref,
                     "feddpa_ref_pool": args.feddpa_ref_pool},
        "rows": rows,
    }
    out_json = output_dir / "summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[infer] saved summary -> %s", out_json)

    # --- 콘솔 표 ---
    print("\n" + "=" * 90)
    print(f"{'client':10s} {'scope':6s} | " + " ".join(f"{m[:14]:>14s}" for m in METHODS))
    print("-" * 90)
    for r in rows:
        accs = " ".join(f"{r['accuracy'][m]*100:>13.1f}%" for m in METHODS)
        print(f"{r['client']:10s} {SCOPE_LABEL.get(r['scope'], r['scope']):32s} | {accs}")
    print("=" * 90)
    print("entropy_min e* (mean) / FedDPA w_local (mean) per row:")
    for r in rows:
        e4 = r["e_star"]["entropy_min"]["mean"]
        wl = r.get("feddpa_w_local", {}).get("mean")
        e4s = f"{e4:.3f}" if e4 is not None else "NA"
        wls = f"{wl:.3f}" if wl is not None else "NA"
        print(f"  {r['client']:10s} {SCOPE_LABEL.get(r['scope'], r['scope']):32s} | entropy_min e*={e4s}  FedDPA w_local={wls}")
    print("=" * 90)
    print(f"saved: {out_json}")


if __name__ == "__main__":
    main()
