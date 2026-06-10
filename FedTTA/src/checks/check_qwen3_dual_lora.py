"""① Qwen3-1.7B-Base + dual LoRA(q/v, rank=8) 주입 점검.

- backbone에 q_proj/v_proj LoRA가 정확히 부착되는지
- global/local LoRA가 trainable, base는 frozen인지
- configure_global_training / configure_local_training 토글 동작
- MedMCQA 프롬프트로 forward → last-token logits에서 A/B/C/D(id 32/33/34/35) 접근 확인

GPU 없이도 구조 점검 가능(forward는 fp32 CPU). 학습 자체는 GPU 노드에서 수행.

사용:
    python src/checks/check_qwen3_dual_lora.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/

from models.dual_lora_adapter import DualLoRAAdapter  # noqa: E402
from models.dual_lora_model import setup_model_with_dual_lora  # noqa: E402

MODEL_ID = "Qwen/Qwen3-1.7B-Base"
RANK = 8
TARGETS = ["q_proj", "v_proj"]
LETTER_IDS = {"A": 32, "B": 33, "C": 34, "D": 35}


def count_params(params) -> int:
    return sum(p.numel() for p in params)


def main() -> None:
    token = os.getenv("HF_TOKEN")  # Qwen3는 public, 토큰 없어도 됨

    print(f"[load] {MODEL_ID}  rank={RANK}  targets={TARGETS}")
    model = setup_model_with_dual_lora(
        MODEL_ID, token, rank=RANK, alpha=16, dropout=0.05, target_modules=TARGETS
    )
    adapter = model.dual_lora_adapter

    # --- 1) 주입된 모듈 점검 ---
    names = adapter.target_module_names
    suffixes = sorted({n.rsplit(".", 1)[-1] for n in names})
    n_q = sum(n.endswith("q_proj") for n in names)
    n_v = sum(n.endswith("v_proj") for n in names)
    print(f"[modules] wrapped={len(names)}  suffixes={suffixes}  q_proj={n_q}  v_proj={n_v}")
    assert suffixes == ["q_proj", "v_proj"], f"예상 외 모듈: {suffixes}"
    assert n_q == n_v, "q_proj/v_proj 개수 불일치"

    # --- 2) trainable / frozen 점검 ---
    base_trainable = count_params(p for p in model.parameters() if p.requires_grad)
    g = count_params(adapter.global_parameters())
    l = count_params(adapter.local_parameters())
    print(f"[params] global_lora={g:,}  local_lora={l:,}  total_trainable={base_trainable:,}")
    assert base_trainable == g + l, "base가 frozen이 아니거나 trainable 집계 불일치"

    # rank, in/out 차원 확인 (한 모듈)
    sample = adapter._wrapped_modules[names[0]]
    print(f"[shape] {names[0]} global A={tuple(sample.global_lora.lora_A.weight.shape)} "
          f"B={tuple(sample.global_lora.lora_B.weight.shape)}")
    assert sample.global_lora.lora_A.weight.shape[0] == RANK

    # --- 3) 학습 토글 점검 ---
    adapter.configure_global_training()
    g_on = all(p.requires_grad for p in adapter.global_parameters())
    l_off = all(not p.requires_grad for p in adapter.local_parameters())
    print(f"[toggle] configure_global_training: global_trainable={g_on}  local_frozen={l_off}")
    assert g_on and l_off
    adapter.configure_local_training()
    g_off = all(not p.requires_grad for p in adapter.global_parameters())
    l_on = all(p.requires_grad for p in adapter.local_parameters())
    print(f"[toggle] configure_local_training:  global_frozen={g_off}  local_trainable={l_on}")
    assert g_off and l_on
    adapter.configure_training()  # 원복

    # --- 4) MedMCQA 프롬프트 forward (fp32 CPU) ---
    if os.getenv("SKIP_FORWARD") == "1":
        print("\n[SKIP_FORWARD=1] forward 생략 (구조 점검만). 학습은 GPU 노드에서 수행.")
        print("\n[OK] Qwen3-1.7B-Base + dual LoRA(q/v, r=8) 주입/토글 점검 통과.")
        return
    model = model.float().eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    tok.pad_token = tok.eos_token
    prompt = (
        "### Instruction:\n"
        "A 60 yr old chronic smoker presents with painless gross hematuria. "
        "Investigation of choice to know the cause of hematuria?\n"
        "(A) USG\n(B) X-ray KUB\n(C) Urine routine\n(D) Urine microscopy for malignant cytology cells\n\n"
        "### Response:\n"
    )
    enc = tok(prompt, return_tensors="pt")

    def last_logits(global_w: float, local_w: float) -> torch.Tensor:
        adapter.set_adapter_weights(global_weight=global_w, local_weight=local_w)
        with torch.no_grad():
            out = model(**enc, return_dict=True)
        return out.logits[:, -1, :].float()[0]

    # LoRA B가 zero-init이라 학습 전엔 global/local 모두 base와 동일(=동치). 단지 forward 무결성 확인.
    z_g = last_logits(1.0, 0.0)
    z_l = last_logits(0.0, 1.0)
    print(f"[forward] logits shape per adapter: {tuple(z_g.shape)}  (vocab)")
    for tag, z in [("global", z_g), ("local", z_l)]:
        letter_logits = {ltr: round(z[i].item(), 3) for ltr, i in LETTER_IDS.items()}
        pred = max(LETTER_IDS, key=lambda k: z[LETTER_IDS[k]].item())
        top1 = tok.decode([int(z.argmax())])
        print(f"  [{tag}] A/B/C/D logits={letter_logits}  -> letter-argmax={pred}  (vocab top1={top1!r})")

    print("\n[OK] Qwen3-1.7B-Base + dual LoRA(q/v, r=8) 주입/토글/forward 점검 통과.")


if __name__ == "__main__":
    main()
