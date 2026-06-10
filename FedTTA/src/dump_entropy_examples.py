"""한 client 모델에 cross(800) query를 넣고, entropy-min이 sample마다 정한
e*(=global 가중치)와 (1-e*)(=local 가중치)를 질문·도메인과 함께 덤프.

→ 타 도메인(foreign) query에서 가중치가 극명히 갈리는 예시 찾기용.
사용: python src/dump_entropy_examples.py --checkpoint_dir src/checkpoints/exp7_f_bal_r8qv --client client_1
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

PROJECT_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_SRC))
from data.loader import get_full_test_eval_dataset  # noqa
from utils.inference.common import PROMPT_NO_INPUT, set_seed  # noqa
from utils.inference.model_loader import load_adapter_model  # noqa
from inference_medmcqa import optimize_e  # noqa
from inference_medmcqa_cloze import cloze_scores, LETTERS  # noqa

OWN = {"client_1": "Medicine", "client_2": "Skin", "client_3": "Surgery", "client_4": "Orthopaedics"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="/scratch2/0630rb/models/Qwen3-4B-Base")
    p.add_argument("--checkpoint_dir", type=Path, required=True)
    p.add_argument("--client", default="client_1")
    p.add_argument("--num_samples", type=int, default=800)
    p.add_argument("--entropy_steps", type=int, default=20)
    p.add_argument("--entropy_lr", type=float, default=0.1)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_id, token=None)
    tok.pad_token = tok.eos_token; tok.padding_side = "right"

    suffix = args.client.split("_")[-1]
    ckpt = args.checkpoint_dir / f"dual_lora_adapter_client_{suffix}.pth"
    model = load_adapter_model(hf_token=None, checkpoint_path=ckpt, device=device, model_id=args.model_id)
    own_dom = OWN[args.client]

    samples = get_full_test_eval_dataset(dataset_name="medmcqa", num_samples=args.num_samples)
    rows = []
    for i, s in enumerate(samples):
        gold = s["answer_letter"]; subj = s.get("subject_name", "")
        options = s["options"]; prompt = PROMPT_NO_INPUT.format(instruction=s["question"].strip())
        z_g = cloze_scores(model, tok, prompt, options, device, global_weight=1.0, local_weight=0.0)
        z_l = cloze_scores(model, tok, prompt, options, device, global_weight=0.0, local_weight=1.0)
        e = optimize_e(z_g, z_l, domain="fullvocab", letter_id_tensor=None,
                       steps=args.entropy_steps, lr=args.entropy_lr, temperature=1.0)
        pred = LETTERS[int((e * z_g + (1 - e) * z_l).argmax())]              # 정상: e=global
        pred_flip = LETTERS[int(((1 - e) * z_g + e * z_l).argmax())]          # 뒤집기: e=local
        pred_g = LETTERS[int(z_g.argmax())]; pred_l = LETTERS[int(z_l.argmax())]
        rows.append({
            "i": i, "subject": subj, "is_own": subj == own_dom,
            "e_global": round(float(e), 4), "w_local": round(1 - float(e), 4),
            "pred": pred, "gold": gold, "correct": pred == gold,
            "correct_flip": pred_flip == gold,
            "correct_global": pred_g == gold, "correct_local": pred_l == gold,
            "q": s["question"].strip()[:90],
            "gold_opt": options[gold][:40],
        })
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(samples)}", flush=True)
    out = args.out or f"outputs/entropy_examples_{args.client}.jsonl"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"saved {len(rows)} -> {out} | client={args.client} own_domain={own_dom}")


if __name__ == "__main__":
    main()
