"""MedMCQA 4-subject federated 데이터 준비 (instruction tuning 용).

CLAUDE.md §3 방법론에 따라 MedMCQA(train split)에서 4개 subject를 필터링하고
subject당 Train 300 / Test 200을 샘플링하여 instruction-tuning 형식 json으로 저장한다.

- backbone/모델과 무관한 순수 전처리 (토크나이저 불필요)
- 출력: src/data/medmcqa/{Subject}_{train,test}.json  +  all_test.json (cross-domain TTP용)
- client 매핑: client_1=Medicine, client_2=Skin, client_3=Surgery, client_4=Orthopaedics

사용:
    python src/data/prepare_medmcqa.py
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

SEED = 42
N_TRAIN = 300
N_TEST = 200
# test는 shuffle 후 "마지막 N_TEST개"로 고정 → train 크기를 바꿔도(300↔600) test set 동일 = 공정 비교.
# train은 "앞 N_TRAIN개". n_train + n_test <= subject별 보유량이면 겹치지 않음.

# client_idx 순서(= client_1..4)와 일치. CLAUDE.md §3.1
SUBJECTS = ["Medicine", "Skin", "Surgery", "Orthopaedics"]
LETTERS = ["A", "B", "C", "D"]

OUT_DIR = Path(__file__).resolve().parent / "medmcqa"

PROMPT_TEMPLATE = (
    "{question}\n"
    "(A) {opa}\n"
    "(B) {opb}\n"
    "(C) {opc}\n"
    "(D) {opd}"
)


def build_record(example: dict, output_format: str = "text",
                 target_letter: str | None = None, rng: random.Random | None = None) -> dict:
    """MedMCQA raw example -> instruction-tuning + MCQA 평가용 record.

    output_format: 'text'=정답 보기 본문(cloze용), 'letter'=A/B/C/D 한 글자.
    target_letter: 지정 시 정답 보기를 그 슬롯(A/B/C/D)으로 재배치(나머지 보기는 셔플)
        → 정답 letter 분포 균등화. None이면 원본 순서 유지.
    """
    cop = int(example["cop"])  # 0..3
    bodies = [str(example[k]).strip() for k in ("opa", "opb", "opc", "opd")]
    correct_body = bodies[cop]

    if target_letter is None:
        new_bodies = bodies
        ans_idx = cop
    else:
        others = [b for i, b in enumerate(bodies) if i != cop]
        if rng is not None:
            rng.shuffle(others)
        ans_idx = LETTERS.index(target_letter)
        new_bodies, oi = [], 0
        for slot in range(4):
            if slot == ans_idx:
                new_bodies.append(correct_body)
            else:
                new_bodies.append(others[oi]); oi += 1

    answer_letter = LETTERS[ans_idx]
    opts = {LETTERS[i]: new_bodies[i] for i in range(4)}
    instruction = PROMPT_TEMPLATE.format(
        question=example["question"].strip(),
        opa=opts["A"], opb=opts["B"], opc=opts["C"], opd=opts["D"],
    )
    output = opts[answer_letter] if output_format == "text" else answer_letter
    return {
        # --- instruction tuning 필드 (loader 호환: instruction/input/output) ---
        "instruction": instruction,
        "input": "",
        "output": output,  # text=정답 보기 본문(cloze) / letter=A~D
        # --- MCQA 평가용 메타 ---
        "answer_letter": answer_letter,
        "cop": ans_idx,
        "options": opts,
        "subject_name": example["subject_name"],
        "id": example["id"],
        "exp": example.get("exp"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MedMCQA 4-subject FL data.")
    parser.add_argument("--n_train", type=int, default=N_TRAIN)
    parser.add_argument("--n_test", type=int, default=N_TEST)
    parser.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--output_format", choices=["text", "letter"], default="text",
                        help="text=정답 보기 본문(cloze) / letter=A~D")
    parser.add_argument("--no_balance", action="store_true",
                        help="정답 letter 균등화 비활성(원본 순서 유지). 기본=균등화 ON")
    args = parser.parse_args()
    n_train, n_test = args.n_train, args.n_test
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # train split만 사용: cop label이 유효함 (공식 test split은 label 없음)
    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    # single-answer 문항만 사용 (4지선다 단일정답 가정, multi 복합추론형 제외). CLAUDE.md §10-1
    ds = ds.filter(lambda x: x["choice_type"] == "single")

    summary = {}
    all_test_records = []
    for subject in SUBJECTS:
        subset = ds.filter(lambda x, s=subject: x["subject_name"] == s)
        n = len(subset)
        if n < n_train + n_test:
            raise ValueError(
                f"{subject}: {n} samples < {n_train + n_test} 필요"
            )
        indices = list(range(n))
        rng.shuffle(indices)
        # test = 마지막 n_test개(train 크기와 무관하게 고정), train = 앞 n_train개
        train_idx = indices[:n_train]
        test_idx = indices[-n_test:]

        # round-robin으로 정답 letter 균등 배치 (shuffle된 index 순서 위에 적용 → 무작위+균등)
        bal = not args.no_balance
        train_records = [build_record(subset[idx], args.output_format,
                                      target_letter=(LETTERS[j % 4] if bal else None), rng=rng)
                         for j, idx in enumerate(train_idx)]
        test_records = [build_record(subset[idx], args.output_format,
                                     target_letter=(LETTERS[j % 4] if bal else None), rng=rng)
                        for j, idx in enumerate(test_idx)]

        (out_dir / f"{subject}_train.json").write_text(
            json.dumps(train_records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / f"{subject}_test.json").write_text(
            json.dumps(test_records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        all_test_records.extend(test_records)
        summary[subject] = {"available": n, "train": len(train_records), "test": len(test_records)}
        print(f"[{subject:14s}] available={n:6d}  train={len(train_records)}  test={len(test_records)}")

    # cross-domain TTP 평가용 통합 test (4 subject 섞임)
    (out_dir / "all_test.json").write_text(
        json.dumps(all_test_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "prepare_summary.json").write_text(
        json.dumps(
            {"seed": SEED, "n_train": n_train, "n_test": n_test,
             "client_map": {f"client_{i+1}": s for i, s in enumerate(SUBJECTS)},
             "subjects": summary, "all_test": len(all_test_records)},
            ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[all_test] {len(all_test_records)} records -> {out_dir/'all_test.json'}")
    print(f"output dir: {out_dir}")


if __name__ == "__main__":
    main()
