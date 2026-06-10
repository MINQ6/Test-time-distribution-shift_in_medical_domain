"""MedMCQA 결과를 FedDPA 논문 스타일 표로 출력.

두 섹션:
  - Personalization        : client_i 모델 → 자기 도메인 test (own)
  - Test-Time Personalization: client_i 모델 → 다른 client들의 test 전부 (own 제외 평균)

method 매핑 (inference_medmcqa.py의 summary.json에서):
  - Local-finetuned        : local_finetuned run 의 global_only (학습 가중치가 global slot에 저장됨)
  - FedAvg                 : feddpa_f run 의 global_only (Stage1 single-global FedAvg)
  - FedDPA-F local         : feddpa_f run 의 local_only (Stage2 personalized local)
  - Equal (e=0.5)          : feddpa_f run 의 equal
  - Entropy-Min TTA (4letter, ours) : feddpa_f run 의 entropy_min_4letter
  - Entropy-Min TTA (fullvocab)     : feddpa_f run 의 entropy_min_fullvocab

사용:
  python src/build_results_table.py --feddpa <feddpa_summary.json> [--local_finetuned <lf_summary.json>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CLIENT_SUBJECT = {
    "client_1": "Medicine", "client_2": "Skin",
    "client_3": "Surgery", "client_4": "Orthopaedics",
}
SUBJECT_ORDER = ["Medicine", "Skin", "Surgery", "Orthopaedics"]


def load(path):
    return json.load(open(path, encoding="utf-8")) if path and Path(path).exists() else None


def row_by(summary, client, scope):
    for r in summary["rows"]:
        if r["client"] == client and r["scope"] == scope:
            return r
    return None


def own_acc(summary, client, method):
    r = row_by(summary, client, "own")
    return r["accuracy"][method] if r else None


def ttp_acc(summary, client, method):
    """다른 client 도메인 test 평균 (own 제외) — cross의 subject별 분해에서 계산."""
    r = row_by(summary, client, "cross")
    if not r:
        return None
    own_subj = CLIENT_SUBJECT[client]
    by_subj = r["accuracy_by_subject"][method]
    others = [acc for subj, acc in by_subj.items() if subj != own_subj]
    return sum(others) / len(others) if others else None


def build_section(specs, acc_fn):
    """specs: [(row_label, summary, inner_method)]. -> rows of {label, per-subject, avg}."""
    out = []
    for label, summary, method in specs:
        if summary is None:
            continue
        vals = []
        for client in ["client_1", "client_2", "client_3", "client_4"]:
            v = acc_fn(summary, client, method)
            vals.append(v)
        valid = [v for v in vals if v is not None]
        avg = sum(valid) / len(valid) if valid else None
        out.append((label, vals, avg))
    return out


def fmt(v):
    return f"{v*100:.2f}" if v is not None else "  -  "


def print_md(title, section):
    print(f"\n### {title}\n")
    header = "| Method | " + " | ".join(SUBJECT_ORDER) + " | Average |"
    print(header)
    print("|" + "---|" * (len(SUBJECT_ORDER) + 2))
    for label, vals, avg in section:
        cells = " | ".join(fmt(v) for v in vals)
        print(f"| {label} | {cells} | **{fmt(avg)}** |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feddpa", required=True, help="feddpa_f inference summary.json")
    ap.add_argument("--local_finetuned", default=None, help="local_finetuned inference summary.json")
    ap.add_argument("--out", default=None, help="markdown 저장 경로(옵션)")
    args = ap.parse_args()

    fed = load(args.feddpa)
    lf = load(args.local_finetuned)

    specs = [
        ("Local-finetuned", lf, "global_only"),
        ("FedAvg", fed, "global_only"),
        ("FedDPA-F local", fed, "local_only"),
        ("Equal (e=0.5)", fed, "equal"),
        ("Entropy-Min TTA (4letter, ours)", fed, "entropy_min_4letter"),
        ("Entropy-Min TTA (fullvocab)", fed, "entropy_min_fullvocab"),
    ]

    pers = build_section(specs, own_acc)
    ttp = build_section(specs, ttp_acc)

    print("=" * 80)
    print("MedMCQA × Qwen3-1.7B-Base — FedDPA-F / Entropy-Min TTA")
    print("컬럼 = 각 도메인 client 모델 / 값 = accuracy(%)")
    print("=" * 80)
    print_md("Personalization (own-domain test)", pers)
    print_md("Test-Time Personalization (다른 client들의 test, own 제외 평균)", ttp)

    if args.out:
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_md("Personalization (own-domain test)", pers)
            print_md("Test-Time Personalization (other clients' tests, own excluded)", ttp)
        Path(args.out).write_text(buf.getvalue(), encoding="utf-8")
        print(f"\nsaved markdown -> {args.out}")


if __name__ == "__main__":
    main()
