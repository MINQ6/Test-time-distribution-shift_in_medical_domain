# MedMCQA × FedDPA Dual-LoRA × **Entropy-Min TTA (inference)** — 최종 스펙

기준일: 2026-06-08 · **이 문서 = 최종 확정(single source of truth).**
**핵심 = inference-time Entropy-Min TTA.** 학습(FedDPA)은 수단일 뿐. ~~FedALT 방법론은 폐기~~([Claude (1).md](Claude%20(1).md) 참고용 보관).

---

## 1. 목적 — 핵심은 "추론 결합법"
FedDPA로 client마다 **Global LoRA + Local LoRA**를 확보한 뒤, **inference 시점에 두 adapter를 결합하는 가중치 `e`를
label 없이 정하는 방법**이 제안의 핵심.
- **제안 = Entropy-Min TTA**: test sample마다 4지선다 출력 분포의 entropy가 최소가 되도록 `e`를 sample별 최적화.
- **검증 질문**: FedDPA 원본의 **cosine routing**(입력 유사도)을 **entropy-min**(출력 확신도)으로 바꾸면 더 나은가?
  + equal(0.5)·global-only(FedIT)·local-only 대비도.

## 2. 데이터셋
- MedMCQA(`openlifescienceai/medmcqa`, train split), `choice_type=="single"`.
- **4 subject = 4 client**: client_1=Medicine, _2=Skin, _3=Surgery, _4=Orthopaedics.
- client당 **train 600 / test 200**, TTP용 **all_test 800**(= 4개 own-test 200 이어붙임).
- **정답 letter 균등화(A/B/C/D 각 25%)**: 보기 순서 round-robin 재배치(본문 불변). [prepare_medmcqa.py](src/data/prepare_medmcqa.py), seed=42.
- 프롬프트: instruction = 질문 + 보기 4개, `### Response:` 직후 **정답 보기 본문**이 학습 target.

## 3. 학습 (수단) — FedDPA-F / FedDPA-T
client마다 **Global(FedAvg) LoRA + Local LoRA(dual)** 확보. F=local 맨끝 1회 personalization / T=매 round local 누적.
→ 이 두 adapter가 §4 추론의 입력. (학습 자체는 entropy-min과 무관.)

## 4. 추론 방법론 (★ 핵심) — [inference_medmcqa_cloze.py](src/inference_medmcqa_cloze.py)
**채점 = cloze**: 보기 4개 본문 likelihood `z`[4] → argmax. 지표 = **Accuracy**.
한 번의 forward로 `z_g`=Global-only(g=1,l=0), `z_l`=Local-only(g=0,l=1) 확보 후 결합법만 다르게:

| 결합법 | 가중치 | 비고 |
|---|---|---|
| **Global-only (FedIT)** | g=1 | 단순 FedAvg |
| **Local-only** | l=1 | 개인 adapter만 |
| **equal** | 0.5/0.5 고정 | naive |
| **FedDPA (cosine)** | 입력 cosine 유사도로 w 결정 | 원본 baseline (test↔own-train 임베딩, ×λ=0.5) |
| **Entropy-Min (ours)** | **4-way entropy 최소 e\*** | **제안.** [optimize_e](src/inference_medmcqa.py): e=0.5→GD(steps20·lr0.1·T1.0), 모델 frozen, label X |

예측 = `argmax(e·z_g + (1−e)·z_l)`. **per-sample, 학습·통신·저장 0.**

## 5. Experiment 세팅
- **2 scope (per-client 실측, 평균 단순계산 금지)**:
  ① **Personalization (own, n=200)**: 각 client 자기 도메인 200.
  ② **TTP (cross, n=800)**: 각 client가 800 전부.
- **결과표 형식**: 행=method, 열=**C1·Medicine / C2·Skin / C3·Surgery / C4·Ortho / Avg**, 두 scope 각각.
  값은 `summary.json`의 own/cross scope `accuracy` **직접** 읽기(by_subject 평균 재계산 X).

## 6. Hyperparameters
| 항목 | 값 |
|---|---|
| backbone | `/scratch2/0630rb/models/Qwen3-4B-Base` (bf16) |
| LoRA rank / target / alpha / dropout | 8 / q_proj,v_proj / 16 / 0.05 |
| num_rounds / local_epochs / pers_epochs | 10 / 3 / 3 |
| local-finetuning 총 epoch | 30 |
| batch / micro / ga / lr | 32 / 8 / 4 / 3e-4 |
| max_length / num_samples / seed | 512 / 600·200 / 42 |
| **entropy** steps/lr/temp/init_e | **20 / 0.1 / 1.0 / 0.5** |
| **cosine** λ / num_ref / emb | 0.5 / 5 / global·last |

## 7. 실행
```bash
# 학습(수단, a6000): METHOD ∈ {feddpa_f, feddpa_t, local_finetuned}
RUN_TIMESTAMP=<tag> METHOD=feddpa_f MODEL_ID=/scratch2/0630rb/models/Qwen3-4B-Base \
  NUM_ROUNDS=10 LOCAL_EPOCHS=3 PERSONALIZATION_EPOCHS=3 NUM_SAMPLES=600 \
  sbatch -p suma_a6000,gigabyte_a6000 train_medmcqa.sbatch
# 추론(★핵심, cloze): global_only/local_only/equal/FedDPA(cosine)/entropy_min 동시 산출
CHECKPOINT_DIR=src/checkpoints/<tag> OUTPUT_DIR=outputs/<tag>_cloze \
  sbatch -p suma_a6000,gigabyte_a6000 infer_cloze.sbatch
```

