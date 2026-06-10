# Test-time Distribution Shift in the Medical Domain

Federated **dual-LoRA** (Global + Local) fine-tuning on **MedMCQA**, with an
**inference-time Entropy-Minimization Test-Time Adaptation (TTA)** that combines the two
LoRA branches per test sample — *without labels, training, or extra communication*.

> 자세한 방법론·설정은 [CLAUDE.md](CLAUDE.md) 참고 (single source of truth).

## Idea
Each client holds `frozen base + Global LoRA + Local LoRA`. At inference, for each
4-choice question we score the options under Global-only (`z_g`) and Local-only (`z_l`),
then pick a scalar mixing weight `e` that **minimizes the entropy** of the 4-way answer
distribution `softmax(e·z_g + (1-e)·z_l)`. Prediction = `argmax(e*·z_g + (1-e*)·z_l)`.

Compared baselines (same checkpoint, different combination):
`Global-only (FedIT)` / `Local-only` / `equal(0.5)` / `FedDPA (cosine routing)` / **`Entropy-Min (ours)`**.

## Setup
```bash
pip install -r requirements.txt
# data: MedMCQA 4-subject (Medicine/Skin/Surgery/Orthopaedics), balanced answer letters
python src/data/prepare_medmcqa.py --n_train 600 --n_test 200
```

## Train (federated dual-LoRA; SLURM)
```bash
RUN_TIMESTAMP=<tag> METHOD=feddpa_f MODEL_ID=<Qwen3-4B-Base> \
  NUM_ROUNDS=10 LOCAL_EPOCHS=3 PERSONALIZATION_EPOCHS=3 NUM_SAMPLES=600 \
  sbatch -p <a6000_partition> train_medmcqa.sbatch
# METHOD in {feddpa_f, feddpa_t, local_finetuned, centralized}
```

## Inference (Entropy-Min TTA + baselines; cloze accuracy)
```bash
CHECKPOINT_DIR=src/checkpoints/<tag> OUTPUT_DIR=outputs/<tag>_cloze \
  sbatch -p <a6000_partition> infer_cloze.sbatch
```
Outputs `summary.json` with per-client / per-scope accuracy for all combination methods.

## Key code
- `src/inference_medmcqa.py` — `optimize_e()` (entropy-min weight search)
- `src/inference_medmcqa_cloze.py` — cloze scoring + all combination methods
- `src/utils/training/` — `feddpa_f.py`, `feddpa_t.py`, `local_finetuned.py`, `centralized.py`
- `src/models/dual_lora_adapter.py` — Global/Local dual-LoRA injection

## Eval
2 scopes — **Personalization** (own-domain test, n=200) and **TTP** (cross, n=800);
metric = **accuracy** (4-choice cloze argmax).
