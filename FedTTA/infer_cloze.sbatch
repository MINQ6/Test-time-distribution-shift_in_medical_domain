#!/bin/bash
#SBATCH --job-name=infer_cloze
#SBATCH --partition=suma_rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
# =========================================================================
# MedMCQA cloze-likelihood 추론 (보기 본문 log-likelihood, A/B/C/D 아님).
# method: base/local_only/fedit/equal/FedDPA/entropy_min, scope: own+cross.
# 기본 체크포인트: src/checkpoints/exp3_4b_cloze (4B, cloze 학습).
# 제출: CHECKPOINT_DIR=src/checkpoints/exp3_4b_cloze sbatch infer_cloze.sbatch
# =========================================================================
set -euo pipefail
PROJECT_ROOT="/home/0630rb/yonsei/medical_ai/FedTTA"
cd "$PROJECT_ROOT"
mkdir -p logs outputs

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${SLURM_JOB_NAME:-infer_cloze}"
CLIENT="${CLIENT:-}"
CLIENT_ARG=""; CLIENT_SUFFIX=""
[ -n "$CLIENT" ] && { CLIENT_ARG="--clients $CLIENT"; CLIENT_SUFFIX="_${CLIENT}"; }
RUN_ID="${RUN_TS}_${RUN_TAG}${CLIENT_SUFFIX}${SLURM_JOB_ID:+_${SLURM_JOB_ID}}"
RUN_LOG_DIR="logs/${RUN_ID}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RUN_ID}}"
mkdir -p "$RUN_LOG_DIR"
exec > >(tee -a "$RUN_LOG_DIR/run.out") 2> >(tee -a "$RUN_LOG_DIR/run.err" >&2)
echo "[logs] per-run log dir = $RUN_LOG_DIR"

source /home/0630rb/miniconda3/etc/profile.d/conda.sh
conda activate py311_llm
export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}/src"
export HF_HOME="${HF_HOME:-/scratch2/0630rb/huggingface}"
export TOKENIZERS_PARALLELISM=false

MODEL_ID="${MODEL_ID:-/scratch2/0630rb/models/Qwen3-4B-Base}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-src/checkpoints/exp3_4b_cloze}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
CROSS_NUM_SAMPLES="${CROSS_NUM_SAMPLES:-}"
CROSS_ARG=""
[ -n "$CROSS_NUM_SAMPLES" ] && CROSS_ARG="--cross_num_samples $CROSS_NUM_SAMPLES"
ENTROPY_STEPS="${ENTROPY_STEPS:-20}"
ENTROPY_LR="${ENTROPY_LR:-0.1}"
ENTROPY_TEMPERATURE="${ENTROPY_TEMPERATURE:-1.0}"

echo "=========================================="
echo "MedMCQA CLOZE inference | $(date) | host=$(hostname)"
echo "model=$MODEL_ID checkpoints=$CHECKPOINT_DIR T=$ENTROPY_TEMPERATURE cross=${CROSS_NUM_SAMPLES:-full}"
echo "=========================================="
nvidia-smi || true

python src/inference_medmcqa_cloze.py \
  --model_id "$MODEL_ID" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --num_samples "$NUM_SAMPLES" \
  $CLIENT_ARG \
  $CROSS_ARG \
  --entropy_steps "$ENTROPY_STEPS" \
  --entropy_lr "$ENTROPY_LR" \
  --entropy_temperature "$ENTROPY_TEMPERATURE" \
  --output_dir "$OUTPUT_DIR"

echo "Done | $(date)"
