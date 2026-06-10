#!/bin/bash
#SBATCH --job-name=infer_gen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
# MedMCQA generation 기반 sequence entropy-min 추론.
set -euo pipefail
PROJECT_ROOT="/home/0630rb/yonsei/medical_ai/FedTTA"; cd "$PROJECT_ROOT"; mkdir -p logs outputs
RUN_TS="$(date +%Y%m%d_%H%M%S)"; RUN_TAG="${SLURM_JOB_NAME:-infer_gen}"
RUN_ID="${RUN_TS}_${RUN_TAG}${SLURM_JOB_ID:+_${SLURM_JOB_ID}}"
RUN_LOG_DIR="logs/${RUN_ID}"; OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RUN_ID}}"; mkdir -p "$RUN_LOG_DIR"
exec > >(tee -a "$RUN_LOG_DIR/run.out") 2> >(tee -a "$RUN_LOG_DIR/run.err" >&2)
source /home/0630rb/miniconda3/etc/profile.d/conda.sh; conda activate py311_llm
export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}/src"; export HF_HOME="${HF_HOME:-/scratch2/0630rb/huggingface}"; export TOKENIZERS_PARALLELISM=false
MODEL_ID="${MODEL_ID:-/scratch2/0630rb/models/Qwen3-4B-Base}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-src/checkpoints/exp3_4b_cloze}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"; CROSS_NUM_SAMPLES="${CROSS_NUM_SAMPLES:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"; ENTROPY_TEMPERATURE="${ENTROPY_TEMPERATURE:-1.0}"
CLIENT="${CLIENT:-}"; CLIENT_ARG=""; [ -n "$CLIENT" ] && CLIENT_ARG="--clients $CLIENT"
CROSS_ARG=""; [ -n "$CROSS_NUM_SAMPLES" ] && CROSS_ARG="--cross_num_samples $CROSS_NUM_SAMPLES"
SKIP_CROSS_ARG=""; [ "${SKIP_CROSS:-0}" = "1" ] && SKIP_CROSS_ARG="--skip_cross"
echo "gen-entropy | model=$MODEL_ID ckpt=$CHECKPOINT_DIR T=$ENTROPY_TEMPERATURE mnt=$MAX_NEW_TOKENS"
nvidia-smi || true
python src/inference_medmcqa_gen_entropy.py \
  --model_id "$MODEL_ID" --checkpoint_dir "$CHECKPOINT_DIR" \
  --num_samples "$NUM_SAMPLES" $CLIENT_ARG $CROSS_ARG $SKIP_CROSS_ARG \
  --max_new_tokens "$MAX_NEW_TOKENS" --entropy_temperature "$ENTROPY_TEMPERATURE" \
  --output_dir "$OUTPUT_DIR"
echo "Done | $(date)"
