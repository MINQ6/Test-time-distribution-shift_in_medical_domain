#!/bin/bash
#SBATCH --job-name=clust_medmcqa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
# MedMCQA warm-up hierarchical clustering only (phase 0). K는 silhouette/gap/elbow로 자동 선택.
set -euo pipefail
PROJECT_ROOT="/home/0630rb/yonsei/medical_ai/FedTTA"; cd "$PROJECT_ROOT"; mkdir -p logs outputs src/checkpoints
RUN_TS="$(date +%Y%m%d_%H%M%S)"; RUN_TAG="${SLURM_JOB_NAME:-clust_medmcqa}"
RUN_LOG_DIR="logs/${RUN_TS}_${RUN_TAG}${SLURM_JOB_ID:+_${SLURM_JOB_ID}}"; mkdir -p "$RUN_LOG_DIR"
exec > >(tee -a "$RUN_LOG_DIR/run.out") 2> >(tee -a "$RUN_LOG_DIR/run.err" >&2)
source /home/0630rb/miniconda3/etc/profile.d/conda.sh; conda activate py311_llm
export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}/src"; export HF_HOME="${HF_HOME:-/scratch2/0630rb/huggingface}"; export TOKENIZERS_PARALLELISM=false
export PARALLEL_CLIENT_GPUS=1
[ -n "${RUN_TIMESTAMP:-}" ] && export RUN_TIMESTAMP
MODEL_ID="${MODEL_ID:-/scratch2/0630rb/models/Qwen3-4B-Base}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
NUM_SAMPLES="${NUM_SAMPLES:-600}"
echo "=== warm-up clustering | warmup_epochs=$WARMUP_EPOCHS | model=$MODEL_ID | rank8 q/v ==="
nvidia-smi || true
python src/main_clustering.py \
  --model_id "$MODEL_ID" \
  --dataset_name medmcqa \
  --num_clients 4 \
  --num_rounds 1 \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --num_samples "$NUM_SAMPLES" \
  --max_length 512 \
  --batch_size 32 --micro_batch_size 8 \
  --rank 8 --lora_alpha 16 --lora_dropout 0.05 --learning_rate 0.0003 \
  --lora_target_modules "q_proj,v_proj" \
  --seed 42
echo "Done | $(date)"
echo "결과: src/checkpoints/${RUN_TIMESTAMP:-?}/cluster_assignments*.json"
