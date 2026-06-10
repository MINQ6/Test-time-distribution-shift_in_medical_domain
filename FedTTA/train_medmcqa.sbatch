#!/bin/bash
#SBATCH --job-name=feddpa_medmcqa
#SBATCH --partition=suma_rtx4090
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
# =========================================================================
# MedMCQA × FedDPA-F (Qwen3-1.7B-Base, LoRA r=8 q/v, 4 clients) 학습.
# - Stage 1: 매 round 모든 client가 global LoRA만 학습/업로드 → single-global FedAvg
# - Stage 2: 최종 global broadcast 후 client별 local LoRA personalization
# 결과: src/checkpoints/<run>/dual_lora_adapter_client_{1..4}.pth  (global+local)
#
# 제출:   sbatch train_medmcqa.sbatch
# 노드지정: sbatch -w node33 train_medmcqa.sbatch
# 덮어쓰기: NUM_ROUNDS=10 LOCAL_EPOCHS=2 sbatch train_medmcqa.sbatch
# =========================================================================
set -euo pipefail

PROJECT_ROOT="/home/0630rb/yonsei/medical_ai/FedTTA"
cd "$PROJECT_ROOT"
mkdir -p logs outputs src/checkpoints

# --- per-run 로그 폴더: logs/<timestamp>_<jobname>/ (outputs 폴더와 동일 컨벤션) ---
# 실험 stdout/stderr 를 이 폴더 안 run.out/run.err 로도 기록(슬럼 로그와 병행).
# RUN_TS 는 로그 폴더 이름 전용 변수다. 체크포인트 경로를 정하는 RUN_TIMESTAMP 와는
# 분리되어 있어, 기존 src/checkpoints/ (또는 src/checkpoints/<RUN_TIMESTAMP>/) 동작을 바꾸지 않는다.
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${SLURM_JOB_NAME:-feddpa_medmcqa}"
# job id 접미사로 동시 제출(같은 초) 시 폴더명 충돌/덮어쓰기 방지
RUN_ID="${RUN_TS}_${RUN_TAG}${SLURM_JOB_ID:+_${SLURM_JOB_ID}}"
RUN_LOG_DIR="logs/${RUN_ID}"
mkdir -p "$RUN_LOG_DIR"
exec > >(tee -a "$RUN_LOG_DIR/run.out") 2> >(tee -a "$RUN_LOG_DIR/run.err" >&2)
echo "[logs] per-run log dir = $RUN_LOG_DIR"

# --- conda ---
source /home/0630rb/miniconda3/etc/profile.d/conda.sh
conda activate py311_llm
export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
# 할당된 GPU 수만큼 client를 병렬 학습 (1이면 순차). --gres=gpu:4로 제출하면 자동 4-way.
export PARALLEL_CLIENT_GPUS="${PARALLEL_CLIENT_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"

# --- hyperparameters (env로 덮어쓰기 가능) ---
METHOD="${METHOD:-feddpa_f}"
# local_finetuned 등은 RUN_TIMESTAMP 필수(체크포인트를 src/checkpoints/<ts>/에 저장). 있으면 export.
[ -n "${RUN_TIMESTAMP:-}" ] && export RUN_TIMESTAMP
# 로컬 scratch2 모델 경로 사용 (compute 노드 HF 다운로드 불필요)
MODEL_ID="${MODEL_ID:-/scratch2/0630rb/models/Qwen3-1.7B-Base}"
export HF_HOME="${HF_HOME:-/scratch2/0630rb/huggingface}"
DATASET_NAME="${DATASET_NAME:-medmcqa}"
NUM_CLIENTS="${NUM_CLIENTS:-4}"
NUM_ROUNDS="${NUM_ROUNDS:-20}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-3}"              # Stage1: round당 client global LoRA 학습 epoch
PERSONALIZATION_EPOCHS="${PERSONALIZATION_EPOCHS:-5}"  # Stage2: local personalization epoch
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-300}"
MAX_LENGTH="${MAX_LENGTH:-512}"
RANK="${RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LEARNING_RATE="${LEARNING_RATE:-0.0003}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,v_proj}"
SEED="${SEED:-42}"
USE_WANDB="${USE_WANDB:-0}"

if [ $((EFFECTIVE_BATCH_SIZE % MICRO_BATCH_SIZE)) -ne 0 ]; then
  echo "[Config Error] EFFECTIVE_BATCH_SIZE($EFFECTIVE_BATCH_SIZE) % MICRO_BATCH_SIZE($MICRO_BATCH_SIZE) != 0"; exit 1
fi
GRAD_ACCUM=$((EFFECTIVE_BATCH_SIZE / MICRO_BATCH_SIZE))

echo "=========================================="
echo "FedDPA-F MedMCQA train | $(date) | host=$(hostname)"
echo "model=$MODEL_ID dataset=$DATASET_NAME clients=$NUM_CLIENTS"
echo "rounds=$NUM_ROUNDS local_epochs=$LOCAL_EPOCHS pers_epochs=$PERSONALIZATION_EPOCHS"
echo "rank=$RANK targets=$LORA_TARGET_MODULES lr=$LEARNING_RATE bs=$EFFECTIVE_BATCH_SIZE/$MICRO_BATCH_SIZE(ga=$GRAD_ACCUM)"
echo "PARALLEL_CLIENT_GPUS=$PARALLEL_CLIENT_GPUS (SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset})"
echo "=========================================="
nvidia-smi || true
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

# --- 데이터 준비 (없으면 생성) ---
if [ ! -f "src/data/medmcqa/Medicine_train.json" ]; then
  echo "[data] medmcqa not found -> running prepare_medmcqa.py"
  python src/data/prepare_medmcqa.py
fi

TRAIN_ARGS=(
  --method "$METHOD"
  --model_id "$MODEL_ID"
  --dataset_name "$DATASET_NAME"
  --num_clients "$NUM_CLIENTS"
  --num_rounds "$NUM_ROUNDS"
  --local_epochs "$LOCAL_EPOCHS"
  --personalization_epochs "$PERSONALIZATION_EPOCHS"
  --batch_size "$EFFECTIVE_BATCH_SIZE"
  --micro_batch_size "$MICRO_BATCH_SIZE"
  --gradient_accumulation_steps "$GRAD_ACCUM"
  --num_samples "$NUM_SAMPLES"
  --max_length "$MAX_LENGTH"
  --seed "$SEED"
  --rank "$RANK"
  --lora_alpha "$LORA_ALPHA"
  --lora_dropout "$LORA_DROPOUT"
  --learning_rate "$LEARNING_RATE"
  --lora_target_modules "$LORA_TARGET_MODULES"
)
if [ "$USE_WANDB" = "1" ]; then
  TRAIN_ARGS+=(--use_wandb --wandb_project research_personalization \
               --wandb_run_name "feddpa_medmcqa_r${RANK}_round${NUM_ROUNDS}_${SLURM_JOB_ID:-local}")
fi

echo "[Run] python src/main.py ${TRAIN_ARGS[*]}"
python src/main.py "${TRAIN_ARGS[@]}"

echo "=========================================="
echo "Done | $(date)"
echo "Checkpoints: ${PROJECT_ROOT}/src/checkpoints/ (latest run dir)"
echo "=========================================="
