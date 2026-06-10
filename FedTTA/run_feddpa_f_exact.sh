#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'EOF'
Usage: ./run_feddpa_f_exact.sh

Runs the FedDPA-F public-code-aligned pipeline:
  1. train FedDPA-F
  2. run personalization + test-time personalization inference
  3. render the paper-style PNG table

Useful environment overrides:
  RUN_ID, RUN_TAG, MODEL_ID, DATASET_NAME, NUM_ROUNDS, LOCAL_EPOCHS,
  PERSONALIZATION_EPOCHS, EFFECTIVE_BATCH_SIZE, MICRO_BATCH_SIZE,
  NUM_SAMPLES, USE_WANDB, AUTO_LAMBDA, AUTO_NUM_INSTANCES
EOF
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_TAG="${RUN_TAG:-feddpa_f_exact}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${RUN_TAG}}"
OUTPUT_DIR="${ROOT_DIR}/outputs/${RUN_ID}"
LOG_DIR="${ROOT_DIR}/logs/${RUN_ID}"
CHECKPOINT_DIR="${ROOT_DIR}/src/checkpoints/${RUN_ID}"
RUN_LOG="${LOG_DIR}/run.log"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$CHECKPOINT_DIR" "${OUTPUT_DIR}/.mplconfig"
exec > "$RUN_LOG" 2>&1

finish_log() {
  local exit_code=$?
  echo "=========================================="
  echo "FedDPA-F exact run finished: $(date)"
  echo "Exit Code: ${exit_code}"
  echo "Output dir: ${OUTPUT_DIR}"
  echo "Log file: ${RUN_LOG}"
  echo "=========================================="
}
trap finish_log EXIT

source /home/kimmg/miniconda3/etc/profile.d/conda.sh
conda activate py311_blackwell

export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}/src"
export RUN_TIMESTAMP="${RUN_ID}"
export LD_LIBRARY_PATH="/home/kimmg/miniconda3/envs/py311_blackwell/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${OUTPUT_DIR}/.mplconfig"

MODEL_ID="${MODEL_ID:-meta-llama/Llama-2-7b-hf}"
DATASET_NAME="${DATASET_NAME:-dataset1}"
NUM_CLIENTS="${NUM_CLIENTS:-8}"
NUM_ROUNDS="${NUM_ROUNDS:-20}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-10}"
PERSONALIZATION_EPOCHS="${PERSONALIZATION_EPOCHS:-10}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$((EFFECTIVE_BATCH_SIZE / MICRO_BATCH_SIZE))}"
NUM_SAMPLES="${NUM_SAMPLES:-300}"
MAX_LENGTH="${MAX_LENGTH:-512}"
SEED="${SEED:-42}"
RANK="${RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LEARNING_RATE="${LEARNING_RATE:-0.0003}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,v_proj}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-research_personalization}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_ID}}"
STATIC_GLOBAL_WEIGHT="${STATIC_GLOBAL_WEIGHT:-0.5}"
AUTO_LAMBDA="${AUTO_LAMBDA:-1.0}"
AUTO_NUM_INSTANCES="${AUTO_NUM_INSTANCES:-1}"

INFERENCE_OUTPUT_DIR="${OUTPUT_DIR}/inference_fedDPA_fixed_full_ttp"
TABLE_OUTPUT_DIR="${OUTPUT_DIR}/inference_fedDPA_table"

if [ $((EFFECTIVE_BATCH_SIZE % MICRO_BATCH_SIZE)) -ne 0 ]; then
  echo "[Config Error] EFFECTIVE_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE"
  exit 1
fi

echo "=========================================="
echo "FedDPA-F exact public-code-aligned run"
echo "Run ID: ${RUN_ID}"
echo "Model: ${MODEL_ID}"
echo "Dataset: ${DATASET_NAME}"
echo "Rounds: ${NUM_ROUNDS}"
echo "Local epochs: ${LOCAL_EPOCHS}"
echo "Personalization epochs: ${PERSONALIZATION_EPOCHS}"
echo "Effective batch: ${EFFECTIVE_BATCH_SIZE}"
echo "Micro batch: ${MICRO_BATCH_SIZE}"
echo "Grad accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Training samples per client: ${NUM_SAMPLES}"
echo "Inference TTP: auto, S=${AUTO_NUM_INSTANCES}, lambda=${AUTO_LAMBDA}, emb_type=last, random_per_sample"
echo "Inference personalization: static global/local weight = ${STATIC_GLOBAL_WEIGHT}/1-${STATIC_GLOBAL_WEIGHT}"
echo "=========================================="

nvidia-smi || true

TRAIN_ARGS=(
  --method feddpa_f
  --model_id "$MODEL_ID"
  --dataset_name "$DATASET_NAME"
  --num_clients "$NUM_CLIENTS"
  --num_rounds "$NUM_ROUNDS"
  --local_epochs "$LOCAL_EPOCHS"
  --personalization_epochs "$PERSONALIZATION_EPOCHS"
  --batch_size "$EFFECTIVE_BATCH_SIZE"
  --micro_batch_size "$MICRO_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
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
  TRAIN_ARGS+=(--use_wandb --wandb_project "$WANDB_PROJECT" --wandb_run_name "$WANDB_RUN_NAME")
fi

echo "[1/3] Training FedDPA-F"
python src/main.py "${TRAIN_ARGS[@]}"

echo "[2/3] Inference FedDPA-F"
python src/utils/inference/inference_fedDPA.py \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --output_dir "$INFERENCE_OUTPUT_DIR" \
  --log_dir "$LOG_DIR" \
  --model_id "$MODEL_ID" \
  --dataset_name "$DATASET_NAME" \
  --num_samples 200 \
  --inference_batch_size 8 \
  --personalization_adapter_mode static \
  --static_global_weight "$STATIC_GLOBAL_WEIGHT" \
  --ttp_adapter_mode auto \
  --auto_num_instances "$AUTO_NUM_INSTANCES" \
  --auto_lambda "$AUTO_LAMBDA" \
  --auto_emb_type last \
  --auto_reference_strategy random_per_sample

echo "[3/3] Render PNG table"
python src/utils/inference/render_feddpa_table.py \
  --summary_json "${INFERENCE_OUTPUT_DIR}/summary.json" \
  --output_dir "$TABLE_OUTPUT_DIR" \
  --log_dir "$LOG_DIR"

echo "PNG table: ${TABLE_OUTPUT_DIR}/fedDPA-F_results_table.png"
