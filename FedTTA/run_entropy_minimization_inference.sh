#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'EOF'
Usage: ./run_entropy_minimization_inference.sh

Runs entropy-minimization TTA inference and renders a FedDPA-F comparison PNG.

Useful environment overrides:
  RUN_ID, MODEL_ID, DATASET_NAME, CHECKPOINT_DIR, FEDDPA_SUMMARY_JSON,
  NUM_SAMPLES, TTP_NUM_SAMPLES, CLIENTS, ENTROPY_STEPS, ENTROPY_LR,
  MAX_NEW_TOKENS, DECODING_STRATEGY, NUM_BEAMS, SAVE_PREDICTIONS
EOF
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_TAG="${RUN_TAG:-entropy_minimization_inference}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${RUN_TAG}}"
OUTPUT_DIR="${ROOT_DIR}/outputs/${RUN_ID}"
LOG_DIR="${ROOT_DIR}/logs/${RUN_ID}"
RUN_LOG="${LOG_DIR}/run.log"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "${OUTPUT_DIR}/.mplconfig"
exec > "$RUN_LOG" 2>&1

finish_log() {
  local exit_code=$?
  echo "=========================================="
  echo "Entropy-minimization inference finished: $(date)"
  echo "Exit Code: ${exit_code}"
  echo "Output dir: ${OUTPUT_DIR}"
  echo "Log file: ${RUN_LOG}"
  echo "=========================================="
}
trap finish_log EXIT

source /home/kimmg/miniconda3/etc/profile.d/conda.sh
conda activate py311_blackwell

export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}/src"
export LD_LIBRARY_PATH="/home/kimmg/miniconda3/envs/py311_blackwell/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${OUTPUT_DIR}/.mplconfig"

MODEL_ID="${MODEL_ID:-meta-llama/Llama-2-7b-hf}"
DATASET_NAME="${DATASET_NAME:-dataset1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/src/checkpoints/20260424_160003_train}"
FEDDPA_SUMMARY_JSON="${FEDDPA_SUMMARY_JSON:-${ROOT_DIR}/outputs/inference_fedDPA_fixed_full_ttp/summary.json}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
TTP_NUM_SAMPLES="${TTP_NUM_SAMPLES:-}"
CLIENTS="${CLIENTS:-}"
ENTROPY_STEPS="${ENTROPY_STEPS:-20}"
ENTROPY_LR="${ENTROPY_LR:-0.1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-80}"
DECODING_STRATEGY="${DECODING_STRATEGY:-greedy}"
NUM_BEAMS="${NUM_BEAMS:-4}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"

ENTROPY_OUTPUT_DIR="${OUTPUT_DIR}/inference_entropy_minimization"
TABLE_OUTPUT_DIR="${OUTPUT_DIR}/comparison_table"

echo "=========================================="
echo "Entropy-minimization TTA inference"
echo "Run ID: ${RUN_ID}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "FedDPA summary: ${FEDDPA_SUMMARY_JSON}"
echo "Dataset: ${DATASET_NAME}"
echo "Num samples: ${NUM_SAMPLES}"
echo "TTP samples: ${TTP_NUM_SAMPLES:-full}"
echo "Clients: ${CLIENTS:-all}"
echo "Entropy steps: ${ENTROPY_STEPS}"
echo "Entropy lr: ${ENTROPY_LR}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo "Decoding strategy: ${DECODING_STRATEGY}"
echo "Num beams: ${NUM_BEAMS}"
echo "=========================================="

INFER_ARGS=(
  python src/utils/inference/inference_entropy_minimization.py
  --checkpoint_dir "$CHECKPOINT_DIR"
  --output_dir "$ENTROPY_OUTPUT_DIR"
  --log_dir "$LOG_DIR"
  --model_id "$MODEL_ID"
  --dataset_name "$DATASET_NAME"
  --num_samples "$NUM_SAMPLES"
  --entropy_steps "$ENTROPY_STEPS"
  --entropy_lr "$ENTROPY_LR"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --decoding_strategy "$DECODING_STRATEGY"
  --num_beams "$NUM_BEAMS"
)

if [ -n "$TTP_NUM_SAMPLES" ]; then
  INFER_ARGS+=(--ttp_num_samples "$TTP_NUM_SAMPLES")
fi

if [ -n "$CLIENTS" ]; then
  # shellcheck disable=SC2206
  CLIENT_ARRAY=($CLIENTS)
  INFER_ARGS+=(--clients "${CLIENT_ARRAY[@]}")
fi

if [ "$SAVE_PREDICTIONS" = "1" ]; then
  INFER_ARGS+=(--save_predictions)
fi

echo "[1/2] Run entropy-minimization inference"
"${INFER_ARGS[@]}"

echo "[2/2] Render comparison table"
python src/utils/inference/render_entropy_comparison_table.py \
  --feddpa_summary_json "$FEDDPA_SUMMARY_JSON" \
  --entropy_summary_json "${ENTROPY_OUTPUT_DIR}/summary.json" \
  --output_dir "$TABLE_OUTPUT_DIR" \
  --log_dir "$LOG_DIR"

echo "Entropy summary: ${ENTROPY_OUTPUT_DIR}/summary.json"
echo "Comparison PNG: ${TABLE_OUTPUT_DIR}/fedDPA_vs_entropy_minimization_table.png"
