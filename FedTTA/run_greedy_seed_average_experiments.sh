#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'EOF'
Usage: ./run_greedy_seed_average_experiments.sh

Runs greedy FedDPA-F vs Entropy-Min inference over multiple seeds and renders
a single PNG table with mean and variance.

Useful environment overrides:
  RUN_ID, CHECKPOINT_DIR, MODEL_ID, DATASET_NAME, NUM_SAMPLES, TTP_NUM_SAMPLES,
  CLIENTS, SEEDS, ENTROPY_STEPS, ENTROPY_LR, MAX_NEW_TOKENS, SAVE_PREDICTIONS

Example:
  SEEDS="42 43 44" ./run_greedy_seed_average_experiments.sh
EOF
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_TAG="${RUN_TAG:-greedy_seed_average_experiments}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${RUN_TAG}}"
OUTPUT_DIR="${ROOT_DIR}/outputs/${RUN_ID}"
LOG_DIR="${ROOT_DIR}/logs/${RUN_ID}"
RUN_LOG="${LOG_DIR}/run.log"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "${OUTPUT_DIR}/.mplconfig"
exec > "$RUN_LOG" 2>&1

finish_log() {
  local exit_code=$?
  echo "=========================================="
  echo "Greedy seed-average experiments finished: $(date)"
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
NUM_SAMPLES="${NUM_SAMPLES:-200}"
TTP_NUM_SAMPLES="${TTP_NUM_SAMPLES:-}"
CLIENTS="${CLIENTS:-}"
SEEDS="${SEEDS:-42 43 44}"
ENTROPY_STEPS="${ENTROPY_STEPS:-20}"
ENTROPY_LR="${ENTROPY_LR:-0.1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-80}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"

echo "=========================================="
echo "Greedy FedDPA-F vs Entropy-Min seed-average experiments"
echo "Run ID: ${RUN_ID}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "Dataset: ${DATASET_NAME}"
echo "Num samples: ${NUM_SAMPLES}"
echo "TTP samples: ${TTP_NUM_SAMPLES:-full}"
echo "Clients: ${CLIENTS:-all}"
echo "Seeds: ${SEEDS}"
echo "Entropy steps: ${ENTROPY_STEPS}"
echo "Entropy lr: ${ENTROPY_LR}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo "=========================================="

FEDDPA_SUMMARIES=()
ENTROPY_SUMMARIES=()

# shellcheck disable=SC2206
SEED_ARRAY=($SEEDS)
for seed in "${SEED_ARRAY[@]}"; do
  seed_run_id="${RUN_ID}/seed_${seed}"
  echo "[Seed ${seed}] Running greedy comparison"
  RUN_ID="$seed_run_id" \
  CHECKPOINT_DIR="$CHECKPOINT_DIR" \
  MODEL_ID="$MODEL_ID" \
  DATASET_NAME="$DATASET_NAME" \
  NUM_SAMPLES="$NUM_SAMPLES" \
  TTP_NUM_SAMPLES="$TTP_NUM_SAMPLES" \
  CLIENTS="$CLIENTS" \
  SEED="$seed" \
  ENTROPY_STEPS="$ENTROPY_STEPS" \
  ENTROPY_LR="$ENTROPY_LR" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  RUN_GREEDY=1 \
  RUN_BEAM=0 \
  SAVE_PREDICTIONS="$SAVE_PREDICTIONS" \
  ./run_decoding_matched_experiments.sh

  FEDDPA_SUMMARIES+=("${OUTPUT_DIR}/seed_${seed}/greedy/feddpa/summary.json")
  ENTROPY_SUMMARIES+=("${OUTPUT_DIR}/seed_${seed}/greedy/entropy_min/summary.json")
done

echo "[Aggregate] Rendering seed-average table"
python src/utils/inference/render_seed_average_comparison_table.py \
  --feddpa_summary_jsons "${FEDDPA_SUMMARIES[@]}" \
  --entropy_summary_jsons "${ENTROPY_SUMMARIES[@]}" \
  --output_dir "${OUTPUT_DIR}/greedy_seed_average_table" \
  --log_dir "$LOG_DIR"

echo "Seed-average PNG: ${OUTPUT_DIR}/greedy_seed_average_table/fedDPA_vs_entropy_minimization_seed_average_table.png"
echo "Seed-average JSON: ${OUTPUT_DIR}/greedy_seed_average_table/seed_average_summary.json"
