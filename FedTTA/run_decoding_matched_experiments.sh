#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'EOF'
Usage: ./run_decoding_matched_experiments.sh

Runs two matched-decoding comparisons:
  1. Greedy: FedDPA-F greedy vs Entropy-Min greedy
  2. Beam:   FedDPA-F beam search vs Entropy-Min beam search

Useful environment overrides:
  RUN_ID, CHECKPOINT_DIR, MODEL_ID, DATASET_NAME, NUM_SAMPLES, TTP_NUM_SAMPLES,
  CLIENTS, SEED, ENTROPY_STEPS, ENTROPY_LR, MAX_NEW_TOKENS, NUM_BEAMS,
  RUN_GREEDY, RUN_BEAM, SAVE_PREDICTIONS
EOF
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_TAG="${RUN_TAG:-decoding_matched_experiments}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${RUN_TAG}}"
OUTPUT_DIR="${ROOT_DIR}/outputs/${RUN_ID}"
LOG_DIR="${ROOT_DIR}/logs/${RUN_ID}"
RUN_LOG="${LOG_DIR}/run.log"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "${OUTPUT_DIR}/.mplconfig"
exec > "$RUN_LOG" 2>&1

finish_log() {
  local exit_code=$?
  echo "=========================================="
  echo "Decoding-matched experiments finished: $(date)"
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
SEED="${SEED:-42}"
ENTROPY_STEPS="${ENTROPY_STEPS:-20}"
ENTROPY_LR="${ENTROPY_LR:-0.1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-80}"
NUM_BEAMS="${NUM_BEAMS:-4}"
RUN_GREEDY="${RUN_GREEDY:-1}"
RUN_BEAM="${RUN_BEAM:-1}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"

common_feddpa_args=(
  --checkpoint_dir "$CHECKPOINT_DIR"
  --model_id "$MODEL_ID"
  --dataset_name "$DATASET_NAME"
  --num_samples "$NUM_SAMPLES"
  --inference_batch_size 8
  --personalization_adapter_mode static
  --static_global_weight 0.5
  --ttp_adapter_mode auto
  --auto_num_instances 1
  --auto_lambda 1.0
  --auto_emb_type last
  --auto_reference_strategy random_per_sample
  --max_new_tokens "$MAX_NEW_TOKENS"
  --seed "$SEED"
)

common_entropy_args=(
  --checkpoint_dir "$CHECKPOINT_DIR"
  --model_id "$MODEL_ID"
  --dataset_name "$DATASET_NAME"
  --num_samples "$NUM_SAMPLES"
  --entropy_steps "$ENTROPY_STEPS"
  --entropy_lr "$ENTROPY_LR"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --seed "$SEED"
)

if [ -n "$TTP_NUM_SAMPLES" ]; then
  common_feddpa_args+=(--ttp_num_samples "$TTP_NUM_SAMPLES")
  common_entropy_args+=(--ttp_num_samples "$TTP_NUM_SAMPLES")
fi

if [ -n "$CLIENTS" ]; then
  # shellcheck disable=SC2206
  CLIENT_ARRAY=($CLIENTS)
  common_feddpa_args+=(--clients "${CLIENT_ARRAY[@]}")
  common_entropy_args+=(--clients "${CLIENT_ARRAY[@]}")
fi

if [ "$SAVE_PREDICTIONS" = "1" ]; then
  common_feddpa_args+=(--save_predictions)
  common_entropy_args+=(--save_predictions)
fi

render_comparison() {
  local feddpa_summary="$1"
  local entropy_summary="$2"
  local table_dir="$3"
  python src/utils/inference/render_entropy_comparison_table.py \
    --feddpa_summary_json "$feddpa_summary" \
    --entropy_summary_json "$entropy_summary" \
    --output_dir "$table_dir" \
    --log_dir "$LOG_DIR"
}

echo "=========================================="
echo "Decoding-matched FedDPA-F vs Entropy-Min experiments"
echo "Run ID: ${RUN_ID}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "Dataset: ${DATASET_NAME}"
echo "Num samples: ${NUM_SAMPLES}"
echo "TTP samples: ${TTP_NUM_SAMPLES:-full}"
echo "Clients: ${CLIENTS:-all}"
echo "Seed: ${SEED}"
echo "Entropy steps: ${ENTROPY_STEPS}"
echo "Entropy lr: ${ENTROPY_LR}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo "Beam count: ${NUM_BEAMS}"
echo "=========================================="

if [ "$RUN_GREEDY" = "1" ]; then
  echo "[Greedy 1/3] FedDPA-F greedy"
  python src/utils/inference/inference_fedDPA.py \
    "${common_feddpa_args[@]}" \
    --num_beams 1 \
    --output_dir "${OUTPUT_DIR}/greedy/feddpa" \
    --log_dir "$LOG_DIR"

  echo "[Greedy 2/3] Entropy-Min greedy"
  python src/utils/inference/inference_entropy_minimization.py \
    "${common_entropy_args[@]}" \
    --decoding_strategy greedy \
    --num_beams 1 \
    --output_dir "${OUTPUT_DIR}/greedy/entropy_min" \
    --log_dir "$LOG_DIR"

  echo "[Greedy 3/3] Render comparison"
  render_comparison \
    "${OUTPUT_DIR}/greedy/feddpa/summary.json" \
    "${OUTPUT_DIR}/greedy/entropy_min/summary.json" \
    "${OUTPUT_DIR}/greedy/comparison_table"
fi

if [ "$RUN_BEAM" = "1" ]; then
  echo "[Beam 1/3] FedDPA-F beam"
  python src/utils/inference/inference_fedDPA.py \
    "${common_feddpa_args[@]}" \
    --num_beams "$NUM_BEAMS" \
    --output_dir "${OUTPUT_DIR}/beam/feddpa" \
    --log_dir "$LOG_DIR"

  echo "[Beam 2/3] Entropy-Min beam"
  python src/utils/inference/inference_entropy_minimization.py \
    "${common_entropy_args[@]}" \
    --decoding_strategy beam \
    --num_beams "$NUM_BEAMS" \
    --output_dir "${OUTPUT_DIR}/beam/entropy_min" \
    --log_dir "$LOG_DIR"

  echo "[Beam 3/3] Render comparison"
  render_comparison \
    "${OUTPUT_DIR}/beam/feddpa/summary.json" \
    "${OUTPUT_DIR}/beam/entropy_min/summary.json" \
    "${OUTPUT_DIR}/beam/comparison_table"
fi

echo "Greedy PNG: ${OUTPUT_DIR}/greedy/comparison_table/fedDPA_vs_entropy_minimization_table.png"
echo "Beam PNG: ${OUTPUT_DIR}/beam/comparison_table/fedDPA_vs_entropy_minimization_table.png"
