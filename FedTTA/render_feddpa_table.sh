#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TAG="${RUN_TAG:-inference_fedDPA_table}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${RUN_TAG}}"

SUMMARY_JSON="${1:-${ROOT_DIR}/outputs/inference_fedDPA_fixed_full_ttp/summary.json}"
OUTPUT_DIR="${ROOT_DIR}/outputs/${RUN_ID}"
LOG_DIR="${ROOT_DIR}/logs/${RUN_ID}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

env LD_LIBRARY_PATH=/home/kimmg/miniconda3/envs/py311_blackwell/lib \
  conda run -n py311_blackwell \
  python "${ROOT_DIR}/src/utils/inference/render_feddpa_table.py" \
    --summary_json "${SUMMARY_JSON}" \
    --output_dir "${OUTPUT_DIR}" \
    --log_dir "${LOG_DIR}"

echo "PNG: ${OUTPUT_DIR}/fedDPA-F_results_table.png"
echo "LOG: ${LOG_DIR}/render_feddpa_table.log"
