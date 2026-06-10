#!/bin/bash

#=========================================================================================
# - train_job.sh는 plain python 실행 기준의 FedDPA-F 학습 스크립트다.
# - Stage 1: 모든 client가 매 round global LoRA만 학습/업로드하고, 서버는 single-global FedAvg를 수행한다.
# - Stage 2: 최종 global LoRA를 모든 client에 다시 broadcast한 뒤, local align + local personalization을 수행한다.
#=========================================================================================

set -euo pipefail

PROJECT_ROOT="/home/kimmg/research/research_personalization_FedDPA-F"
cd "$PROJECT_ROOT"
mkdir -p logs outputs src/checkpoints

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
RUN_LOG_DIR="logs/${CURRENT_TIME}_train"
RUN_OUTPUT_DIR="outputs/${CURRENT_TIME}_train"
RUN_CHECKPOINT_DIR="src/checkpoints/${CURRENT_TIME}_train"
mkdir -p "$RUN_LOG_DIR" "$RUN_OUTPUT_DIR" "$RUN_CHECKPOINT_DIR"
RUN_STDOUT="${RUN_LOG_DIR}/train_${CURRENT_TIME}.out"
exec > "$RUN_STDOUT" 2>&1

GPUMON_INTERVAL="${GPUMON_INTERVAL:-5}"
GPU_MONITOR_LOG="${RUN_LOG_DIR}/gpu_metrics_${CURRENT_TIME}.csv"
GPU_MONITOR_PID=""

start_gpu_monitor() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[GPU Monitor] nvidia-smi not found. Skipping GPU monitor."
    return
  fi

  echo "timestamp,index,name,utilization_gpu,utilization_memory,memory_used_mb,memory_total_mb,power_draw_w,temperature_c" > "$GPU_MONITOR_LOG"

  (
    while true; do
      TS=$(date +"%Y-%m-%d %H:%M:%S")
      nvidia-smi \
        --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv,noheader,nounits \
        | awk -v ts="$TS" -F', ' '{print ts "," $1 "," $2 "," $3 "," $4 "," $5 "," $6 "," $7 "," $8}'
      sleep "$GPUMON_INTERVAL"
    done
  ) >> "$GPU_MONITOR_LOG" 2>/dev/null &

  GPU_MONITOR_PID=$!
  echo "[GPU Monitor] started (pid=${GPU_MONITOR_PID}, interval=${GPUMON_INTERVAL}s, log=${GPU_MONITOR_LOG})"
}

stop_gpu_monitor() {
  if [ -n "$GPU_MONITOR_PID" ] && kill -0 "$GPU_MONITOR_PID" >/dev/null 2>&1; then
    kill "$GPU_MONITOR_PID" >/dev/null 2>&1 || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
    echo "[GPU Monitor] stopped"
  fi
}

finish_log() {
  EXIT_CODE=$?
  stop_gpu_monitor
  echo "=========================================="
  echo "Train job finished on $(date)"
  echo "Exit Code: ${EXIT_CODE}"
  echo "Log file: ${RUN_STDOUT}"
  echo "=========================================="
}
trap finish_log EXIT

source /home/kimmg/miniconda3/etc/profile.d/conda.sh
conda activate py311_blackwell
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export RUN_TIMESTAMP="${CURRENT_TIME}_train"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="$(pwd)/outputs/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

MODEL_ID="${MODEL_ID:-meta-llama/Llama-2-7b-hf}"
DATASET_NAME="${DATASET_NAME:-dataset1}"
NUM_ROUNDS="${NUM_ROUNDS:-20}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-10}"
PERSONALIZATION_EPOCHS="${PERSONALIZATION_EPOCHS:-10}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
if [ $((EFFECTIVE_BATCH_SIZE % MICRO_BATCH_SIZE)) -ne 0 ]; then
  echo "[Config Error] EFFECTIVE_BATCH_SIZE (${EFFECTIVE_BATCH_SIZE}) must be divisible by MICRO_BATCH_SIZE (${MICRO_BATCH_SIZE})"
  exit 1
fi
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$((EFFECTIVE_BATCH_SIZE / MICRO_BATCH_SIZE))}"
NUM_SAMPLES="${NUM_SAMPLES:-300}"
MAX_LENGTH="${MAX_LENGTH:-512}"
RANK="${RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LEARNING_RATE="${LEARNING_RATE:-0.0003}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,v_proj}"
SAVE_EPOCH_SNAPSHOTS="${SAVE_EPOCH_SNAPSHOTS:-0}"
NUM_CLIENTS="${NUM_CLIENTS:-8}"
SEED="${SEED:-42}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-research_personalization}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-feddpa_train_${DATASET_NAME}_r${RANK}_round${NUM_ROUNDS}_${CURRENT_TIME}}"

OURS_STDOUT="${RUN_OUTPUT_DIR}/train_ours_${CURRENT_TIME}.out"
OURS_STDERR="${RUN_LOG_DIR}/train_ours_${CURRENT_TIME}.err"
FINAL_ANALYSIS_DIR="${RUN_OUTPUT_DIR}/global_lora_analysis/${DATASET_NAME}_round_${NUM_ROUNDS}_clients_${NUM_CLIENTS}"
POST_SUMMARY_JSON="${RUN_OUTPUT_DIR}/post_training_summary_${CURRENT_TIME}.json"

echo "=========================================="
echo "FedDPA-F training run"
echo "Start Time: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "MPLCONFIGDIR: ${MPLCONFIGDIR}"
echo "Model ID: ${MODEL_ID}"
echo "Dataset: ${DATASET_NAME}"
echo "Rounds: ${NUM_ROUNDS}"
echo "Stage-1 Local Epochs (global-only FL): ${LOCAL_EPOCHS}"
echo "Stage-2 Personalization Epochs: ${PERSONALIZATION_EPOCHS}"
echo "Effective Batch Size: ${EFFECTIVE_BATCH_SIZE}"
echo "Micro Batch Size: ${MICRO_BATCH_SIZE}"
echo "Gradient Accumulation Steps: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Num Samples: ${NUM_SAMPLES}"
echo "Cutoff Len: ${MAX_LENGTH}"
echo "Rank: ${RANK}"
echo "LoRA Alpha: ${LORA_ALPHA}"
echo "LoRA Dropout: ${LORA_DROPOUT}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "LoRA Target Modules: ${LORA_TARGET_MODULES}"
echo "Num Clients: ${NUM_CLIENTS}"
echo "Save Epoch Snapshots: ${SAVE_EPOCH_SNAPSHOTS}"
echo "Run checkpoint dir: $(pwd)/${RUN_CHECKPOINT_DIR}"
echo "=========================================="

echo "[1/2] nvidia-smi"
nvidia-smi || true
echo "[Debug] /dev/nvidia*"
ls -l /dev/nvidia* 2>/dev/null || true

echo "[2/2] plain python CUDA check"
python - <<'PY'
import os
import torch

print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    p = torch.cuda.get_device_properties(0)
    print("gpu:", p.name)
    print("compute capability:", f"{p.major}.{p.minor}")
    x = torch.randn(1024, 1024, device="cuda")
    y = torch.randn(1024, 1024, device="cuda")
    z = x @ y
    print("matmul ok:", tuple(z.shape), z.dtype, z.device)
else:
    print("CUDA not available in plain python context.")
    raise SystemExit(1)
PY

echo "[Preflight] Hugging Face token check"
python - <<'PY'
import os
from pathlib import Path

from utils.training.common import load_hf_token

project_root = Path.cwd()
candidate_paths = [
    project_root / "src" / ".env",
    project_root / "src" / "utils" / ".env",
]
print("HF_TOKEN env present:", bool(os.getenv("HF_TOKEN")))
for path in candidate_paths:
    print(f"dotenv candidate: {path} exists={path.exists()}")

try:
    token = load_hf_token()
except Exception as exc:
    print(f"HF token preflight failed: {exc}")
    raise SystemExit(1)

print(f"HF token preflight ok (length={len(token)})")
PY

echo "[Preflight] wandb availability check"
python - <<'PY'
import importlib.util
import os

use_wandb = os.getenv("USE_WANDB", "1") == "1"
spec = importlib.util.find_spec("wandb")
print("USE_WANDB:", use_wandb)
print("wandb spec origin:", getattr(spec, "origin", None))
if use_wandb:
    try:
        import wandb  # type: ignore
        print("wandb has init:", hasattr(wandb, "init"))
        if not hasattr(wandb, "init"):
            print("wandb preflight warning: local namespace/package conflict detected; training will continue without wandb.")
    except ModuleNotFoundError:
        print("wandb preflight warning: wandb is not installed; training will continue without wandb.")
PY

start_gpu_monitor

TRAIN_ARGS=(
  --method feddpa_f
  --use_wandb
  --wandb_project "$WANDB_PROJECT"
  --wandb_run_name "$WANDB_RUN_NAME"
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

if [ "$SAVE_EPOCH_SNAPSHOTS" = "1" ]; then
  TRAIN_ARGS+=(--save_epoch_snapshots)
fi

echo "[Run] FedDPA-F stage-1 global FL + stage-2 local personalization"
python src/main.py \
  "${TRAIN_ARGS[@]}" \
  > "$OURS_STDOUT" 2> "$OURS_STDERR"

OURS_EXIT_CODE=$?
if [ $OURS_EXIT_CODE -ne 0 ]; then
  echo "[Fail] Training failed with exit code $OURS_EXIT_CODE"
  echo "  stdout: $OURS_STDOUT"
  echo "  stderr: $OURS_STDERR"
  exit $OURS_EXIT_CODE
fi

echo "[Summary] Building post-training summary JSON"
python src/summarize_training_artifacts.py \
  --gpu_csv "$GPU_MONITOR_LOG" \
  --analysis_dir "$FINAL_ANALYSIS_DIR" \
  --output_json "$POST_SUMMARY_JSON" || echo "[Summary] Summary generation failed (non-fatal)."

echo "=========================================="
echo "Train run completed."
echo "Log file: ${RUN_STDOUT}"
echo "Training stdout: ${OURS_STDOUT}"
echo "Training stderr: ${OURS_STDERR}"
echo "GPU CSV: ${GPU_MONITOR_LOG}"
echo "Phase-1 analysis dir: ${FINAL_ANALYSIS_DIR}"
echo "Checkpoint dir: ${PROJECT_ROOT}/${RUN_CHECKPOINT_DIR}"
echo "Post summary json: ${POST_SUMMARY_JSON}"
echo "=========================================="
