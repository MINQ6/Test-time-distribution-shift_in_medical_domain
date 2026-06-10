#!/bin/bash

#=========================================================================================
# check.sh
# - plain bash 실행 기준의 GPU / model smoke test 스크립트
# - py311_blackwell 가상환경에서 torch CUDA 사용 가능 여부를 확인한다.
# - 실제 학습에 쓰는 dual LoRA model 초기화와 짧은 forward pass까지 수행해
#   "모델이 GPU를 할당받아 돌아가는지"를 확인한다.
#=========================================================================================

set -euo pipefail

PROJECT_ROOT="/home/kimmg/research/research_personalization"
cd "$PROJECT_ROOT"
mkdir -p logs outputs

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
RUN_LOG_DIR="logs/${CURRENT_TIME}_check"
RUN_OUTPUT_DIR="outputs/${CURRENT_TIME}_check"
mkdir -p "$RUN_LOG_DIR" "$RUN_OUTPUT_DIR"
RUN_STDOUT="${RUN_LOG_DIR}/check_${CURRENT_TIME}.out"
exec > "$RUN_STDOUT" 2>&1

finish_log() {
  EXIT_CODE=$?
  echo "=========================================="
  echo "Check finished on $(date)"
  echo "Exit Code: ${EXIT_CODE}"
  echo "Log File: ${RUN_STDOUT}"
  echo "Output Dir: ${RUN_OUTPUT_DIR}"
  echo "=========================================="
}
trap finish_log EXIT

source /home/kimmg/miniconda3/etc/profile.d/conda.sh
conda activate py311_blackwell
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="$(pwd)/outputs/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

MODEL_ID="${MODEL_ID:-meta-llama/Llama-2-7b-hf}"
PROMPT_TEXT="${PROMPT_TEXT:-Hello from GPU smoke test.}"
MAX_LENGTH="${MAX_LENGTH:-64}"
RANK="${RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,v_proj}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/src/checkpoints}"
export MODEL_ID PROMPT_TEXT MAX_LENGTH RANK LORA_ALPHA LORA_DROPOUT LORA_TARGET_MODULES CHECKPOINT_ROOT

echo "=========================================="
echo "GPU / Model Smoke Test"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "PythonPath: ${PYTHONPATH}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "MPLCONFIGDIR: ${MPLCONFIGDIR}"
echo "Model ID: ${MODEL_ID}"
echo "Prompt Text: ${PROMPT_TEXT}"
echo "Max Length: ${MAX_LENGTH}"
echo "LoRA Rank: ${RANK}"
echo "LoRA Alpha: ${LORA_ALPHA}"
echo "LoRA Dropout: ${LORA_DROPOUT}"
echo "LoRA Target Modules: ${LORA_TARGET_MODULES}"
echo "Checkpoint Root (current actual dir): ${CHECKPOINT_ROOT}"
echo "=========================================="

echo "[1/3] nvidia-smi"
nvidia-smi || true
echo "[Debug] /dev/nvidia*"
ls -l /dev/nvidia* 2>/dev/null || true

echo "[2/3] torch CUDA sanity check"
python - <<'PY'
import os
import torch

print("python:", os.sys.version.replace("\n", " "))
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    for idx in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(idx)
        print(
            f"device[{idx}]: name={prop.name} "
            f"cc={prop.major}.{prop.minor} "
            f"total_memory_mb={prop.total_memory // (1024 * 1024)}"
        )
    x = torch.randn(2048, 2048, device="cuda")
    y = torch.randn(2048, 2048, device="cuda")
    z = x @ y
    print("cuda matmul ok:", tuple(z.shape), z.dtype, z.device)
else:
    raise SystemExit("CUDA not available in this plain python environment.")
PY

echo "[3/3] dual LoRA model load + forward pass"
python - <<'PY'
import os
import torch
from transformers import AutoTokenizer

from models.dual_lora_model import setup_model_with_dual_lora
from utils.training.common import load_hf_token

model_id = os.environ["MODEL_ID"]
prompt_text = os.environ["PROMPT_TEXT"]
max_length = int(os.environ["MAX_LENGTH"])
rank = int(os.environ["RANK"])
alpha = int(os.environ["LORA_ALPHA"])
dropout = float(os.environ["LORA_DROPOUT"])
target_modules = [item.strip() for item in os.environ["LORA_TARGET_MODULES"].split(",") if item.strip()]

print("loading hf token...")
hf_token = load_hf_token()
print("loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("loading model with dual LoRA...")
model = setup_model_with_dual_lora(
    model_id=model_id,
    token=hf_token,
    rank=rank,
    alpha=alpha,
    dropout=dropout,
    target_modules=target_modules,
)
model.eval()

first_param = next(model.parameters())
input_device = first_param.device
print("first parameter device:", input_device)
print("model dtype:", model.dtype)
print("target module count:", len(model.dual_lora_adapter.target_module_names))

inputs = tokenizer(
    prompt_text,
    return_tensors="pt",
    truncation=True,
    max_length=max_length,
)
inputs = {k: v.to(input_device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

print("forward pass ok")
print("logits shape:", tuple(outputs.logits.shape))

if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        alloc_mb = torch.cuda.memory_allocated(idx) / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved(idx) / (1024 ** 2)
        print(
            f"cuda memory[{idx}]: allocated_mb={alloc_mb:.2f} reserved_mb={reserved_mb:.2f}"
        )
PY

echo
echo "[Done] GPU / model smoke test passed."
echo "Log file: ${RUN_STDOUT}"
