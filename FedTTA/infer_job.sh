#!/bin/bash

#=========================================================================================
# SBATCH 옵션 설정
#=========================================================================================
#SBATCH --job-name=fedmoe_infer
#SBATCH --output=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=suma_pro6000
#SBATCH --qos=pro6000_qos
#SBATCH --gres=gpu:1
#SBATCH --mem=80G

#=========================================================================================
# 1. 작업 경로 및 환경 설정
#=========================================================================================
cd /home/0630rb/research_personalization
mkdir -p logs outputs

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
RUN_LOG_DIR="logs/${CURRENT_TIME}"
RUN_OUTPUT_DIR="outputs/${CURRENT_TIME}"
RUN_INFER_JSON_DIR="${RUN_OUTPUT_DIR}/inference_json"
mkdir -p "$RUN_LOG_DIR" "$RUN_OUTPUT_DIR" "$RUN_INFER_JSON_DIR"
SLURM_STDOUT="${RUN_LOG_DIR}/slurm_system_${SLURM_JOB_ID:-manual}_${CURRENT_TIME}.out"
exec > "$SLURM_STDOUT" 2>&1

finish_log() {
  EXIT_CODE=$?
  echo "=========================================="
  echo "Job finished on $(date)"
  echo "Exit Code: ${EXIT_CODE}"
  echo "=========================================="
}
trap finish_log EXIT

source /home/0630rb/miniconda3/etc/profile.d/conda.sh
conda activate py311_blackwell
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

#=========================================================================================
# 2. 실험 설정
#=========================================================================================
MODEL_ID="meta-llama/Meta-Llama-3-8B"
DATASET_NAME="dataset1"
PERSONALIZED_NUM_SAMPLES=200
TTP_NUM_SAMPLES=""
INFERENCE_BATCH_SIZE=32
LORA_CHECKPOINT_PREFIX="lora_adapter"
SUMMARY_JSON="${RUN_OUTPUT_DIR}/inference_summary_${CURRENT_TIME}.json"
SUMMARY_MD="${RUN_OUTPUT_DIR}/inference_summary_${CURRENT_TIME}.md"

#=========================================================================================
# 3. 정보 출력
#=========================================================================================
echo "=========================================="
echo "Allocated Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"
echo "Allocated GPUs: $SLURM_JOB_GPUS"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Model ID: $MODEL_ID"
echo "Dataset: $DATASET_NAME"
echo "Personalized Eval Samples: $PERSONALIZED_NUM_SAMPLES"
if [ -z "$TTP_NUM_SAMPLES" ]; then
  echo "TTP Eval Samples: full test file"
else
  echo "TTP Eval Samples: $TTP_NUM_SAMPLES"
fi
echo "Inference Batch Size: $INFERENCE_BATCH_SIZE"
echo "Compare Methods: DP-LoRA vs Ours"
echo "Result JSON Dir: $RUN_INFER_JSON_DIR"
echo "Summary JSON: $SUMMARY_JSON"
echo "Summary MD: $SUMMARY_MD"
echo "=========================================="

echo "[Debug] nvidia-smi 확인"
nvidia-smi

echo "[Debug] PyTorch CUDA 환경 확인"
python - <<'PY2'
import torch
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu:", p.name, "cc", f"{p.major}.{p.minor}")
PY2

#=========================================================================================
# 4. Inference / 성능 비교
#=========================================================================================
echo "[Phase 1] Inference start for all clients..."
RESULT_JSON_FILES=()
for CLIENT_ID in client_1 client_2 client_3 client_4 client_5 client_6 client_7 client_8
do
  echo "  -> Running inference for ${CLIENT_ID}"

  INFER_JSON="${RUN_INFER_JSON_DIR}/${CLIENT_ID}_${CURRENT_TIME}.json"
  INFER_STDERR="${RUN_LOG_DIR}/inference_${CLIENT_ID}_${CURRENT_TIME}.err"

  INFER_CMD=(
    python src/inference.py
    --client "$CLIENT_ID"
    --model_id "$MODEL_ID"
    --dataset_name "$DATASET_NAME"
    --num_samples "$PERSONALIZED_NUM_SAMPLES"
    --compare_lora
    --disable_local_finetuned
    --result_json "$INFER_JSON"
    --inference_batch_size "$INFERENCE_BATCH_SIZE"
    --lora_checkpoint_prefix "$LORA_CHECKPOINT_PREFIX"
  )

  if [ -n "$TTP_NUM_SAMPLES" ]; then
    INFER_CMD+=(--ttp_num_samples "$TTP_NUM_SAMPLES")
  fi

  "${INFER_CMD[@]}" 2> "$INFER_STDERR"

  INFER_EXIT_CODE=$?
  if [ $INFER_EXIT_CODE -ne 0 ]; then
    echo "[Phase 1] Inference failed for ${CLIENT_ID} with exit code $INFER_EXIT_CODE"
    echo "  stderr: $INFER_STDERR"
    exit $INFER_EXIT_CODE
  fi

  if [ ! -f "$INFER_JSON" ]; then
    echo "[Phase 1] Missing result json for ${CLIENT_ID}: $INFER_JSON"
    exit 1
  fi

  RESULT_JSON_FILES+=("$INFER_JSON")
done

#=========================================================================================
# 5. Inference 종합 표 생성
#=========================================================================================
echo "[Phase 2] Building merged inference summary..."
python src/utils/inference/summary.py   --dataset_name "$DATASET_NAME"   --summary_json "$SUMMARY_JSON"   --summary_md "$SUMMARY_MD"   "${RESULT_JSON_FILES[@]}"

SUMMARY_EXIT_CODE=$?
if [ $SUMMARY_EXIT_CODE -ne 0 ]; then
  echo "[Phase 2] Summary build failed with exit code $SUMMARY_EXIT_CODE"
  exit $SUMMARY_EXIT_CODE
fi

echo "[Phase 2] Inference summary generated successfully."
