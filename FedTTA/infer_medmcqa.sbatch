#!/bin/bash
#SBATCH --job-name=infer_medmcqa
#SBATCH --partition=suma_rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
# =========================================================================
# MedMCQA FedDPA-F entropy-min TTA 추론 (letter-likelihood).
# 학습으로 저장된 src/checkpoints/dual_lora_adapter_client_{1..4}.pth 사용.
# method: local/global/equal/entropy_min(4letter)/entropy_min(fullvocab)
# scope:  own(자기 subject) + cross(4 subject 통합 800)
#
# 제출:  sbatch infer_medmcqa.sbatch
#       CHECKPOINT_DIR=src/checkpoints/<run> sbatch infer_medmcqa.sbatch
# =========================================================================
set -euo pipefail
PROJECT_ROOT="/home/0630rb/yonsei/medical_ai/FedTTA"
cd "$PROJECT_ROOT"
mkdir -p logs outputs

# --- per-run 로그 폴더: logs/<timestamp>_<jobname>/ (outputs 폴더와 동일 컨벤션) ---
# 실험 stdout/stderr 를 이 폴더 안 run.out/run.err 로도 기록(슬럼 로그와 병행).
# RUN_TS 는 로그/산출물 폴더 이름 공유용일 뿐, 체크포인트 경로(RUN_TIMESTAMP)와는 무관.
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${SLURM_JOB_NAME:-infer_medmcqa}"
# CLIENT 지정 시 그 client만 평가(client별 4잡 분할용). CLIENT_SUFFIX로 로그/출력 구분.
CLIENT="${CLIENT:-}"
CLIENT_ARG=""; CLIENT_SUFFIX=""
[ -n "$CLIENT" ] && { CLIENT_ARG="--clients $CLIENT"; CLIENT_SUFFIX="_${CLIENT}"; }
# job id 접미사로 동시 제출(같은 초) 시 폴더명 충돌/덮어쓰기 방지
RUN_ID="${RUN_TS}_${RUN_TAG}${CLIENT_SUFFIX}${SLURM_JOB_ID:+_${SLURM_JOB_ID}}"
RUN_LOG_DIR="logs/${RUN_ID}"
# OUTPUT_DIR 외부 지정 가능(launcher가 그룹 폴더의 client 하위로 모음). 없으면 기존 컨벤션.
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RUN_ID}}"
mkdir -p "$RUN_LOG_DIR"
exec > >(tee -a "$RUN_LOG_DIR/run.out") 2> >(tee -a "$RUN_LOG_DIR/run.err" >&2)
echo "[logs] per-run log dir = $RUN_LOG_DIR"

source /home/0630rb/miniconda3/etc/profile.d/conda.sh
conda activate py311_llm
export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}/src"
export HF_HOME="${HF_HOME:-/scratch2/0630rb/huggingface}"
export TOKENIZERS_PARALLELISM=false

MODEL_ID="${MODEL_ID:-/scratch2/0630rb/models/Qwen3-1.7B-Base}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-src/checkpoints}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
CROSS_NUM_SAMPLES="${CROSS_NUM_SAMPLES:-}"   # 비우면 전체(800). 진단 땐 200 등으로 축소
CROSS_ARG=""
[ -n "$CROSS_NUM_SAMPLES" ] && CROSS_ARG="--cross_num_samples $CROSS_NUM_SAMPLES"
ENTROPY_STEPS="${ENTROPY_STEPS:-20}"
ENTROPY_LR="${ENTROPY_LR:-0.1}"
ENTROPY_TEMPERATURE="${ENTROPY_TEMPERATURE:-1.0}"   # T>1 → entropy-min 분포 평탄화(진단용). 1=기존

echo "=========================================="
echo "MedMCQA inference | $(date) | host=$(hostname)"
echo "model=$MODEL_ID checkpoints=$CHECKPOINT_DIR T=$ENTROPY_TEMPERATURE"
echo "=========================================="
nvidia-smi || true

python src/inference_medmcqa.py \
  --model_id "$MODEL_ID" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --num_samples "$NUM_SAMPLES" \
  $CLIENT_ARG \
  $CROSS_ARG \
  --entropy_steps "$ENTROPY_STEPS" \
  --entropy_lr "$ENTROPY_LR" \
  --entropy_temperature "$ENTROPY_TEMPERATURE" \
  --output_dir "$OUTPUT_DIR"

echo "Done | $(date)"
