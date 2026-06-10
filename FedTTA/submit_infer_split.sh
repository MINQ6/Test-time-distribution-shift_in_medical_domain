#!/bin/bash
# client별 4잡으로 쪼개 추론 제출 (각 1 GPU 병렬) → 끝나면 merge job이 합침.
#
# 사용 예:
#   SBATCH=infer_cloze.sbatch CHECKPOINT_DIR=src/checkpoints/exp3_4b_cloze TAG=cloze_exp3 bash submit_infer_split.sh
#   SBATCH=infer_medmcqa.sbatch CHECKPOINT_DIR=src/checkpoints/exp2_4b_600 TAG=letter_exp2 ENTROPY_TEMPERATURE=1.0 bash submit_infer_split.sh
#
# 넘길 수 있는 env(=각 sbatch가 읽음): MODEL_ID, CHECKPOINT_DIR, NUM_SAMPLES,
#   CROSS_NUM_SAMPLES(비우면 full 800), ENTROPY_TEMPERATURE, ENTROPY_STEPS, ENTROPY_LR
set -euo pipefail
cd /home/0630rb/yonsei/medical_ai/FedTTA

SBATCH="${SBATCH:-infer_medmcqa.sbatch}"
TAG="${TAG:-split}"
DEP="${DEP:-}"                 # 예: afterok:1609170 (학습 끝나면 추론 시작)
EXTRA="${EXTRA:-}"            # 예: --partition=... --qos=... --exclude=cs-gpu-01
TS="$(date +%Y%m%d_%H%M%S)"
GROUP="outputs/${TS}_${TAG}"
mkdir -p "$GROUP"
echo "[split] group dir = $GROUP | sbatch=$SBATCH | dep=${DEP:-none} | extra=${EXTRA:-none}"

jids=()
for i in 1 2 3 4; do
  J=$(CLIENT="client_${i}" OUTPUT_DIR="${GROUP}/client_${i}" \
      sbatch --parsable ${DEP:+--dependency=$DEP} ${EXTRA} "$SBATCH")
  echo "  client_${i} -> job $J"
  jids+=("$J")
done
dep=$(IFS=:; echo "${jids[*]}")

# 4잡 모두 정상 종료 후 병합 (GPU 불필요한 CPU 잡)
MJ=$(sbatch --parsable --job-name=merge --partition=base_suma_rtx3090,dell_rtx3090,suma_rtx4090 \
     --exclude=cs-gpu-01 --gres=gpu:1 \
     --cpus-per-task=1 --mem=4G --time=00:10:00 \
     --dependency="afterok:${dep}" \
     --output="${GROUP}/merge_%j.out" --error="${GROUP}/merge_%j.err" \
     --wrap="cd /home/0630rb/yonsei/medical_ai/FedTTA; source /home/0630rb/miniconda3/etc/profile.d/conda.sh; conda activate py311_llm; export PYTHONPATH=\$PYTHONPATH:\$(pwd)/src; python src/merge_summaries.py ${GROUP}")
echo "[split] merge job $MJ (afterok:${dep})"
echo "[split] 최종 결과: ${GROUP}/summary.json"
echo "$GROUP" > /tmp/last_split_group.txt
