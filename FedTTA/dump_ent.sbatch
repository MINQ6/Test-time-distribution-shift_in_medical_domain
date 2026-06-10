#!/bin/bash
#SBATCH --job-name=dump_ent
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
set -euo pipefail
cd /home/0630rb/yonsei/medical_ai/FedTTA
source /home/0630rb/miniconda3/etc/profile.d/conda.sh; conda activate py311_llm
export PYTHONPATH="$PWD/src"; export HF_HOME="${HF_HOME:-/scratch2/0630rb/huggingface}"; export TOKENIZERS_PARALLELISM=false
python src/dump_entropy_examples.py --checkpoint_dir src/checkpoints/exp7_f_bal_r8qv --client "${CLIENT:-client_1}"
