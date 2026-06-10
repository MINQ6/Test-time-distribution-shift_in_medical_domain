#!/bin/bash
#SBATCH --job-name=prep_medmcqa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
set -euo pipefail
cd /home/0630rb/yonsei/medical_ai/FedTTA
source /home/0630rb/miniconda3/etc/profile.d/conda.sh; conda activate py311_llm
export PYTHONPATH="$PWD/src"; export HF_HOME="${HF_HOME:-/scratch2/0630rb/huggingface}"
python src/data/prepare_medmcqa.py --n_train 600 --n_test 200
