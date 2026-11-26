#!/bin/bash
#SBATCH --job-name=unet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /mnt/cv_data/users/seven/language-guided-medical-segmentation

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

DATASET="${1:?Please provide dataset: QaTa-COV19 or MosMedDataPlus}"
SEED="${2:?Please provide seed: 1001, 1002, or 1003}"
shift 2

/mnt/cv_data/users/seven/envs/ml/bin/python -u src/train_unet.py \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --workers 2 \
    "$@"