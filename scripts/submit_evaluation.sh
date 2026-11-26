#!/bin/bash
#SBATCH --job-name=phase0_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

cd /mnt/cv_data/users/seven/language-guided-medical-segmentation

MODEL=$1
DATASET=$2
SEED=$3

if [ "$MODEL" = "nnunet" ]; then
    PYTHON=/mnt/cv_data/users/seven/envs/nnunet/bin/python
else
    PYTHON=/mnt/cv_data/users/seven/envs/ml/bin/python
fi

"$PYTHON" -u src/evaluate_phase0.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --batch-size 4 \
    --workers 2