#!/bin/bash
#SBATCH --job-name=nnunet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /mnt/cv_data/users/seven/language-guided-medical-segmentation

source configs/nnunet_env.sh

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export nnUNet_compile=false
export nnUNet_n_proc_DA=2

DATASET="${1:?Please provide dataset: QaTa-COV19 or MosMedDataPlus}"
SEED="${2:?Please provide seed: 1001, 1002, or 1003}"
shift 2

/mnt/cv_data/users/seven/envs/nnunet/bin/python -u src/train_nnunet.py \
    --dataset "$DATASET" \
    --seed "$SEED" \
    "$@"