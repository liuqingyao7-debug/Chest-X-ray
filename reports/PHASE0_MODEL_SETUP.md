# Phase 0.3: Model Setup

## Status

All three model setups passed GPU execution checks on NVIDIA GB10.
No main benchmark experiments have been completed by these checks.

## Model Summary

| Method | Source | Configuration | Input | Initialization |
|---|---|---|---|---|
| U-Net | LViT | Binary U-Net, one sigmoid output | 224 × 224 × 3 | Random |
| TransUNet | Official TransUNet | R50-ViT-B_16, two output classes | 224 × 224 × 3 | Official mirrored R50+ViT-B_16 checkpoint |
| nnU-Net | Official nnU-Net v2.8.1 | Planned PlainConvUNet, 2D, deep supervision | 224 × 224 × 3 | Random |

## Source Versions

- LViT: ba90775ead37d6f445f65ce83d1a0be64157f76f
- TransUNet: 02ef0010b36eb8328b5e689eadaf613602edf9b8
- nnU-Net: 7bb926124ce58a611fa53d3510c714726c800031

Repositories:
- https://github.com/HUANGLIZI/LViT
- https://github.com/Beckschen/TransUNet
- https://github.com/MIC-DKFZ/nnUNet

## Training Settings

| Setting | U-Net | TransUNet | nnU-Net 2D |
|---|---|---|---|
| Batch size | 2 | 24 | 65 |
| Maximum epochs | 2000 | 150 | 1000 |
| Optimizer | Adam | SGD | SGD with Nesterov |
| Initial learning rate | 0.0003 | 0.01 | 0.01 |
| Weight decay | 0 | 0.0001 | 0.00003 |
| Momentum | Not applicable | 0.9 | 0.99 |
| Loss | 0.5 WeightedDice + 0.5 WeightedBCE | 0.5 CE + 0.5 Dice | Native Dice + CE with deep supervision |
| Scheduler | Cosine warm restarts | Polynomial, power 0.9 | Native PolyLRScheduler |
| Initialization | Random | Pretrained encoder | Random |

U-Net uses early-stopping patience 50.
Its COVID19 learning rate follows the COVID19-specific source comment.

Seeds for the main experiments: 1001, 1002, 1003.

## Data and Augmentation

All methods use the same prepared images and binary target masks.

- Prepared data: `datasets/processed/`
- Resolution: 224 × 224
- Labels: background=0, lesion=1
- Original train/validation/test membership is preserved.
- U-Net and TransUNet loaders scale image intensities by 255.
- Their training augmentation uses matched image/mask rotations and flips.
- nnU-Net uses its planned per-channel Z-score normalization and native
  2D training augmentation.
- Validation is performed without random augmentation.
- No test performance was used to choose settings.

nnU-Net datasets:
- Dataset501_QaTa: 5716 training / 1429 validation cases
- Dataset502_MosMed: 2183 training / 273 validation cases

Both use the fixed split stored as fold 0.
No new random cross-validation split is generated.

## Checkpoint Selection

The benchmark policy is maximum mean per-image foreground validation
Dice at threshold 0.5. Both-empty prediction and target receive Dice 1.

For nnU-Net, the custom trainer evaluates every validation image,
restores cropped predictions to 224 × 224, and selects:

`checkpoint_best_fullval.safetensors`

The native EMA pseudo-Dice checkpoint may also be written, but it is
not the selected benchmark checkpoint.

Custom trainer:
`src/nnunet_custom/nnUNetTrainerPhase0FullVal.py`

Installed copy:
`external/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainerPhase0FullVal.py`

U-Net and TransUNet full-training entry points must enforce the same
validation policy when implemented for Task 0.4. Their smoke scripts
are execution checks, not full-training entry points.

## Verification

Both datasets passed:

- Image/mask batch loading.
- Model forward pass and expected output dimensions.
- Finite loss computation.
- Finite, nonzero gradients.
- Ten small-batch training updates.
- Configured batch-size execution.

Additional nnU-Net checks passed:

- Native preprocessed-data and augmentation loading.
- Deep-supervision outputs.
- Full validation over all 1429 QaTa and 273 MosMed validation cases.
- Checkpoint writing and readability of all stored tensors.

Full-validation checks used untrained networks. Their Dice values are
not benchmark performance results.

Checkpoint resume and automatic best-checkpoint replacement have not
yet been integration-tested. Configured-batch checks demonstrate short
execution, not a guarantee about full-run resource usage.

## Files

Configurations:
- `configs/unet.yaml`
- `configs/transunet.yaml`
- `configs/nnunet.yaml`

Checks and logs:
- `src/check_unet.py`
- `src/check_transunet.py`
- `src/check_nnunet.py`
- `reports/setup_checks/`

## Next Task

Implement and verify the full-training entry points, seed-specific run
directories, validation checkpoint selection, and restart handling
before launching the 18 experiments in Task 0.4.
