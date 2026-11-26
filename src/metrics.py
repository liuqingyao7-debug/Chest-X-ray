import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def compute_metrics(pred, target):
    """
    Compute binary segmentation metrics.

    Metrics:
        Dice
        Foreground IoU
        HD95 (pixels)
        ASSD (pixels)
    """

    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)

    if pred.shape != (224, 224) or target.shape != (224, 224):
        raise ValueError("Both masks must have shape (224, 224)")

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    pred_sum = pred.sum()
    target_sum = target.sum()

    # Dice
    denominator = pred_sum + target_sum
    dice = 2.0 * intersection / denominator if denominator else 1.0

    # Foreground IoU
    iou = intersection / union if union else 1.0

    # Both masks are empty
    if pred_sum == 0 and target_sum == 0:
        hd95 = 0.0
        assd = 0.0

    # Only one mask is empty
    elif pred_sum == 0 or target_sum == 0:
        hd95 = float(np.hypot(223, 223))
        assd = hd95

    # Both masks are non-empty
    else:
        # Extract mask boundaries
        pred_surface = pred ^ binary_erosion(pred)
        target_surface = target ^ binary_erosion(target)

        # Compute surface distances
        distance_to_target = distance_transform_edt(~target_surface)
        distance_to_pred = distance_transform_edt(~pred_surface)

        distances = np.concatenate([
            distance_to_target[pred_surface],
            distance_to_pred[target_surface]
        ])

        # 95th percentile Hausdorff Distance
        hd95 = float(np.percentile(distances, 95))

        # Average Symmetric Surface Distance
        assd = float(np.mean(distances))

    return {
        "dice": float(dice),
        "miou": float(iou),
        "hd95": hd95,
        "assd": assd
    }