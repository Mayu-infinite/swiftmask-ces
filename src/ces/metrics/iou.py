
import numpy as np


def update_confusion_matrix(total, y_true, y_pred):
    """
    Update a binary confusion matrix.

    Matrix layout:
        [[TN, FP],
         [FN, TP]]

    Accumulates counts across images.
    """

    y_true = np.asarray(y_true).astype(bool).ravel()
    y_pred = np.asarray(y_pred).astype(bool).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError("Ground truth and prediction shapes differ.")

    tn = np.count_nonzero(~y_true & ~y_pred)
    fp = np.count_nonzero(~y_true & y_pred)
    fn = np.count_nonzero(y_true & ~y_pred)
    tp = np.count_nonzero(y_true & y_pred)

    total[0, 0] += tn
    total[0, 1] += fp
    total[1, 0] += fn
    total[1, 1] += tp

    return total


def confusion_matrix_binary(y_true, y_pred):
    """
    Calculate the confusion matrix for one binary mask.
    """

    total = np.zeros((2, 2), dtype=np.int64)

    return update_confusion_matrix(
        total,
        y_true,
        y_pred
    )


def compute_iou(confusion):
    """
    Calculate foreground IoU, background IoU and mIoU.
    """

    tn, fp = confusion[0]
    fn, tp = confusion[1]

    object_union = tp + fp + fn
    background_union = tn + fp + fn

    object_iou = (
        tp / object_union
        if object_union > 0 else 0.0
    )

    background_iou = (
        tn / background_union
        if background_union > 0 else 0.0
    )

    mean_iou = (
        object_iou + background_iou
    ) / 2.0

    return {
        "object_iou": float(object_iou),
        "background_iou": float(background_iou),
        "miou": float(mean_iou)
    }
