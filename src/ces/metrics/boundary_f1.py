
import numpy as np


def extract_boundary(mask):
    """
    Extract foreground boundary pixels using 4-connectivity.

    A foreground pixel is a boundary pixel if at least
    one of its four direct neighbors is background.

    Pixels outside the image are treated as background.
    """

    mask = np.asarray(mask).astype(bool)

    if mask.ndim != 2:
        raise ValueError("Expected a 2D binary mask.")

    height, width = mask.shape

    padded = np.pad(
        mask,
        ((1, 1), (1, 1)),
        mode="constant",
        constant_values=False
    )

    up = padded[0:height, 1:width + 1]
    down = padded[2:height + 2, 1:width + 1]
    left = padded[1:height + 1, 0:width]
    right = padded[1:height + 1, 2:width + 2]

    boundary = mask & (
        ~up | ~down | ~left | ~right
    )

    return boundary


def boundary_matches(boundary_a, boundary_b, tolerance=2):
    """
    Check whether every boundary pixel in A has at least
    one boundary pixel in B within the tolerance window.

    tolerance=2 corresponds to a 5x5 neighborhood.
    """

    height, width = boundary_a.shape

    padded = np.pad(
        boundary_b,
        ((tolerance, tolerance), (tolerance, tolerance)),
        mode="constant",
        constant_values=False
    )

    matched = np.zeros_like(
        boundary_a,
        dtype=bool
    )

    for dy in range(-tolerance, tolerance + 1):
        for dx in range(-tolerance, tolerance + 1):

            nearby = padded[
                tolerance + dy:tolerance + dy + height,
                tolerance + dx:tolerance + dx + width
            ]

            matched |= nearby

    return boundary_a & matched


def boundary_f1_score(
    ground_truth,
    prediction,
    tolerance=2
):
    """
    Calculate Boundary F1 for one image.

    Precision:
        Fraction of predicted boundary pixels matching
        a true boundary pixel.

    Recall:
        Fraction of true boundary pixels matched by
        a predicted boundary pixel.

    Both masks empty: BF1 = 1.
    Only one mask has boundaries: BF1 = 0.
    """

    ground_truth = np.asarray(ground_truth).astype(bool)
    prediction = np.asarray(prediction).astype(bool)

    if ground_truth.shape != prediction.shape:
        raise ValueError("Ground truth and prediction shapes differ.")

    true_boundary = extract_boundary(ground_truth)
    pred_boundary = extract_boundary(prediction)

    true_count = np.count_nonzero(true_boundary)
    pred_count = np.count_nonzero(pred_boundary)

    if true_count == 0 and pred_count == 0:
        return 1.0

    if true_count == 0 or pred_count == 0:
        return 0.0

    matched_pred = boundary_matches(
        pred_boundary,
        true_boundary,
        tolerance
    )

    matched_true = boundary_matches(
        true_boundary,
        pred_boundary,
        tolerance
    )

    precision = (
        np.count_nonzero(matched_pred) / pred_count
    )

    recall = (
        np.count_nonzero(matched_true) / true_count
    )

    denominator = precision + recall

    if denominator == 0:
        return 0.0

    return float(
        2.0 * precision * recall / denominator
    )
