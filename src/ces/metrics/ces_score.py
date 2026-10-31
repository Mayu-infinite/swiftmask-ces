
from ces.config import (
    QUALITY_IOU_WEIGHT,
    QUALITY_BOUNDARY_WEIGHT,
    MAX_CPU_TIME_MS,
    MAX_MODEL_SIZE_BYTES
)


def calculate_quality_score(miou, boundary_f1):
    """
    Q = 0.70 * mIoU + 0.30 * BoundaryF1
    """

    return (
        QUALITY_IOU_WEIGHT * miou
        + QUALITY_BOUNDARY_WEIGHT * boundary_f1
    )


def calculate_ces_score(
    miou,
    boundary_f1,
    cpu_time_ms,
    model_size_bytes
):
    """
    Calculate Compute-Efficient Segmentation score.

    CES = Q * [
        1 - 0.20 * (TCPU / 50)
          - 0.10 * min(1, Sdisk / 4MiB)
    ]

    If TCPU exceeds 50 ms, CES is zero.
    """

    quality = calculate_quality_score(
        miou,
        boundary_f1
    )

    if cpu_time_ms > MAX_CPU_TIME_MS:
        return {
            "quality_score": float(quality),
            "ces_score": 0.0
        }

    time_penalty = (
        0.20 * cpu_time_ms / MAX_CPU_TIME_MS
    )

    size_penalty = (
        0.10 * min(
            1.0,
            model_size_bytes / MAX_MODEL_SIZE_BYTES
        )
    )

    ces = quality * (
        1.0 - time_penalty - size_penalty
    )

    return {
        "quality_score": float(quality),
        "ces_score": float(ces)
    }
