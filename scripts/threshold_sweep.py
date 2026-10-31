
import os
import sys
import json
from pathlib import Path

# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# IMPORTS
# ============================================================

import cv2
import torch
import numpy as np

from ces.config import (
    PROCESSED_DATA_DIR,
    FINAL_MODEL_PATH,
    MASK_THRESHOLD,
    BOUNDARY_TOLERANCE
)

from ces.pipeline import (
    load_model,
    predict_mask,
    configure_inference
)

from ces.metrics.evaluation import SegmentationEvaluator

from ces.metrics.ces_score import calculate_quality_score


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Configure single-threaded inference
    # --------------------------------------------------------

    configure_inference()
    torch.set_num_threads(1)

    # --------------------------------------------------------
    # 2. Locate validation dataset
    # --------------------------------------------------------

    image_dir = PROCESSED_DATA_DIR / "val" / "images"
    mask_dir = PROCESSED_DATA_DIR / "val" / "masks"

    image_paths = sorted(
        image_dir.glob("*.png")
    )

    if not image_paths:
        raise RuntimeError(
            "No validation images found."
        )

    print("=" * 75)
    print("SWIFTMASK - VALIDATION THRESHOLD EXPERIMENT")
    print("=" * 75)

    print("Validation images:", len(image_paths))
    print("Thresholds:", THRESHOLDS)

    # --------------------------------------------------------
    # 3. Load trained model
    # --------------------------------------------------------

    model = load_model(
        FINAL_MODEL_PATH,
        device="cpu"
    )

    model.eval()

    print("Model loaded successfully.")
    print("Device: CPU")
    print("CPU threads:", torch.get_num_threads())

    # --------------------------------------------------------
    # 4. Warmup
    # --------------------------------------------------------

    first_image = cv2.imread(
        str(image_paths[0]),
        cv2.IMREAD_COLOR
    )

    if first_image is None:
        raise RuntimeError(
            "Could not load warmup image."
        )

    # Warmup is excluded from evaluation.
    predict_mask(
        first_image,
        model,
        threshold=0.50
    )

    # --------------------------------------------------------
    # 5. Create separate evaluators
    # --------------------------------------------------------

    evaluators = {
        threshold: SegmentationEvaluator(
            boundary_tolerance=BOUNDARY_TOLERANCE
        )
        for threshold in THRESHOLDS
    }

    # --------------------------------------------------------
    # 6. Evaluate all thresholds
    # --------------------------------------------------------

    print("\nEvaluating thresholds...\n")

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        mask_path = mask_dir / image_path.name

        # Read image and ground truth.
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        ground_truth = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        if ground_truth is None:
            raise RuntimeError(
                f"Could not read mask: {mask_path}"
            )

        # Convert ground truth into binary mask.
        ground_truth = (
            ground_truth >= MASK_THRESHOLD
        ).astype(np.uint8)

        # Evaluate every threshold on the same image.
        for threshold in THRESHOLDS:

            prediction = predict_mask(
                image,
                model,
                threshold=threshold
            )

            evaluators[threshold].update(
                ground_truth,
                prediction
            )

        # Progress information.
        if index % 100 == 0 or index == len(image_paths):

            print(
                f"Processed {index}/{len(image_paths)}"
            )

    # --------------------------------------------------------
    # 7. Calculate metrics for each threshold
    # --------------------------------------------------------

    results = []

    print("\n" + "=" * 85)

    print(
        f"{'Threshold':<14}"
        f"{'Object IoU':<14}"
        f"{'Background IoU':<18}"
        f"{'mIoU':<14}"
        f"{'Boundary F1':<16}"
        f"{'Q':<12}"
    )

    print("=" * 85)

    for threshold in THRESHOLDS:

        metrics = evaluators[threshold].compute()

        # Calculate assignment quality score.
        quality_score = calculate_quality_score(
            metrics["miou"],
            metrics["boundary_f1"]
        )

        result = {
            "threshold": threshold,
            "object_iou": metrics["object_iou"],
            "background_iou": metrics["background_iou"],
            "miou": metrics["miou"],
            "boundary_f1": metrics["boundary_f1"],
            "quality_score": quality_score,
            "confusion_matrix": metrics["confusion_matrix"]
        }

        results.append(result)

        print(
            f"{threshold:<14.2f}"
            f"{metrics['object_iou']:<14.6f}"
            f"{metrics['background_iou']:<18.6f}"
            f"{metrics['miou']:<14.6f}"
            f"{metrics['boundary_f1']:<16.6f}"
            f"{quality_score:<12.6f}"
        )

    print("=" * 85)

    # --------------------------------------------------------
    # 8. Select best threshold using validation Q
    # --------------------------------------------------------

    best = max(
        results,
        key=lambda item: item["quality_score"]
    )

    # --------------------------------------------------------
    # 9. Save experiment results
    # --------------------------------------------------------

    output_dir = ROOT_DIR / "outputs" / "logs"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir / "threshold_sweep.json"
    )

    experiment_results = {
        "experiment": "validation_threshold_sweep",
        "num_images": len(image_paths),
        "thresholds": THRESHOLDS,
        "results": results,
        "best_threshold": best["threshold"],
        "best_quality_score": best["quality_score"]
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            experiment_results,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # 10. Display best result
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("BEST VALIDATION THRESHOLD")
    print("=" * 75)

    print(
        f"Threshold:     {best['threshold']:.2f}"
    )

    print(
        f"Object IoU:    {best['object_iou']:.6f}"
    )

    print(
        f"Background IoU:{best['background_iou']:.6f}"
    )

    print(
        f"mIoU:          {best['miou']:.6f}"
    )

    print(
        f"Boundary F1:   {best['boundary_f1']:.6f}"
    )

    print(
        f"Quality Q:     {best['quality_score']:.6f}"
    )

    print("\nResults saved to:")
    print(output_path)

    print("=" * 75)


if __name__ == "__main__":
    main()
