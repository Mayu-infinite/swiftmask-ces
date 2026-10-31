
"""
Evaluate SwiftMask on manually captured photographs.

Outputs:
    outputs/captured_evaluation/
        predictions/       Binary predicted masks (0/255)
        overlays/          Predicted foreground over original image
        comparisons/        2x2 visual comparison images
        captured_results.json
        per_image_results.csv

This is a separate evaluation. It does not modify the model or checkpoint.
"""

# ============================================================
# 1. ENVIRONMENT CONFIGURATION
# ============================================================

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# ============================================================
# 2. IMPORTS AND PROJECT PATHS
# ============================================================

import sys
import csv
import json
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(1, str(ROOT_DIR))

import cv2
import numpy as np

from ces.config import (
    IMAGE_SIZE,
    MASK_THRESHOLD,
    FINAL_MODEL_PATH,
    BOUNDARY_TOLERANCE,
    CPU_THREADS,
)

from ces.pipeline import (
    configure_inference,
    load_model,
    predict_mask,
)

from ces.metrics.evaluation import SegmentationEvaluator
from ces.metrics.boundary_f1 import extract_boundary
from ces.metrics.ces_score import calculate_ces_score


# ============================================================
# 3. DIRECTORIES
# ============================================================

IMAGE_DIR = ROOT_DIR / "captured" / "images 256"
MASK_DIR = ROOT_DIR / "captured" / "masks 256"

OUTPUT_DIR = ROOT_DIR / "outputs" / "captured_evaluation"

PREDICTION_DIR = OUTPUT_DIR / "predictions"
OVERLAY_DIR = OUTPUT_DIR / "overlays"
COMPARISON_DIR = OUTPUT_DIR / "comparisons"

JSON_PATH = OUTPUT_DIR / "captured_results.json"
CSV_PATH = OUTPUT_DIR / "per_image_results.csv"


# ============================================================
# 4. IMAGE PAIR DISCOVERY
# ============================================================

def get_image_pairs():
    """Find matching captured images and ground-truth masks."""

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Captured image directory not found: {IMAGE_DIR}"
        )

    if not MASK_DIR.is_dir():
        raise FileNotFoundError(
            f"Captured mask directory not found: {MASK_DIR}"
        )

    image_files = sorted(
        IMAGE_DIR.glob("*.png"),
        key=lambda path: path.name
    )

    mask_files = sorted(
        MASK_DIR.glob("*.png"),
        key=lambda path: path.name
    )

    image_names = {path.name for path in image_files}
    mask_names = {path.name for path in mask_files}

    missing_masks = sorted(image_names - mask_names)
    missing_images = sorted(mask_names - image_names)

    if missing_masks or missing_images:
        raise RuntimeError(
            f"Image-mask mismatch.\n"
            f"Missing masks: {missing_masks}\n"
            f"Missing images: {missing_images}"
        )

    if not image_files:
        raise RuntimeError("No captured PNG images found.")

    return [
        (image_path, MASK_DIR / image_path.name)
        for image_path in image_files
    ]


# ============================================================
# 5. READ AND VALIDATE IMAGE PAIR
# ============================================================

def read_pair(image_path, mask_path):
    """Read a 256x256 BGR image and binary ground-truth mask."""

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    if mask is None:
        raise RuntimeError(
            f"Could not read mask: {mask_path}"
        )

    expected_image_shape = (
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    )

    expected_mask_shape = (
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    if image.shape != expected_image_shape:
        raise ValueError(
            f"Invalid image shape for {image_path.name}: "
            f"{image.shape}"
        )

    if mask.shape != expected_mask_shape:
        raise ValueError(
            f"Invalid mask shape for {mask_path.name}: "
            f"{mask.shape}"
        )

    ground_truth = (
        mask >= MASK_THRESHOLD
    ).astype(np.uint8)

    return image, ground_truth


# ============================================================
# 6. VISUALIZATION HELPERS
# ============================================================

def create_overlay(image, ground_truth, prediction):
    """
    Overlay predicted foreground in green.
    Mark ground-truth boundary in red.
    """

    overlay = image.copy()

    foreground = prediction.astype(bool)

    green = np.zeros_like(image)
    green[:, :] = (0, 255, 0)

    overlay[foreground] = cv2.addWeighted(
        image[foreground],
        0.45,
        green[foreground],
        0.55,
        0
    )

    # Ground-truth boundary is shown in red.
    gt_boundary = extract_boundary(
        ground_truth.astype(bool)
    )

    overlay[gt_boundary] = (0, 0, 255)

    return overlay


def add_label(image, label):
    """Add a readable label to a visualization panel."""

    output = image.copy()

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 28),
        (35, 35, 35),
        thickness=-1
    )

    cv2.putText(
        output,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return output


def create_comparison(
    image,
    ground_truth,
    prediction,
    overlay
):
    """Create a 2x2 comparison panel."""

    gt_visual = cv2.cvtColor(
        ground_truth * 255,
        cv2.COLOR_GRAY2BGR
    )

    pred_visual = cv2.cvtColor(
        prediction * 255,
        cv2.COLOR_GRAY2BGR
    )

    panels = [
        add_label(image, "Original Image"),
        add_label(gt_visual, "Ground Truth"),
        add_label(pred_visual, "Prediction"),
        add_label(overlay, "Overlay: Green=Prediction, Red=GT Boundary"),
    ]

    top = cv2.hconcat(panels[:2])
    bottom = cv2.hconcat(panels[2:])

    return cv2.vconcat([top, bottom])


# ============================================================
# 7. EVALUATION
# ============================================================

def evaluate_captured():

    print("=" * 70)
    print("SWIFTMASK - CAPTURED PHOTO EVALUATION")
    print("=" * 70)

    # Create output directories.
    for directory in (
        PREDICTION_DIR,
        OVERLAY_DIR,
        COMPARISON_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    # Configure CPU inference.
    configure_inference()

    print(f"Checkpoint: {FINAL_MODEL_PATH}")
    print(f"CPU threads: {CPU_THREADS}")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Mask directory: {MASK_DIR}")

    # Discover matching pairs.
    pairs = get_image_pairs()

    print(f"Captured images found: {len(pairs)}")

    # Load final trained checkpoint.
    model = load_model(
        model_path=FINAL_MODEL_PATH,
        device="cpu"
    )

    # Aggregate evaluator.
    aggregate_evaluator = SegmentationEvaluator(
        boundary_tolerance=BOUNDARY_TOLERANCE
    )

    per_image_results = []

    print("\nPerforming warmup inference...")

    first_image, _ = read_pair(*pairs[0])
    _ = predict_mask(first_image, model)

    print("Warmup completed.")
    print("\nStarting captured-photo evaluation...\n")

    total_cpu_seconds = 0.0

    # --------------------------------------------------------
    # Process every captured image.
    # --------------------------------------------------------

    for index, (image_path, mask_path) in enumerate(
        pairs,
        start=1
    ):

        image, ground_truth = read_pair(
            image_path,
            mask_path
        )

        # Time prediction only. File I/O and visualization
        # are excluded.
        start_time = time.process_time()

        prediction = predict_mask(
            image,
            model
        )

        end_time = time.process_time()

        inference_seconds = end_time - start_time
        inference_ms = inference_seconds * 1000.0

        total_cpu_seconds += inference_seconds

        # Validate prediction.
        if prediction.shape != ground_truth.shape:
            raise RuntimeError(
                f"Prediction shape mismatch: {image_path.name}"
            )

        if not np.isin(
            prediction,
            [0, 1]
        ).all():
            raise RuntimeError(
                f"Prediction is not binary: {image_path.name}"
            )

        # Update aggregate metrics.
        aggregate_evaluator.update(
            ground_truth,
            prediction
        )

        # Calculate individual image metrics.
        image_evaluator = SegmentationEvaluator(
            boundary_tolerance=BOUNDARY_TOLERANCE
        )

        image_evaluator.update(
            ground_truth,
            prediction
        )

        image_metrics = image_evaluator.compute()

        result = {
            "image": image_path.name,
            "object_iou": image_metrics["object_iou"],
            "background_iou": image_metrics["background_iou"],
            "miou": image_metrics["miou"],
            "boundary_f1": image_metrics["boundary_f1"],
            "cpu_time_ms": inference_ms,
        }

        per_image_results.append(result)

        # ----------------------------------------------------
        # Save predicted mask.
        # ----------------------------------------------------

        prediction_path = (
            PREDICTION_DIR / image_path.name
        )

        cv2.imwrite(
            str(prediction_path),
            prediction * 255
        )

        # ----------------------------------------------------
        # Save overlay.
        # ----------------------------------------------------

        overlay = create_overlay(
            image,
            ground_truth,
            prediction
        )

        overlay_path = (
            OVERLAY_DIR / image_path.name
        )

        cv2.imwrite(
            str(overlay_path),
            overlay
        )

        # ----------------------------------------------------
        # Save 2x2 comparison.
        # ----------------------------------------------------

        comparison = create_comparison(
            image,
            ground_truth,
            prediction,
            overlay
        )

        comparison_path = (
            COMPARISON_DIR / image_path.name
        )

        cv2.imwrite(
            str(comparison_path),
            comparison
        )

        print(
            f"[{index:02d}/{len(pairs):02d}] "
            f"{image_path.name:12s} | "
            f"mIoU: {image_metrics['miou']:.4f} | "
            f"BF1: {image_metrics['boundary_f1']:.4f} | "
            f"CPU: {inference_ms:.2f} ms"
        )

    # ========================================================
    # 8. AGGREGATE METRICS
    # ========================================================

    aggregate = aggregate_evaluator.compute()

    average_cpu_ms = (
        total_cpu_seconds / len(pairs)
    ) * 1000.0

    model_size_bytes = os.path.getsize(
        FINAL_MODEL_PATH
    )

    quality_q = (
        0.70 * aggregate["miou"]
        + 0.30 * aggregate["boundary_f1"]
    )

    # Captured-photo CES is reported separately from DUTS-TE.
    ces_result = calculate_ces_score(
        miou=aggregate["miou"],
        boundary_f1=aggregate["boundary_f1"],
        cpu_time_ms=average_cpu_ms,
        model_size_bytes=model_size_bytes,
    )

    # ========================================================
    # 9. SAVE JSON RESULTS
    # ========================================================

    final_results = {
        "dataset": "Captured Phone Photographs",
        "num_images": aggregate["num_images"],
        "checkpoint": str(FINAL_MODEL_PATH),
        "device": "CPU",
        "cpu_threads": CPU_THREADS,
        "average_cpu_time_ms": average_cpu_ms,
        "model_size_bytes": model_size_bytes,
        "object_iou": aggregate["object_iou"],
        "background_iou": aggregate["background_iou"],
        "miou": aggregate["miou"],
        "boundary_f1": aggregate["boundary_f1"],
        "quality_q": quality_q,
        "ces": ces_result["ces_score"],
        "ces_quality_score": ces_result["quality_score"],
        "confusion_matrix": aggregate["confusion_matrix"],
        "per_image_results": per_image_results,
    }

    with open(
        JSON_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            final_results,
            file,
            indent=4
        )

    # ========================================================
    # 10. SAVE PER-IMAGE CSV
    # ========================================================

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "image",
            "object_iou",
            "background_iou",
            "miou",
            "boundary_f1",
            "cpu_time_ms",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(per_image_results)

    # ========================================================
    # 11. PRINT FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL CAPTURED PHOTO RESULTS")
    print("=" * 70)

    print(f"Images evaluated:  {aggregate['num_images']}")
    print(f"Object IoU:        {aggregate['object_iou']:.6f}")
    print(f"Background IoU:    {aggregate['background_iou']:.6f}")
    print(f"mIoU:              {aggregate['miou']:.6f}")
    print(f"Boundary F1:       {aggregate['boundary_f1']:.6f}")
    print(f"Quality Q:         {quality_q:.6f}")
    print(f"Average CPU time:  {average_cpu_ms:.4f} ms/image")
    print(f"Model size:        {model_size_bytes} bytes")
    print(f"CES:               {ces_result['ces_score']:.6f}")

    print("-" * 70)
    print("Confusion matrix:")
    print(np.array(aggregate["confusion_matrix"]))

    print("-" * 70)
    print(f"JSON results:      {JSON_PATH}")
    print(f"CSV results:       {CSV_PATH}")
    print(f"Predictions:       {PREDICTION_DIR}")
    print(f"Overlays:          {OVERLAY_DIR}")
    print(f"Comparisons:       {COMPARISON_DIR}")

    print("=" * 70)

    return final_results


# ============================================================
# 12. MAIN
# ============================================================

if __name__ == "__main__":
    evaluate_captured()
