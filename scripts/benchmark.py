
import os
import sys
import json
import time
import argparse
from pathlib import Path

# ============================================================
# 1. ENVIRONMENT CONFIGURATION
# ============================================================

# Set thread limits before importing numerical libraries.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# ============================================================
# 2. PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ============================================================
# 3. IMPORTS
# ============================================================

import cv2
import numpy as np
import torch

from ces.config import (
    PROCESSED_DATA_DIR,
    IMAGE_SIZE,
    MASK_THRESHOLD,
    FINAL_MODEL_PATH,
    BOUNDARY_TOLERANCE,
    CPU_THREADS,
)

from ces.pipeline import (
    load_model,
    predict_mask,
    configure_inference,
)

from ces.metrics.evaluation import SegmentationEvaluator

from ces.metrics.ces_score import calculate_ces_score


# ============================================================
# 4. INFERENCE CONFIGURATION
# ============================================================

cv2.setNumThreads(1)
torch.set_num_threads(CPU_THREADS)


# ============================================================
# 5. COLLECT IMAGE-MASK PAIRS
# ============================================================

def get_image_pairs(split):
    """
    Collect processed image-mask pairs for a dataset split.

    Supported splits:
        train
        val
        test
    """

    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported dataset split: {split}")

    image_dir = PROCESSED_DATA_DIR / split / "images"
    mask_dir = PROCESSED_DATA_DIR / split / "masks"

    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory does not exist: {image_dir}"
        )

    if not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Mask directory does not exist: {mask_dir}"
        )

    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )

    if not image_paths:
        raise RuntimeError(
            f"No PNG images found in {image_dir}"
        )

    pairs = []

    for image_path in image_paths:

        mask_path = mask_dir / image_path.name

        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Missing mask for image: {image_path.name}"
            )

        pairs.append((image_path, mask_path))

    return pairs


# ============================================================
# 6. READ IMAGE AND GROUND-TRUTH MASK
# ============================================================

def read_pair(image_path, mask_path):
    """
    Read an already processed 256 x 256 image-mask pair.

    File reading is excluded from inference timing.
    """

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
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
        3,
    )

    expected_mask_shape = (
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    if image.shape != expected_image_shape:
        raise ValueError(
            f"Unexpected image shape for {image_path.name}: "
            f"{image.shape}"
        )

    if mask.shape != expected_mask_shape:
        raise ValueError(
            f"Unexpected mask shape for {mask_path.name}: "
            f"{mask.shape}"
        )

    # Binarize the ground-truth mask.
    ground_truth = (
        mask >= MASK_THRESHOLD
    ).astype(np.uint8)

    return image, ground_truth


# ============================================================
# 7. BENCHMARK A DATASET SPLIT
# ============================================================

def benchmark_split(split="val"):
    """
    Evaluate the selected model on a dataset split.

    Inference:
        CPU only
        Single thread
        One warmup run

    Timing:
        time.process_time()
        File I/O excluded
        Metric computation excluded

    Supported:
        val  - validation evaluation
        test - final DUTS-TE evaluation
    """

    if split not in {"val", "test"}:
        raise ValueError(
            "Split must be either 'val' or 'test'."
        )

    print("=" * 70)
    print("SWIFTMASK - CPU SEGMENTATION BENCHMARK")
    print("=" * 70)

    print("Dataset split:", split)
    print("Checkpoint:", FINAL_MODEL_PATH)

    # --------------------------------------------------------
    # Configure single-threaded inference
    # --------------------------------------------------------

    configure_inference()

    cv2.setNumThreads(1)
    torch.set_num_threads(CPU_THREADS)

    # --------------------------------------------------------
    # Load model on CPU
    # --------------------------------------------------------

    if not FINAL_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {FINAL_MODEL_PATH}"
        )

    model = load_model(
        FINAL_MODEL_PATH,
        device="cpu",
    )

    model.eval()

    print("Device: CPU")
    print("CPU threads:", torch.get_num_threads())

    # --------------------------------------------------------
    # Collect image-mask pairs
    # --------------------------------------------------------

    pairs = get_image_pairs(split)

    print("Images found:", len(pairs))

    if split == "test" and len(pairs) != 5019:
        raise RuntimeError(
            f"Expected 5019 DUTS-TE images, found {len(pairs)}. "
            "Check the processed test directory before evaluation."
        )

    # --------------------------------------------------------
    # Initialize evaluator
    # --------------------------------------------------------

    evaluator = SegmentationEvaluator(
        boundary_tolerance=BOUNDARY_TOLERANCE,
    )

    total_cpu_seconds = 0.0

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print("\nPerforming one warmup inference...")

    first_image, _ = read_pair(*pairs[0])

    warmup_prediction = predict_mask(
        first_image,
        model,
    )

    if warmup_prediction.shape != (
        IMAGE_SIZE,
        IMAGE_SIZE,
    ):
        raise RuntimeError(
            "Warmup prediction has an unexpected shape."
        )

    del warmup_prediction

    print("Warmup completed.")

    # --------------------------------------------------------
    # Run inference on every image
    # --------------------------------------------------------

    print("\nStarting evaluation...")
    print("-" * 70)

    for index, (image_path, mask_path) in enumerate(pairs):

        # File I/O is outside the timed section.
        image, ground_truth = read_pair(
            image_path,
            mask_path,
        )

        # ----------------------------------------------------
        # Timed inference
        # ----------------------------------------------------

        start_time = time.process_time()

        prediction = predict_mask(
            image,
            model,
        )

        end_time = time.process_time()

        total_cpu_seconds += (
            end_time - start_time
        )

        # ----------------------------------------------------
        # Validate prediction
        # ----------------------------------------------------

        if prediction.shape != (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            raise RuntimeError(
                f"Invalid prediction shape for {image_path.name}: "
                f"{prediction.shape}"
            )

        if not np.isin(prediction, [0, 1]).all():
            raise RuntimeError(
                f"Prediction is not binary for {image_path.name}"
            )

        # ----------------------------------------------------
        # Metrics are outside inference timing
        # ----------------------------------------------------

        evaluator.update(
            ground_truth,
            prediction,
        )

        completed = index + 1

        if completed % 100 == 0 or completed == len(pairs):
            print(
                f"Processed {completed}/{len(pairs)} images"
            )

    # ========================================================
    # 8. COMPUTE FINAL METRICS
    # ========================================================

    metrics = evaluator.compute()

    average_cpu_ms = (
        total_cpu_seconds / len(pairs)
    ) * 1000.0

    model_size_bytes = os.path.getsize(
        FINAL_MODEL_PATH
    )

    ces_results = calculate_ces_score(
        miou=metrics["miou"],
        boundary_f1=metrics["boundary_f1"],
        cpu_time_ms=average_cpu_ms,
        model_size_bytes=model_size_bytes,
    )

    # ========================================================
    # 9. SAVE RESULTS
    # ========================================================

    results = {
        "model": FINAL_MODEL_PATH.name,
        "split": split,
        "num_images": len(pairs),
        "input_resolution": 128,
        "output_resolution": IMAGE_SIZE,
        "threshold": 0.5,
        "device": "CPU",
        "cpu_threads": torch.get_num_threads(),
        "total_cpu_seconds": total_cpu_seconds,
        "average_cpu_ms": average_cpu_ms,
        "model_size_bytes": model_size_bytes,
        "object_iou": metrics["object_iou"],
        "background_iou": metrics["background_iou"],
        "miou": metrics["miou"],
        "boundary_f1": metrics["boundary_f1"],
        "quality_score": ces_results["quality_score"],
        "ces_score": ces_results["ces_score"],
        "confusion_matrix": metrics["confusion_matrix"],
    }

    output_dir = ROOT_DIR / "outputs" / "logs"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir / f"{split}_benchmark.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    # ========================================================
    # 10. DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 70)

    print("Model:", FINAL_MODEL_PATH.name)
    print("Split:", split)
    print("Images evaluated:", len(pairs))
    print("Device: CPU")
    print("CPU threads:", torch.get_num_threads())

    print("-" * 70)

    print(
        f"Total CPU time:   {total_cpu_seconds:.6f} seconds"
    )

    print(
        f"Average CPU time: {average_cpu_ms:.4f} ms/image"
    )

    print(
        f"Model size:       {model_size_bytes} bytes"
    )

    print("-" * 70)

    print(
        f"Object IoU:       {metrics['object_iou']:.6f}"
    )

    print(
        f"Background IoU:   {metrics['background_iou']:.6f}"
    )

    print(
        f"mIoU:             {metrics['miou']:.6f}"
    )

    print(
        f"Boundary F1:      {metrics['boundary_f1']:.6f}"
    )

    print(
        f"Quality Q:        {ces_results['quality_score']:.6f}"
    )

    print(
        f"CES:              {ces_results['ces_score']:.6f}"
    )

    print("-" * 70)

    print("Confusion matrix:")

    print(
        np.array(metrics["confusion_matrix"])
    )

    print("-" * 70)

    print("Results saved to:", output_path)

    print("=" * 70)

    return results


# ============================================================
# 11. COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="SwiftMask CPU segmentation benchmark.",
    )

    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help=(
            "Dataset split to evaluate. "
            "Use val for validation or test for final DUTS-TE evaluation."
        ),
    )

    args = parser.parse_args()

    benchmark_split(args.split)


if __name__ == "__main__":
    main()
