
import os
import sys
import csv
import json
from pathlib import Path

# ============================================================
# 1. ENVIRONMENT CONFIGURATION
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


# ============================================================
# 2. REPOSITORY ROOT
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# 3. IMPORTS
# ============================================================

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from ces.config import (
    PROCESSED_DATA_DIR,
    MODEL_INPUT_SIZE,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    CPU_THREADS,
    FINAL_MODEL_PATH,
    BOUNDARY_TOLERANCE,
    THRESHOLD_CANDIDATES,
    THRESHOLD_TUNING_CPU_MS,
    MAX_CPU_TIME_MS,
    MAX_MODEL_SIZE_BYTES,
    QUALITY_IOU_WEIGHT,
    QUALITY_BOUNDARY_WEIGHT,
)

from ces.data_loader import DUTSDataset

from ces.methods.track_b import SwiftMaskFCN

from ces.metrics.evaluation import SegmentationEvaluator

from ces.metrics.ces_score import calculate_ces_score


# ============================================================
# 4. CONFIGURATION
# ============================================================

cv2.setNumThreads(1)
torch.set_num_threads(CPU_THREADS)

OUTPUT_DIR = (
    ROOT_DIR
    / "outputs"
    / "logs"
    / "threshold_optimization"
)


def get_device():
    """Select evaluation device."""

    if torch.cuda.is_available():
        device = torch.device("cuda")

        print(
            "Evaluation device:",
            torch.cuda.get_device_name(0)
        )
    else:
        device = torch.device("cpu")
        print("Evaluation device: CPU")

    return device


# ============================================================
# 5. LOAD MODEL
# ============================================================

def load_model(device):
    """Load the existing V2 checkpoint."""

    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {FINAL_MODEL_PATH}"
        )

    model = SwiftMaskFCN()

    state_dict = torch.load(
        FINAL_MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    print("Loaded checkpoint:", FINAL_MODEL_PATH)

    return model


# ============================================================
# 6. VALIDATION DATASET
# ============================================================

def create_validation_loader():
    """Create validation-only data loader."""

    validation_dataset = DUTSDataset(
        root=PROCESSED_DATA_DIR / "val",
        image_size=IMAGE_SIZE,
        augment=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
    )

    print(
        "Validation images:",
        len(validation_dataset)
    )

    return validation_loader


# ============================================================
# 7. EVALUATE ONE THRESHOLD
# ============================================================

@torch.inference_mode()
def evaluate_threshold(model, loader, device, threshold):
    """
    Evaluate the model using one sigmoid threshold.

    All images are evaluated using the same threshold.
    Metrics are aggregated across the complete validation set.
    """

    evaluator = SegmentationEvaluator(
        boundary_tolerance=BOUNDARY_TOLERANCE
    )

    total_batches = len(loader)

    print(
        f"\nEvaluating threshold: {threshold:.2f}"
    )

    for batch_index, (images, masks) in enumerate(loader):

        # Original images are 256 x 256.
        images = images.to(
            device,
            non_blocking=True,
        )

        # Resize images to the trained model resolution.
        model_images = F.interpolate(
            images,
            size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        # Model forward pass.
        logits = model(model_images)

        # Convert probabilities to binary predictions.
        probabilities = torch.sigmoid(logits)

        predictions = probabilities >= threshold

        # Restore predictions to 256 x 256.
        predictions = F.interpolate(
            predictions.float(),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="nearest",
        )

        predictions = predictions.cpu().numpy()
        targets = masks.cpu().numpy()

        # Update aggregate metrics image by image.
        for index in range(predictions.shape[0]):

            prediction = predictions[index, 0] > 0.5
            target = targets[index, 0] >= 0.5

            evaluator.update(
                target,
                prediction,
            )

        completed = batch_index + 1

        if completed % 10 == 0 or completed == total_batches:
            print(
                f"  Batches: {completed}/{total_batches}"
            )

    return evaluator.compute()


# ============================================================
# 8. CALCULATE QUALITY AND CES
# ============================================================

def calculate_scores(metrics, model_size_bytes):
    """Calculate Q and CES using assignment formula."""

    quality_q = (
        QUALITY_IOU_WEIGHT * metrics["miou"]
        + QUALITY_BOUNDARY_WEIGHT * metrics["boundary_f1"]
    )

    ces_result = calculate_ces_score(
        miou=metrics["miou"],
        boundary_f1=metrics["boundary_f1"],
        cpu_time_ms=THRESHOLD_TUNING_CPU_MS,
        model_size_bytes=model_size_bytes,
    )

    return {
        "quality_q": float(quality_q),
        "ces": float(ces_result["ces_score"]),
        "ces_quality_score": float(
            ces_result["quality_score"]
        ),
    }


# ============================================================
# 9. SAVE RESULTS
# ============================================================

def save_results(results, best_result):
    """Save CSV and JSON threshold comparison."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = OUTPUT_DIR / "threshold_comparison.json"
    csv_path = OUTPUT_DIR / "threshold_comparison.csv"
    best_path = OUTPUT_DIR / "best_threshold.json"

    # JSON comparison.
    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    # CSV comparison.
    fieldnames = [
        "threshold",
        "object_iou",
        "background_iou",
        "miou",
        "boundary_f1",
        "quality_q",
        "cpu_time_ms",
        "model_size_bytes",
        "ces",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results["experiments"]:
            writer.writerow({
                key: result[key]
                for key in fieldnames
            })

    # Best threshold.
    with open(
        best_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            best_result,
            file,
            indent=4,
        )

    print("\nSaved results:")
    print("JSON:", json_path)
    print("CSV:", csv_path)
    print("Best threshold:", best_path)


# ============================================================
# 10. MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 75)
    print("SWIFTMASK V2 - THRESHOLD OPTIMIZATION")
    print("=" * 75)

    print("Checkpoint:", FINAL_MODEL_PATH)
    print("Input resolution:", MODEL_INPUT_SIZE)
    print("Candidate thresholds:", THRESHOLD_CANDIDATES)
    print(
        "CPU latency used for CES:",
        THRESHOLD_TUNING_CPU_MS,
        "ms",
    )

    if THRESHOLD_TUNING_CPU_MS > MAX_CPU_TIME_MS:
        raise RuntimeError(
            "Reference CPU inference time exceeds the assignment limit."
        )

    model_size_bytes = FINAL_MODEL_PATH.stat().st_size

    if model_size_bytes > MAX_MODEL_SIZE_BYTES:
        raise RuntimeError(
            "Model size exceeds the assignment limit."
        )

    print(
        "Model size:",
        model_size_bytes,
        "bytes",
    )

    device = get_device()

    model = load_model(device)

    loader = create_validation_loader()

    experiments = []

    # --------------------------------------------------------
    # Evaluate every threshold.
    # --------------------------------------------------------

    for threshold in THRESHOLD_CANDIDATES:

        metrics = evaluate_threshold(
            model,
            loader,
            device,
            threshold,
        )

        scores = calculate_scores(
            metrics,
            model_size_bytes,
        )

        result = {
            "threshold": float(threshold),
            "num_images": int(metrics["num_images"]),
            "object_iou": float(metrics["object_iou"]),
            "background_iou": float(metrics["background_iou"]),
            "miou": float(metrics["miou"]),
            "boundary_f1": float(metrics["boundary_f1"]),
            "quality_q": scores["quality_q"],
            "cpu_time_ms": float(THRESHOLD_TUNING_CPU_MS),
            "model_size_bytes": int(model_size_bytes),
            "ces": scores["ces"],
            "confusion_matrix": np.asarray(
                metrics["confusion_matrix"]
            ).tolist(),
        }

        experiments.append(result)

        print(
            f"\nThreshold {threshold:.2f} | "
            f"mIoU {result['miou']:.6f} | "
            f"BF1 {result['boundary_f1']:.6f} | "
            f"Q {result['quality_q']:.6f} | "
            f"CES {result['ces']:.6f}"
        )

    # --------------------------------------------------------
    # Select best threshold by validation CES.
    # --------------------------------------------------------

    best_result = max(
        experiments,
        key=lambda item: item["ces"],
    )

    results = {
        "experiment": "SwiftMask V2 threshold optimization",
        "dataset": "DUTS validation",
        "checkpoint": str(FINAL_MODEL_PATH),
        "input_resolution": MODEL_INPUT_SIZE,
        "num_experiments": len(experiments),
        "cpu_time_reference_ms": THRESHOLD_TUNING_CPU_MS,
        "model_size_bytes": model_size_bytes,
        "selection_metric": "validation CES",
        "experiments": experiments,
        "best_result": best_result,
    }

    save_results(
        results,
        best_result,
    )

    # --------------------------------------------------------
    # Print comparison table.
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("THRESHOLD COMPARISON")
    print("=" * 75)

    print(
        f"{'Threshold':>10} "
        f"{'mIoU':>12} "
        f"{'BF1':>12} "
        f"{'Q':>12} "
        f"{'CES':>12}"
    )

    print("-" * 65)

    for result in experiments:

        marker = (
            "  <-- BEST"
            if result["threshold"] == best_result["threshold"]
            else ""
        )

        print(
            f"{result['threshold']:>10.2f} "
            f"{result['miou']:>12.6f} "
            f"{result['boundary_f1']:>12.6f} "
            f"{result['quality_q']:>12.6f} "
            f"{result['ces']:>12.6f}"
            f"{marker}"
        )

    print("-" * 65)

    print(
        "\nBEST VALIDATION THRESHOLD:",
        best_result["threshold"],
    )

    print(
        "Best validation CES:",
        f"{best_result['ces']:.6f}",
    )

    print(
        "\nIMPORTANT: This experiment does not modify "
        "the model checkpoint or config.py."
    )

    print(
        "Update THRESHOLD in config.py only after "
        "reviewing and accepting the results."
    )

    print("=" * 75)


if __name__ == "__main__":
    main()
