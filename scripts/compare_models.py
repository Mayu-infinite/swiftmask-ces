
import os
import sys
import json
import time
from pathlib import Path

# Configure threads before importing numerical libraries.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ces.config import (
    IMAGE_SIZE,
    MODEL_INPUT_SIZE,
    MASK_THRESHOLD,
    BOUNDARY_TOLERANCE,
    CPU_THREADS,
)

from ces.methods.track_b.model_v1 import SwiftMaskFCN as SwiftMaskV1
from ces.methods.track_b.model_v2 import SwiftMaskFCN as SwiftMaskV2

from ces.metrics.evaluation import SegmentationEvaluator
from ces.metrics.ces_score import calculate_ces_score


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

VAL_DIR = ROOT_DIR / "data" / "processed" / "DUTS" / "val"

MODELS = {
    "V1": {
        "class": SwiftMaskV1,
        "checkpoint": ROOT_DIR / "models/final/swiftmask_fcn_v1_backup.pt",
    },
    "V2": {
        "class": SwiftMaskV2,
        "checkpoint": ROOT_DIR / "models/final/swiftmask_u2_v2.pt",
    },
}

OUTPUT_DIR = ROOT_DIR / "outputs/logs/comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.50


# ------------------------------------------------------------
# ENVIRONMENT
# ------------------------------------------------------------

cv2.setNumThreads(1)
torch.set_num_threads(1)


# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

def get_validation_pairs():

    image_dir = VAL_DIR / "images"
    mask_dir = VAL_DIR / "masks"

    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Validation directories not found: {VAL_DIR}"
        )

    image_paths = sorted(image_dir.glob("*.png"))

    if not image_paths:
        raise RuntimeError("No validation images found.")

    pairs = []

    for image_path in image_paths:
        mask_path = mask_dir / image_path.name

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Missing mask: {mask_path}"
            )

        pairs.append((image_path, mask_path))

    return pairs


def read_pair(image_path, mask_path):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None or mask is None:
        raise RuntimeError(
            f"Unable to read {image_path.name}"
        )

    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(
            f"Unexpected image shape: {image.shape}"
        )

    if mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"Unexpected mask shape: {mask.shape}"
        )

    mask = (mask >= MASK_THRESHOLD).astype(np.uint8)

    return image, mask


# ------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------

def load_model(name, config):

    checkpoint_path = config["checkpoint"]

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    model = config["class"]()

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict, strict=True)

    model.eval()

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"\n{name} model loaded")
    print("Parameters:", parameters)
    print("Checkpoint:", checkpoint_path.name)
    print("Checkpoint bytes:", checkpoint_path.stat().st_size)

    return model


# ------------------------------------------------------------
# INFERENCE
# ------------------------------------------------------------

@torch.inference_mode()
def predict_mask(image_bgr, model):

    if image_bgr.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError("Expected a 256x256 BGR image.")

    # BGR -> RGB
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    # Resize to model resolution.
    image_rgb = cv2.resize(
        image_rgb,
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    # Normalize.
    image_rgb = image_rgb.astype(np.float32) / 255.0

    image_chw = np.transpose(
        image_rgb,
        (2, 0, 1),
    )

    image_chw = np.ascontiguousarray(
        image_chw,
        dtype=np.float32,
    )

    image_tensor = torch.from_numpy(
        image_chw
    ).unsqueeze(0)

    # Forward pass.
    logits = model(image_tensor)

    if logits.shape[-2:] != (
        MODEL_INPUT_SIZE,
        MODEL_INPUT_SIZE,
    ):
        raise ValueError(
            f"Unexpected model output: {logits.shape}"
        )

    prediction = (
        torch.sigmoid(logits) >= THRESHOLD
    ).float()

    # Restore 256x256 binary mask.
    prediction = F.interpolate(
        prediction,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="nearest",
    )

    return (
        prediction[0, 0]
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


# ------------------------------------------------------------
# EVALUATION AND TIMING
# ------------------------------------------------------------

def evaluate_model(name, config, pairs):

    model = load_model(name, config)

    evaluator = SegmentationEvaluator(
        boundary_tolerance=BOUNDARY_TOLERANCE,
    )

    total_cpu_seconds = 0.0

    # One warm-up inference, excluded from timing.
    first_image, _ = read_pair(*pairs[0])
    predict_mask(first_image, model)

    print(f"\nEvaluating {name}...")
    
    for index, (image_path, mask_path) in enumerate(pairs):

        # File I/O is outside the timed section.
        image, ground_truth = read_pair(
            image_path,
            mask_path,
        )

        start = time.process_time()

        prediction = predict_mask(
            image,
            model,
        )

        end = time.process_time()

        total_cpu_seconds += end - start

        # Metric calculations are outside inference timing.
        evaluator.update(
            ground_truth,
            prediction,
        )

        completed = index + 1

        if completed % 100 == 0 or completed == len(pairs):
            print(
                f"{name}: {completed}/{len(pairs)} images"
            )

    metrics = evaluator.compute()

    average_cpu_ms = (
        total_cpu_seconds / len(pairs)
    ) * 1000.0

    model_size_bytes = config["checkpoint"].stat().st_size

    ces = calculate_ces_score(
        miou=metrics["miou"],
        boundary_f1=metrics["boundary_f1"],
        cpu_time_ms=average_cpu_ms,
        model_size_bytes=model_size_bytes,
    )

    results = {
        "model": name,
        "split": "validation",
        "num_images": len(pairs),
        "input_resolution": MODEL_INPUT_SIZE,
        "output_resolution": IMAGE_SIZE,
        "threshold": THRESHOLD,
        "device": "CPU",
        "cpu_threads": torch.get_num_threads(),
        "parameters": sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ),
        "total_cpu_seconds": total_cpu_seconds,
        "average_cpu_ms": average_cpu_ms,
        "model_size_bytes": model_size_bytes,
        "object_iou": metrics["object_iou"],
        "background_iou": metrics["background_iou"],
        "miou": metrics["miou"],
        "boundary_f1": metrics["boundary_f1"],
        "quality_score": ces["quality_score"],
        "ces_score": ces["ces_score"],
        "confusion_matrix": metrics["confusion_matrix"],
    }

    del model

    return results


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 65)
    print("SWIFTMASK V1 vs V2 - VALIDATION COMPARISON")
    print("=" * 65)

    print("Validation directory:", VAL_DIR)
    print("Model input:", MODEL_INPUT_SIZE)
    print("Output resolution:", IMAGE_SIZE)
    print("Device: CPU")
    print("Threads:", torch.get_num_threads())

    pairs = get_validation_pairs()

    print("Validation images:", len(pairs))

    all_results = {}

    for name, config in MODELS.items():

        results = evaluate_model(
            name,
            config,
            pairs,
        )

        all_results[name] = results

    # Save complete results.
    output_path = OUTPUT_DIR / "v1_vs_v2_validation.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            all_results,
            file,
            indent=4,
        )

    # Display comparison.
    print("\n" + "=" * 85)
    print("FINAL VALIDATION COMPARISON")
    print("=" * 85)

    columns = [
        ("Model", "model"),
        ("mIoU", "miou"),
        ("BF1", "boundary_f1"),
        ("Q", "quality_score"),
        ("CPU ms", "average_cpu_ms"),
        ("Size bytes", "model_size_bytes"),
        ("CES", "ces_score"),
    ]

    print(
        " | ".join(
            f"{label:>12}"
            for label, _ in columns
        )
    )

    print("-" * 85)

    for name, result in all_results.items():

        values = []

        for label, key in columns:
            value = result[key]

            if isinstance(value, float):
                values.append(f"{value:12.6f}")
            else:
                values.append(f"{str(value):>12}")

        print(" | ".join(values))

    print("\nDetailed results:", output_path)
    print("=" * 85)


if __name__ == "__main__":
    main()
