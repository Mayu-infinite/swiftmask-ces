
import os
import sys
import argparse
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

from ces.config import (
    PROCESSED_DATA_DIR,
    VISUALIZATION_DIR,
    IMAGE_SIZE,
    MASK_THRESHOLD,
    FINAL_MODEL_PATH
)

from ces.pipeline import load_model, predict_mask


def add_label(image, label):
    image = image.copy()

    cv2.rectangle(
        image,
        (0, 0),
        (IMAGE_SIZE, 28),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        image,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return image


def create_visualization(image, ground_truth, prediction):
    """
    Create a four-panel comparison.

    Green: true positive foreground
    Red: false positive foreground
    Blue: false negative foreground
    """

    gt = ground_truth.astype(bool)
    pred = prediction.astype(bool)

    original_panel = add_label(
        image,
        "Original"
    )

    gt_panel = cv2.cvtColor(
        ground_truth * 255,
        cv2.COLOR_GRAY2BGR
    )
    gt_panel = add_label(gt_panel, "Ground Truth")

    pred_panel = cv2.cvtColor(
        prediction * 255,
        cv2.COLOR_GRAY2BGR
    )
    pred_panel = add_label(pred_panel, "Prediction")

    overlay = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.uint8
    )

    true_positive = gt & pred
    false_positive = (~gt) & pred
    false_negative = gt & (~pred)

    # BGR colors
    overlay[true_positive] = (0, 200, 0)
    overlay[false_positive] = (0, 0, 255)
    overlay[false_negative] = (255, 0, 0)

    overlay = add_label(
        overlay,
        "Error Overlay"
    )

    combined = cv2.hconcat([
        original_panel,
        gt_panel,
        pred_panel,
        overlay
    ])

    return combined


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--count",
        type=int,
        default=12
    )

    args = parser.parse_args()

    if args.count < 1:
        raise ValueError("Count must be positive.")

    image_dir = PROCESSED_DATA_DIR / "val" / "images"
    mask_dir = PROCESSED_DATA_DIR / "val" / "masks"

    image_paths = sorted(image_dir.glob("*.png"))

    if not image_paths:
        raise RuntimeError("No validation images found.")

    image_paths = image_paths[:args.count]

    model = load_model(
        FINAL_MODEL_PATH,
        device="cpu"
    )

    output_dir = VISUALIZATION_DIR / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating validation visualizations...")

    for index, image_path in enumerate(image_paths, start=1):

        mask_path = mask_dir / image_path.name

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        ground_truth = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None or ground_truth is None:
            print("Skipping:", image_path.name)
            continue

        ground_truth = (
            ground_truth >= MASK_THRESHOLD
        ).astype(np.uint8)

        prediction = predict_mask(
            image,
            model
        )

        visualization = create_visualization(
            image,
            ground_truth,
            prediction
        )

        output_path = (
            output_dir / f"{image_path.stem}_comparison.png"
        )

        cv2.imwrite(
            str(output_path),
            visualization
        )

        print(
            f"[{index}/{len(image_paths)}] Saved: {output_path}"
        )

    print("\nFinished.")
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
