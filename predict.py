import os
import sys
from pathlib import Path

# Set thread limits before importing numerical libraries.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# Add src directory to Python module search path.
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import argparse
import time

import cv2
import torch

from ces.config import (
    IMAGE_SIZE,
    FINAL_MODEL_PATH,
    BENCHMARK_RUNS,
    CPU_THREADS,
)

from ces.pipeline import (
    load_model,
    predict_mask,
    configure_inference,
)


def main():

    parser = argparse.ArgumentParser(
        description="SwiftMask binary segmentation"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input photo"
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Path to save the predicted mask"
    )

    args = parser.parse_args()

    configure_inference()

    torch.set_num_threads(CPU_THREADS)

    # -----------------------------------------------------
    # Load input image (outside timed section)
    # -----------------------------------------------------

    image_path = Path(args.image)

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise FileNotFoundError(
            f"Cannot read input image: {image_path}"
        )

    # Assignment requires a 256 x 256 image.
    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # -----------------------------------------------------
    # Load model (outside timed section)
    # -----------------------------------------------------

    model = load_model(
        FINAL_MODEL_PATH,
        device="cpu"
    )

    model.eval()

    # -----------------------------------------------------
    # One warmup run (excluded from timing)
    # -----------------------------------------------------

    prediction = predict_mask(
        image,
        model
    )

    # -----------------------------------------------------
    # Benchmark 20 additional inference runs
    # -----------------------------------------------------

    start = time.process_time()

    for _ in range(BENCHMARK_RUNS):

        prediction = predict_mask(
            image,
            model
        )

    end = time.process_time()

    total_cpu_seconds = end - start

    average_cpu_ms = (
        total_cpu_seconds / BENCHMARK_RUNS
    ) * 1000.0

    # -----------------------------------------------------
    # Save prediction (outside timed section)
    # -----------------------------------------------------

    output_path = Path(args.out)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_mask = (
        prediction.astype("uint8") * 255
    )

    success = cv2.imwrite(
        str(output_path),
        output_mask
    )

    if not success:
        raise RuntimeError(
            f"Failed to save prediction: {output_path}"
        )

    # -----------------------------------------------------
    # Model size
    # -----------------------------------------------------

    model_size_bytes = os.path.getsize(
        FINAL_MODEL_PATH
    )

    # Assignment requires exactly two output lines.
    print(f"T CPU ms: {average_cpu_ms:.2f}")
    print(f"S disk bytes: {model_size_bytes}")


if __name__ == "__main__":
    main()