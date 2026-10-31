
import os
import sys
import json
from pathlib import Path

# =========================================================
# 1. ENVIRONMENT CONFIGURATION
# =========================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# =========================================================
# 2. REPOSITORY ROOT
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# =========================================================
# 3. IMPORTS
# =========================================================

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
    BOUNDARY_TOLERANCE
)

from ces.data_loader import DUTSDataset

from ces.methods.track_b import SwiftMaskFCN

from ces.metrics.evaluation import SegmentationEvaluator

# =========================================================
# 4. CONFIGURATION
# =========================================================

cv2.setNumThreads(1)
torch.set_num_threads(CPU_THREADS)


def get_device():

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Evaluation device:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Evaluation device: CPU")

    return device


# =========================================================
# 5. LOAD MODEL
# =========================================================

def load_model(device):

    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {FINAL_MODEL_PATH}"
        )

    model = SwiftMaskFCN()

    state_dict = torch.load(
        FINAL_MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    print("Loaded checkpoint:", FINAL_MODEL_PATH)

    return model


# =========================================================
# 6. VALIDATION DATASET
# =========================================================

def create_validation_loader():

    validation_dataset = DUTSDataset(
        root=PROCESSED_DATA_DIR / "val",
        image_size=IMAGE_SIZE
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0)
    )

    print("Validation images:", len(validation_dataset))

    return validation_loader


# =========================================================
# 7. EVALUATION
# =========================================================

@torch.inference_mode()
def evaluate(model, loader, device):

    evaluator = SegmentationEvaluator(
        boundary_tolerance=BOUNDARY_TOLERANCE
    )

    total_batches = len(loader)

    print("\nStarting validation evaluation...")
    print("-" * 60)

    for batch_index, (images, masks) in enumerate(loader):

        # Original images and masks are 256 x 256.
        images = images.to(
            device,
            non_blocking=True
        )

        # Model operates at 128 x 128.
        model_images = F.interpolate(
            images,
            size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            mode="bilinear",
            align_corners=False
        )

        logits = model(model_images)

        # Convert logits into binary foreground predictions.
        predictions = torch.sigmoid(logits) >= 0.5

        # Upsample predictions to original 256 x 256.
        predictions = F.interpolate(
            predictions.float(),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="nearest"
        )

        predictions = predictions.cpu().numpy()
        targets = masks.cpu().numpy()

        for index in range(predictions.shape[0]):

            prediction = predictions[index, 0] >= 0.5
            target = targets[index, 0] >= 0.5

            evaluator.update(
                target,
                prediction
            )

        completed = batch_index + 1

        if completed % 10 == 0 or completed == total_batches:
            print(
                f"Evaluated batches: {completed}/{total_batches}"
            )

    return evaluator.compute()


# =========================================================
# 8. SAVE RESULTS
# =========================================================

def save_results(results):

    output_dir = ROOT_DIR / "outputs" / "logs"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = output_dir / "validation_metrics.json"

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print("\nResults saved:", output_path)


# =========================================================
# 9. MAIN
# =========================================================

def main():

    print("=" * 60)
    print("SWIFTMASK - VALIDATION EVALUATION")
    print("=" * 60)

    device = get_device()

    model = load_model(device)

    loader = create_validation_loader()

    results = evaluate(
        model,
        loader,
        device
    )

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    print("Images evaluated:", results["num_images"])

    print(
        f"Object IoU:      {results['object_iou']:.6f}"
    )

    print(
        f"Background IoU:  {results['background_iou']:.6f}"
    )

    print(
        f"mIoU:            {results['miou']:.6f}"
    )

    print(
        f"Boundary F1:     {results['boundary_f1']:.6f}"
    )

    print("\nConfusion matrix:")
    print(np.array(results["confusion_matrix"]))

    save_results(results)

    print("=" * 60)


if __name__ == "__main__":
    main()
