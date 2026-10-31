
import os
import sys
import json
import time
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
import torch
from torch.utils.data import DataLoader

from ces.config import (
    PROCESSED_DATA_DIR,
    MODEL_INPUT_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_WORKERS,
    CPU_THREADS,
    RANDOM_SEED,
    CHECKPOINT_INTERVAL,
    set_random_seed,
)

from ces.data_loader import DUTSDataset

from ces.methods.track_b import (
    SwiftMaskFCN,
    SegmentationLoss,
    count_parameters,
)

# ============================================================
# 4. EXPERIMENT B OUTPUT PATHS
# ============================================================

EXPERIMENT_NAME = "experiment_b_192"

BEST_MODEL_PATH = (
    ROOT_DIR / "models" / "final" / "swiftmask_fcn_192.pt"
)

CHECKPOINT_DIR = (
    ROOT_DIR / "models" / "checkpoints" / EXPERIMENT_NAME
)

LOG_DIR = ROOT_DIR / "outputs" / "logs"

HISTORY_PATH = LOG_DIR / "training_history_192.json"

# ============================================================
# 5. DEVICE AND THREAD CONFIGURATION
# ============================================================

cv2.setNumThreads(1)
torch.set_num_threads(CPU_THREADS)


def get_device():
    """Select CUDA GPU when available."""

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA device:", torch.cuda.get_device_name(0))
        print(
            "GPU memory:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3),
                2,
            ),
            "GB",
        )
    else:
        device = torch.device("cpu")
        print("CUDA unavailable. Using CPU.")

    return device


# ============================================================
# 6. DATASET AND DATALOADER
# ============================================================

def create_dataloaders():

    train_dataset = DUTSDataset(
        root=PROCESSED_DATA_DIR / "train",
        image_size=MODEL_INPUT_SIZE,
    )

    val_dataset = DUTSDataset(
        root=PROCESSED_DATA_DIR / "val",
        image_size=MODEL_INPUT_SIZE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False,
    )

    print("\nDataset information")
    print("-------------------")
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Batch size:", BATCH_SIZE)
    print("Input resolution:", MODEL_INPUT_SIZE)

    return train_loader, val_loader


# ============================================================
# 7. TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    running_loss = 0.0
    total_samples = 0

    for batch_index, (images, masks) in enumerate(loader):

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)

        loss = criterion(logits, masks)

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        total_samples += batch_size

        if (batch_index + 1) % 100 == 0:
            print(
                f"    Batch {batch_index + 1}/{len(loader)}"
                f" | Loss: {loss.item():.5f}"
            )

    epoch_loss = running_loss / max(total_samples, 1)

    return epoch_loss


# ============================================================
# 8. VALIDATION FUNCTION
# ============================================================

@torch.no_grad()
def validate(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0
    total_samples = 0

    for images, masks in loader:

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)

        loss = criterion(logits, masks)

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        total_samples += batch_size

    validation_loss = running_loss / max(total_samples, 1)

    return validation_loss


# ============================================================
# 9. MODEL SAVING
# ============================================================

def save_model(model, output_path):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        output_path,
    )


# ============================================================
# 10. TRAINING PIPELINE
# ============================================================

def main():

    print("=" * 65)
    print("SWIFTMASK - EXPERIMENT B: 192x192 FCN TRAINING")
    print("=" * 65)

    # Prevent accidentally training with the baseline resolution.
    if MODEL_INPUT_SIZE != 192:
        raise ValueError(
            f"Experiment B requires MODEL_INPUT_SIZE=192, "
            f"but config.py currently has {MODEL_INPUT_SIZE}."
        )

    set_random_seed(RANDOM_SEED)

    device = get_device()

    # Create output directories.
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("\nExperiment configuration")
    print("------------------------")
    print("Experiment:", EXPERIMENT_NAME)
    print("Input resolution:", MODEL_INPUT_SIZE)
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Random seed:", RANDOM_SEED)
    print("Best model output:", BEST_MODEL_PATH)
    print("History output:", HISTORY_PATH)

    train_loader, val_loader = create_dataloaders()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SwiftMaskFCN().to(device)

    print("\nModel information")
    print("-----------------")
    print("Architecture: SwiftMaskFCN")
    print("Trainable parameters:", count_parameters(model))
    print("Device:", device)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = SegmentationLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = []

    best_validation_loss = float("inf")
    best_epoch = 0

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    print("\nStarting Experiment B training")
    print("=" * 65)

    training_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):

        epoch_start = time.perf_counter()

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 65)

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        epoch_time = time.perf_counter() - epoch_start

        print("\nEpoch results")
        print("-------------")
        print(f"Training loss:   {train_loss:.6f}")
        print(f"Validation loss: {val_loss:.6f}")
        print(f"Epoch time:      {epoch_time:.2f} seconds")

        # ----------------------------------------------------
        # Save best model using validation loss
        # ----------------------------------------------------

        if val_loss < best_validation_loss:

            best_validation_loss = val_loss
            best_epoch = epoch

            save_model(
                model,
                BEST_MODEL_PATH,
            )

            print("\nNew best Experiment B model saved!")
            print("Path:", BEST_MODEL_PATH)

        # ----------------------------------------------------
        # Periodic checkpoints
        # ----------------------------------------------------

        if epoch % CHECKPOINT_INTERVAL == 0:

            checkpoint_path = (
                CHECKPOINT_DIR / f"epoch_{epoch:03d}.pt"
            )

            save_model(
                model,
                checkpoint_path,
            )

            print(
                "Checkpoint saved:",
                checkpoint_path.name,
            )

        # ----------------------------------------------------
        # Save training history after every epoch
        # ----------------------------------------------------

        history.append(
            {
                "experiment": EXPERIMENT_NAME,
                "input_resolution": MODEL_INPUT_SIZE,
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "epoch_time_seconds": epoch_time,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "batch_size": BATCH_SIZE,
            }
        )

        with open(
            HISTORY_PATH,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                history,
                file,
                indent=4,
            )

    total_training_time = time.perf_counter() - training_start

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 65)
    print("EXPERIMENT B TRAINING COMPLETED")
    print("=" * 65)

    print("Experiment:", EXPERIMENT_NAME)
    print("Input resolution:", MODEL_INPUT_SIZE)
    print("Best epoch:", best_epoch)

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.6f}"
    )

    print(
        f"Total training time: "
        f"{total_training_time / 60:.2f} minutes"
    )

    print("Best model:", BEST_MODEL_PATH)
    print("Training history:", HISTORY_PATH)
    print("Checkpoint directory:", CHECKPOINT_DIR)

    print(
        "Model parameter count:",
        count_parameters(model),
    )

    if BEST_MODEL_PATH.exists():
        print(
            "Best model size:",
            BEST_MODEL_PATH.stat().st_size,
            "bytes",
        )

    print("=" * 65)


# ============================================================
# 11. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
