
import os
import sys
import json
import time
import random
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

import cv2
import numpy as np
import torch

from torch.utils.data import DataLoader

from ces.config import (
    PROCESSED_DATA_DIR,
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


# V2-specific experiment settings.
EXPERIMENT_NAME = "swiftmask_u2_v2"
INPUT_SIZE = 128

BATCH_SIZE_V2 = 16
EPOCHS_V2 = EPOCHS

MODEL_OUTPUT = (
    ROOT_DIR / "models" / "final" / "swiftmask_u2_v2.pt"
)

CHECKPOINT_OUTPUT = (
    ROOT_DIR / "models" / "checkpoints" / EXPERIMENT_NAME
)

LOG_OUTPUT = (
    ROOT_DIR / "outputs" / "logs" / "v2"
)

HISTORY_PATH = LOG_OUTPUT / "training_history_v2.json"


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA device:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("CUDA unavailable. Using CPU.")

    return device


def create_dataloaders():
    train_dataset = DUTSDataset(
        root=PROCESSED_DATA_DIR / "train",
        image_size=INPUT_SIZE,
        augment=True,
    )

    val_dataset = DUTSDataset(
        root=PROCESSED_DATA_DIR / "val",
        image_size=INPUT_SIZE,
        augment=False,
    )

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_V2,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE_V2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
        worker_init_fn=seed_worker,
        drop_last=False,
    )

    print("\nDataset information")
    print("-------------------")
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Input resolution:", INPUT_SIZE)
    print("Training augmentation: enabled")
    print("Validation augmentation: disabled")

    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
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

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if (batch_index + 1) % 100 == 0:
            print(
                f"Batch {batch_index + 1}/{len(loader)} "
                f"| Loss: {loss.item():.5f}"
            )

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def save_model(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def main():
    print("=" * 65)
    print("SWIFTMASK V2 - LIGHTWEIGHT U2-INSPIRED TRAINING")
    print("=" * 65)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    set_random_seed(RANDOM_SEED)

    cv2.setNumThreads(1)
    torch.set_num_threads(CPU_THREADS)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = get_device()

    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_OUTPUT.mkdir(parents=True, exist_ok=True)
    LOG_OUTPUT.mkdir(parents=True, exist_ok=True)

    print("\nExperiment configuration")
    print("------------------------")
    print("Experiment:", EXPERIMENT_NAME)
    print("Input resolution:", INPUT_SIZE)
    print("Epochs:", EPOCHS_V2)
    print("Batch size:", BATCH_SIZE_V2)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Device:", device)
    print("Model output:", MODEL_OUTPUT)

    train_loader, val_loader = create_dataloaders()

    model = SwiftMaskFCN().to(device)

    print("\nModel information")
    print("-----------------")
    print("Architecture: SwiftMask U2-inspired")
    print("Trainable parameters:", count_parameters(model))

    criterion = SegmentationLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history = []
    best_validation_loss = float("inf")
    best_epoch = 0

    training_start = time.perf_counter()

    for epoch in range(1, EPOCHS_V2 + 1):
        epoch_start = time.perf_counter()

        print("\n" + "=" * 65)
        print(f"Epoch {epoch}/{EPOCHS_V2}")
        print("=" * 65)

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

        print(f"Training loss:   {train_loss:.6f}")
        print(f"Validation loss: {val_loss:.6f}")
        print(f"Epoch time:      {epoch_time:.2f} seconds")

        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_epoch = epoch

            save_model(model, MODEL_OUTPUT)

            print("New best V2 model saved:", MODEL_OUTPUT)

        if epoch % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = (
                CHECKPOINT_OUTPUT / f"epoch_{epoch:03d}.pt"
            )
            save_model(model, checkpoint_path)
            print("Checkpoint saved:", checkpoint_path.name)

        history.append({
            "experiment": EXPERIMENT_NAME,
            "input_resolution": INPUT_SIZE,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "epoch_time_seconds": epoch_time,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "batch_size": BATCH_SIZE_V2,
            "augmentation": True,
        })

        with open(HISTORY_PATH, "w", encoding="utf-8") as file:
            json.dump(history, file, indent=4)

    total_time = time.perf_counter() - training_start

    print("\n" + "=" * 65)
    print("SWIFTMASK V2 TRAINING COMPLETED")
    print("=" * 65)
    print("Best epoch:", best_epoch)
    print("Best validation loss:", f"{best_validation_loss:.6f}")
    print("Total training time:", f"{total_time / 60:.2f} minutes")
    print("Model:", MODEL_OUTPUT)
    print("Model size:", MODEL_OUTPUT.stat().st_size, "bytes")
    print("History:", HISTORY_PATH)
    print("Parameters:", count_parameters(model))


if __name__ == "__main__":
    main()
