
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from ces.config import PROCESSED_DATA_DIR
from ces.data_loader import DUTSDataset
from ces.methods.track_b import (
    SwiftMaskFCN,
    SegmentationLoss,
    count_parameters,
)


def main():
    print("===== SWIFTMASK V2 INTEGRATION TEST =====")

    # Verify package exports.
    model = SwiftMaskFCN()
    print("Model parameters:", count_parameters(model))

    criterion = SegmentationLoss()
    print("Loss function:", type(criterion).__name__)

    # Verify augmented training data.
    train_dataset = DUTSDataset(
        PROCESSED_DATA_DIR / "train",
        image_size=128,
        augment=True,
    )

    image, mask = train_dataset[0]

    print("Training image shape:", tuple(image.shape))
    print("Training mask shape:", tuple(mask.shape))
    print("Training mask values:", torch.unique(mask).tolist())

    assert image.shape == (3, 128, 128)
    assert mask.shape == (1, 128, 128)
    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})

    # Verify validation data without augmentation.
    val_dataset = DUTSDataset(
        PROCESSED_DATA_DIR / "val",
        image_size=128,
        augment=False,
    )

    val_image, val_mask = val_dataset[0]

    assert val_image.shape == (3, 128, 128)
    assert val_mask.shape == (1, 128, 128)

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Augmentation test: PASSED")
    print("Validation dataset test: PASSED")
    print("SWIFTMASK V2 INTEGRATION TEST: PASSED")


if __name__ == "__main__":
    main()
