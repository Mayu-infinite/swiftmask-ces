
import os
import sys
import random
from pathlib import Path

# =========================================================
# 1. ENVIRONMENT CONFIGURATION
# =========================================================

# Set thread limits before importing NumPy, OpenCV or PyTorch.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# =========================================================
# 2. REPOSITORY ROOT CONFIGURATION
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# =========================================================
# 3. THIRD-PARTY IMPORTS
# =========================================================

import cv2

cv2.setNumThreads(1)

# =========================================================
# 4. PROJECT IMPORTS
# =========================================================

from ces.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SPLIT_DIR,
    IMAGE_SIZE,
    MASK_THRESHOLD,
    TRAIN_RATIO,
    RANDOM_SEED
)


# =========================================================
# 5. DIRECTORY CREATION
# =========================================================

def create_directories():
    """
    Create processed dataset directories.
    """

    for split in ["train", "val", "test"]:

        image_dir = PROCESSED_DATA_DIR / split / "images"
        mask_dir = PROCESSED_DATA_DIR / split / "masks"

        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 6. IMAGE-MASK PAIR VALIDATION
# =========================================================

def find_pairs(image_dir, mask_dir):
    """
    Match images and masks using filename stems.

    Each image must have a corresponding PNG mask.
    Also checks for masks without corresponding images.
    """

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory does not exist: {image_dir}"
        )

    if not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Mask directory does not exist: {mask_dir}"
        )

    valid_extensions = (".jpg", ".jpeg", ".png")

    image_files = sorted(
        path for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in valid_extensions
    )

    mask_files = sorted(
        path for path in mask_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".png"
    )

    mask_lookup = {
        path.stem: path
        for path in mask_files
    }

    pairs = []
    missing_masks = []

    for image_path in image_files:

        mask_path = mask_lookup.get(image_path.stem)

        if mask_path is None:
            missing_masks.append(image_path.name)
            continue

        pairs.append((image_path, mask_path))

    image_stems = {
        image_path.stem
        for image_path in image_files
    }

    unmatched_masks = [
        mask_path.name
        for mask_path in mask_files
        if mask_path.stem not in image_stems
    ]

    if missing_masks:
        raise RuntimeError(
            f"{len(missing_masks)} images have no matching masks. "
            f"Examples: {missing_masks[:5]}"
        )

    if unmatched_masks:
        raise RuntimeError(
            f"{len(unmatched_masks)} masks have no matching images. "
            f"Examples: {unmatched_masks[:5]}"
        )

    if not pairs:
        raise RuntimeError(
            f"No image-mask pairs found in {image_dir}"
        )

    print(
        f"Validated {len(pairs)} image-mask pairs "
        f"in {image_dir.parent.name}"
    )

    return pairs


# =========================================================
# 7. IMAGE AND MASK PREPROCESSING
# =========================================================

def process_pair(
    image_path,
    mask_path,
    output_image_path,
    output_mask_path
):
    """
    Resize one image-mask pair according to assignment rules.

    Image:
        256 x 256 using INTER_AREA.

    Mask:
        256 x 256 using INTER_NEAREST.
        Binary threshold >= 128.
        Output values: 0 and 255.
    """

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
            f"Cannot read image: {image_path}"
        )

    if mask is None:
        raise RuntimeError(
            f"Cannot read mask: {mask_path}"
        )

    # Resize image.
    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # Resize mask.
    mask = cv2.resize(
        mask,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    # Convert mask to binary.
    mask = (
        mask >= MASK_THRESHOLD
    ).astype("uint8") * 255

    # Save processed image.
    success_image = cv2.imwrite(
        str(output_image_path),
        image
    )

    # Save processed mask.
    success_mask = cv2.imwrite(
        str(output_mask_path),
        mask
    )

    if not success_image:
        raise RuntimeError(
            f"Failed to save image: {output_image_path}"
        )

    if not success_mask:
        raise RuntimeError(
            f"Failed to save mask: {output_mask_path}"
        )


# =========================================================
# 8. PROCESS DATASET SPLIT
# =========================================================

def process_split(pairs, split_name):
    """
    Process all image-mask pairs belonging to a split.
    """

    image_output = (
        PROCESSED_DATA_DIR / split_name / "images"
    )

    mask_output = (
        PROCESSED_DATA_DIR / split_name / "masks"
    )

    total = len(pairs)

    print(f"\nProcessing {split_name}: {total} images")

    for index, (image_path, mask_path) in enumerate(pairs):

        filename = image_path.stem + ".png"

        process_pair(
            image_path,
            mask_path,
            image_output / filename,
            mask_output / filename
        )

        completed = index + 1

        if completed % 500 == 0 or completed == total:
            print(
                f"{split_name}: {completed}/{total}"
            )

    print(
        f"{split_name}: completed {total} images."
    )


# =========================================================
# 9. SAVE SPLIT INFORMATION
# =========================================================

def save_split_file(filename, pairs):
    """
    Save original image filenames for reproducibility.
    """

    output_path = SPLIT_DIR / filename

    with open(output_path, "w", encoding="utf-8") as file:

        for image_path, _ in pairs:
            file.write(image_path.name + "\n")

    print(
        f"Saved {len(pairs)} entries to {output_path}"
    )


# =========================================================
# 10. MAIN DATASET PREPARATION
# =========================================================

def main():

    print("=" * 60)
    print("SWIFTMASK - DUTS DATASET PREPARATION")
    print("=" * 60)

    random.seed(RANDOM_SEED)

    create_directories()

    # -----------------------------------------------------
    # DUTS-TR: TRAINING AND VALIDATION
    # -----------------------------------------------------

    print("\n[1/2] Preparing DUTS-TR")

    train_image_dir = (
        RAW_DATA_DIR / "DUTS-TR" / "DUTS-TR-Image"
    )

    train_mask_dir = (
        RAW_DATA_DIR / "DUTS-TR" / "DUTS-TR-Mask"
    )

    train_pairs = find_pairs(
        train_image_dir,
        train_mask_dir
    )

    random.shuffle(train_pairs)

    split_index = int(
        len(train_pairs) * TRAIN_RATIO
    )

    training_pairs = train_pairs[:split_index]
    validation_pairs = train_pairs[split_index:]

    print("\nDUTS-TR split:")
    print(f"Total:      {len(train_pairs)}")
    print(f"Training:   {len(training_pairs)}")
    print(f"Validation: {len(validation_pairs)}")

    process_split(
        training_pairs,
        "train"
    )

    process_split(
        validation_pairs,
        "val"
    )

    save_split_file(
        "train.txt",
        training_pairs
    )

    save_split_file(
        "val.txt",
        validation_pairs
    )

    # -----------------------------------------------------
    # DUTS-TE: FINAL TEST SET
    # -----------------------------------------------------

    print("\n[2/2] Preparing DUTS-TE")

    test_image_dir = (
        RAW_DATA_DIR / "DUTS-TE" / "DUTS-TE-Image"
    )

    test_mask_dir = (
        RAW_DATA_DIR / "DUTS-TE" / "DUTS-TE-Mask"
    )

    test_pairs = find_pairs(
        test_image_dir,
        test_mask_dir
    )

    print(
        f"\nDUTS-TE images: {len(test_pairs)}"
    )

    process_split(
        test_pairs,
        "test"
    )

    # -----------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 60)

    print(f"Training images:   {len(training_pairs)}")
    print(f"Validation images: {len(validation_pairs)}")
    print(f"Test images:       {len(test_pairs)}")

    print(f"\nProcessed data: {PROCESSED_DATA_DIR}")
    print(f"Split files:     {SPLIT_DIR}")

    print("=" * 60)


# =========================================================
# 11. SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
