
import os
import sys
from pathlib import Path

# Configure CPU threads before importing numerical libraries.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

from ces.config import CAPTURED_DIR, IMAGE_SIZE


def image_sort_key(path: Path):
    """Sort img1, img2, ..., img10 in numerical order."""
    stem = path.stem.lower()

    if stem.startswith("img") and stem[3:].isdigit():
        return (0, int(stem[3:]))

    return (1, stem)


def main():
    original_dir = CAPTURED_DIR / "original"
    image_dir = CAPTURED_DIR / "images 256"
    mask_dir = CAPTURED_DIR / "masks 256"

    if not original_dir.exists():
        raise FileNotFoundError(
            f"Original image directory not found: {original_dir}"
        )

    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    cv2.setNumThreads(1)

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    original_paths = sorted(
        [
            path
            for path in original_dir.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        ],
        key=image_sort_key,
    )

    if not 5 <= len(original_paths) <= 10:
        raise ValueError(
            f"Expected 5–10 original photographs, found {len(original_paths)}."
        )

    # Prevent two originals from producing the same PNG filename.
    output_names = [f"{path.stem}.png" for path in original_paths]

    if len(output_names) != len(set(output_names)):
        raise ValueError(
            "Duplicate output filenames detected. "
            "Rename the original photographs before preprocessing."
        )

    print("=" * 65)
    print("SWIFTMASK - CAPTURED IMAGE PREPROCESSING")
    print("=" * 65)
    print(f"Original photographs : {len(original_paths)}")
    print(f"Target image size    : {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Original directory   : {original_dir}")
    print(f"Resized image folder : {image_dir}")
    print(f"Mask folder          : {mask_dir}")
    print()

    created_masks = 0
    preserved_masks = 0

    for index, original_path in enumerate(original_paths, start=1):
        image = cv2.imread(
            str(original_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"Could not read original image: {original_path}"
            )

        original_height, original_width = image.shape[:2]

        # Assignment requirement: direct resizing to 256 x 256.
        # This intentionally squashes non-square images.
        resized = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        output_name = f"{original_path.stem}.png"
        image_output = image_dir / output_name
        mask_output = mask_dir / output_name

        if not cv2.imwrite(str(image_output), resized):
            raise RuntimeError(
                f"Could not save resized image: {image_output}"
            )

        # Preserve existing masks, including manually annotated masks.
        if mask_output.exists():
            existing_mask = cv2.imread(
                str(mask_output),
                cv2.IMREAD_GRAYSCALE,
            )

            if existing_mask is None:
                raise RuntimeError(
                    f"Could not read existing mask: {mask_output}"
                )

            expected_shape = (IMAGE_SIZE, IMAGE_SIZE)

            if existing_mask.shape != expected_shape:
                raise ValueError(
                    f"Incorrect mask dimensions for {mask_output.name}: "
                    f"{existing_mask.shape}; expected {expected_shape}."
                )

            unique_values = np.unique(existing_mask)

            if not np.all(np.isin(unique_values, [0, 255])):
                raise ValueError(
                    f"Mask {mask_output.name} must contain only 0 and 255. "
                    f"Found values: {unique_values.tolist()}"
                )

            preserved_masks += 1
            mask_status = "EXISTING MASK PRESERVED"

        else:
            # Blank templates are created only for images without masks.
            blank_mask = np.zeros(
                (IMAGE_SIZE, IMAGE_SIZE),
                dtype=np.uint8,
            )

            if not cv2.imwrite(str(mask_output), blank_mask):
                raise RuntimeError(
                    f"Could not create mask template: {mask_output}"
                )

            created_masks += 1
            mask_status = "BLANK MASK CREATED"

        print(
            f"[{index:02d}/{len(original_paths):02d}] "
            f"{original_path.name}"
        )
        print(
            f"    Original: {original_width} x {original_height}"
        )
        print(
            f"    Resized : {IMAGE_SIZE} x {IMAGE_SIZE}"
        )
        print(f"    Mask    : {mask_status}")

    print()
    print("=" * 65)
    print("PREPROCESSING COMPLETED")
    print("=" * 65)
    print(f"Images processed          : {len(original_paths)}")
    print(f"New blank masks created   : {created_masks}")
    print(f"Existing masks preserved : {preserved_masks}")
    print(f"Resized images            : {image_dir}")
    print(f"Masks                    : {mask_dir}")
    print()
    print("IMPORTANT:")
    print("- Original photographs were not modified.")
    print("- Resized images follow the assignment's 256 x 256 requirement.")
    print("- Existing annotated masks were not overwritten.")
    print("- Annotate any remaining blank masks before evaluation.")


if __name__ == "__main__":
    main()
