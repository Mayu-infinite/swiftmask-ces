import os

# Set thread limits before importing numerical libraries.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import cv2
import numpy as np

from ces.config import IMAGE_SIZE, MASK_THRESHOLD


def resize_image(image, size=IMAGE_SIZE):
    """
    Resize an image to a square resolution.

    Uses INTER_AREA as required by the assignment.
    """

    resized = cv2.resize(
        image,
        (size, size),
        interpolation=cv2.INTER_AREA
    )

    return resized


def resize_mask(mask, size=IMAGE_SIZE):
    """
    Resize a segmentation mask.

    Nearest-neighbor interpolation preserves discrete labels.
    """

    resized = cv2.resize(
        mask,
        (size, size),
        interpolation=cv2.INTER_NEAREST
    )

    return resized


def binarize_mask(mask):
    """
    Convert a grayscale mask into binary foreground/background.

    Pixels >= 128 become 1.
    Other pixels become 0.
    """

    return (mask >= MASK_THRESHOLD).astype(np.uint8)


def preprocess_image(image):
    """
    Prepare an image for CNN input.

    Input:
        BGR image, uint8, H x W x 3

    Output:
        RGB image normalized to [0, 1], float32
    """

    if image is None:
        raise ValueError("Input image is None.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a three-channel BGR image.")

    image = resize_image(image)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(np.float32) / 255.0

    return image


def preprocess_mask(mask):
    """
    Prepare a segmentation mask.

    Output:
        Binary mask with values 0 and 1.
    """

    if mask is None:
        raise ValueError("Input mask is None.")

    if mask.ndim == 3:
        mask = cv2.cvtColor(
            mask,
            cv2.COLOR_BGR2GRAY
        )

    mask = resize_mask(mask)

    return binarize_mask(mask)


def image_to_tensor_array(image):
    """
    Convert HWC image to CHW format.

    Normalized image values remain in [0, 1].
    """

    image = np.transpose(image, (2, 0, 1))

    return np.ascontiguousarray(image, dtype=np.float32)


def mask_to_tensor_array(mask):
    """
    Convert binary mask into a single-channel array.
    """

    return np.ascontiguousarray(
        mask[np.newaxis, :, :],
        dtype=np.float32
    )