
import os

# ============================================================
# 1. ENVIRONMENT CONFIGURATION
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# ============================================================
# 2. IMPORTS
# ============================================================

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ces.config import (
    IMAGE_SIZE,
    MODEL_INPUT_SIZE,
    CPU_THREADS,
    FINAL_MODEL_PATH,
)

from ces.methods.track_b import SwiftMaskFCN


# ============================================================
# 3. INFERENCE CONFIGURATION
# ============================================================

def configure_inference():
    """Configure single-threaded inference."""

    cv2.setNumThreads(1)
    torch.set_num_threads(CPU_THREADS)


# ============================================================
# 4. MODEL LOADING
# ============================================================

def load_model(model_path=FINAL_MODEL_PATH, device="cpu"):
    """
    Load the trained SwiftMaskFCN checkpoint.

    Model loading is outside the timed inference path.
    """

    configure_inference()

    model_path = os.fspath(model_path)
    device = torch.device(device)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    model = SwiftMaskFCN()

    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


# ============================================================
# 5. PREDICTION
# ============================================================

@torch.inference_mode()
def predict_mask(
    image_bgr,
    model,
    threshold=0.5,
    input_size=None,
):
    """
    Generate a binary segmentation mask.

    Parameters
    ----------
    image_bgr:
        uint8 BGR image with shape (256, 256, 3).

    model:
        Loaded SwiftMaskFCN model.

    threshold:
        Sigmoid probability threshold.

    input_size:
        Model input resolution.
        Use 128 for Experiment A.
        Use 192 for Experiment B.
        Defaults to MODEL_INPUT_SIZE if omitted.

    Returns
    -------
    uint8 binary mask of shape (256, 256).

    Background = 0
    Foreground = 1

    All operations inside this function are part
    of the timed inference path.
    """

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if image_bgr is None:
        raise ValueError("Input image is None.")

    if image_bgr.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(
            f"Expected image shape "
            f"({IMAGE_SIZE}, {IMAGE_SIZE}, 3), "
            f"received {image_bgr.shape}"
        )

    if image_bgr.dtype != np.uint8:
        raise ValueError(
            "Input image must have dtype uint8."
        )

    if input_size is None:
        input_size = MODEL_INPUT_SIZE

    if not isinstance(input_size, int) or input_size <= 0:
        raise ValueError(
            "input_size must be a positive integer."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    # --------------------------------------------------------
    # 1. BGR to RGB
    # --------------------------------------------------------

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    # --------------------------------------------------------
    # 2. Resize to the selected model resolution
    # --------------------------------------------------------

    image_rgb = cv2.resize(
        image_rgb,
        (input_size, input_size),
        interpolation=cv2.INTER_AREA,
    )

    # --------------------------------------------------------
    # 3. Normalize and convert to tensor
    # --------------------------------------------------------

    image_rgb = image_rgb.astype(
        np.float32
    ) / 255.0

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

    # Model and input must be on the same device.
    device = next(model.parameters()).device

    image_tensor = image_tensor.to(device)

    # --------------------------------------------------------
    # 4. Model forward pass
    # --------------------------------------------------------

    logits = model(image_tensor)

    # --------------------------------------------------------
    # 5. Binary prediction at model resolution
    # --------------------------------------------------------

    prediction = (
        torch.sigmoid(logits) >= threshold
    ).float()

    # --------------------------------------------------------
    # 6. Resize binary mask to 256 x 256
    # --------------------------------------------------------

    prediction = F.interpolate(
        prediction,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="nearest",
    )

    # --------------------------------------------------------
    # 7. Convert to NumPy binary mask
    # --------------------------------------------------------

    prediction = prediction[0, 0].cpu().numpy()

    return prediction.astype(np.uint8)
