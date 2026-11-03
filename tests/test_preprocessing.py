import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ces.preprocessing import preprocess_image, preprocess_mask


def test_preprocess_image_and_mask_shapes_are_valid():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)

    processed_image = preprocess_image(image)
    processed_mask = preprocess_mask(mask)

    assert processed_image.shape == (256, 256, 3)
    assert processed_mask.shape == (256, 256)
