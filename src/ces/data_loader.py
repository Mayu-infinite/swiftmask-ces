
import os
import random

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import cv2
import numpy as np
import torch

from torch.utils.data import Dataset

from ces.config import MODEL_INPUT_SIZE, MASK_THRESHOLD


class DUTSDataset(Dataset):
    """
    DUTS binary foreground segmentation dataset.

    Augmentation is optional and is enabled only for the training set.
    Every geometric transformation is applied identically to the image
    and its corresponding mask.
    """

    def __init__(
        self,
        root,
        image_size=MODEL_INPUT_SIZE,
        augment=False,
    ):
        self.root = str(root)
        self.image_size = image_size
        self.augment = augment

        self.image_dir = os.path.join(self.root, "images")
        self.mask_dir = os.path.join(self.root, "masks")

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(self.image_dir)

        if not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(self.mask_dir)

        self.image_names = sorted(
            name
            for name in os.listdir(self.image_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        self.samples = []

        for name in self.image_names:
            image_path = os.path.join(self.image_dir, name)
            mask_name = os.path.splitext(name)[0] + ".png"
            mask_path = os.path.join(self.mask_dir, mask_name)

            if os.path.isfile(mask_path):
                self.samples.append((image_path, mask_path))

        if not self.samples:
            raise RuntimeError("No matching image-mask pairs found.")

    def __len__(self):
        return len(self.samples)

    def _geometric_augmentation(self, image, mask):
        # Horizontal and vertical flips.
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        if random.random() < 0.15:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        # Moderate zoom variation around the image centre.
        if random.random() < 0.5:
            height, width = image.shape[:2]
            scale = random.uniform(0.85, 1.15)

            matrix = cv2.getRotationMatrix2D(
                (width / 2.0, height / 2.0),
                0,
                scale,
            )

            image = cv2.warpAffine(
                image,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            mask = cv2.warpAffine(
                mask,
                matrix,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        return image, mask

    def _photometric_augmentation(self, image):
        image = image.astype(np.float32)

        # Contrast and brightness variation.
        alpha = random.uniform(0.85, 1.15)
        beta = random.uniform(-20.0, 20.0)

        image = alpha * image + beta

        # Independent mild colour-channel variation.
        channel_gains = np.array(
            [
                random.uniform(0.92, 1.08),
                random.uniform(0.92, 1.08),
                random.uniform(0.92, 1.08),
            ],
            dtype=np.float32,
        )

        image = image * channel_gains.reshape(1, 1, 3)

        image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise RuntimeError(f"Unable to read image: {image_path}")

        if mask is None:
            raise RuntimeError(f"Unable to read mask: {mask_path}")

        # Match the assignment's fixed 256x256 processed images,
        # then resize to the experiment's model input resolution.
        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        if self.augment:
            image, mask = self._geometric_augmentation(image, mask)
            image = self._photometric_augmentation(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        mask = (mask >= MASK_THRESHOLD).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = mask[np.newaxis, :, :]

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image, dtype=np.float32)
        )

        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(mask, dtype=np.float32)
        )

        return image_tensor, mask_tensor
