from pathlib import Path
import random

import cv2
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset


class Phase0Dataset(Dataset):
    def __init__(self, root, dataset, split, augment=False):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")
        if augment and split != "train":
            raise ValueError("Augmentation is allowed only for training")

        self.folder = Path(root) / dataset / split
        self.images = sorted((self.folder / "images").glob("*.png"))
        self.augment = augment

        if not self.images:
            raise FileNotFoundError(f"No images found: {self.folder}")

        image_names = {p.name for p in self.images}
        mask_names = {
            p.name for p in (self.folder / "masks").glob("*.png")
        }
        if image_names != mask_names:
            raise ValueError(f"Image/mask filenames differ: {self.folder}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.folder / "masks" / image_path.name

        # Preserve the OpenCV channel order used by the source loader.
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

        if image is None or mask is None:
            raise RuntimeError(f"Unreadable pair: {image_path}")
        if image.shape != (224, 224, 3) or mask.shape != (224, 224):
            raise ValueError(f"Unexpected shape: {image_path}")
        if not np.isin(mask, [0, 1]).all():
            raise ValueError(f"Nonbinary mask: {mask_path}")

        # Match the source's branching probabilities:
        # 50% rotate/flip, 25% arbitrary rotation, 25% unchanged.
        if self.augment:
            if random.random() > 0.5:
                k = np.random.randint(0, 4)
                axis = np.random.randint(0, 2)
                image = np.flip(np.rot90(image, k), axis=axis)
                mask = np.flip(np.rot90(mask, k), axis=axis)
            elif random.random() > 0.5:
                angle = np.random.randint(-20, 20)
                image = ndimage.rotate(
                    image, angle, axes=(0, 1),
                    order=0, reshape=False
                )
                mask = ndimage.rotate(
                    mask, angle, axes=(0, 1),
                    order=0, reshape=False
                )

        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        mask = np.ascontiguousarray(mask[None])

        return {
            "image": torch.from_numpy(image).float() / 255.0,
            "mask": torch.from_numpy(mask).float(),
            "sample_id": image_path.stem,
        }
