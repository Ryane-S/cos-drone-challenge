"""
Loader du dataset Cityscapes. (Inspiré de : https://github.com/JCruan519/VM-UNet/blob/main/datasets/dataset.py)
"""

from pathlib import Path
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from dataset_loaders.utils import random_rot_flip, random_rotate


class CityscapesDataset(Dataset):
    """
    Charge les paires (image, masque) du dataset Cityscapes.
    """
    def __init__(self, data_path, split="train", transform=None):
        super().__init__()
        self.split = split
        self.transform = transform

        self.data_path = Path(data_path)
        self.images_dir = self.data_path / "leftImg8bit" / split

        images_paths = sorted(self.images_dir.glob("**/*_leftImg8bit.png"))
        if not images_paths:
            raise FileNotFoundError(f"Aucune image dans {self.images_dir}")

        self.data = []
        missing = []
        for img_path in images_paths:
            mask_path = Path(
                str(img_path)
                .replace("leftImg8bit", "gtFine")
                .replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png")
            )
            if not mask_path.exists():
                missing.append(str(mask_path))
                continue
            self.data.append((img_path, mask_path))

        if missing:
            raise FileNotFoundError(f"{len(missing)} masque(s) manquant(s)")

        print(f"[CityscapesDataset] split={split} -> {len(self.data)} paires")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, mask_path = self.data[index]
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        sample = {"image": img, "label": mask}
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample["image"], sample["label"]
    

class CityscapesGenerator:
    """
    Generateur unique, mode train (augmenté) ou val (déterministe).

    Prend un dict {"image": HWC uint8, "label": HW uint8} et retourne
    {"image": CHW float32 [0,1], "label": HW int64}.
    """

    def __init__(self, output_size=(512, 1024), train=True):
        self.output_size = output_size  # (H, W)
        self.train = train

    def __call__(self, sample):
        image, mask = sample["image"], sample["label"]

        # Augmentations aléatoires (uniquement en train)
        if self.train:
            r = random.random()
            if r > 0.5:
                image, mask = random_rot_flip(image, mask)
            elif r > 0.25:
                image, mask = random_rotate(image, mask)

        # Resize
        h, w = image.shape[:2]
        th, tw = self.output_size
        if (h, w) != (th, tw):
            image = cv2.resize(image, (tw, th), interpolation=cv2.INTER_CUBIC)
            mask  = cv2.resize(mask,  (tw, th), interpolation=cv2.INTER_NEAREST)

        # Conversion tenseurs - .float() et .long() remplacent numpy
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_t  = torch.from_numpy(mask).long()

        return {"image": image_t, "label": mask_t}


def build_dataloader(data_path, split, batch_size=2, num_workers=4, img_size=(512, 1024), pin_memory=True, drop_last=None):
    """Construit un DataLoader Cityscapes pour un split donné."""
    train = (split == "train")
    
    if drop_last is None:
        drop_last = train

    transform = CityscapesGenerator(img_size, train=train)
    dataset = CityscapesDataset(data_path, split=split, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=drop_last,
    )


if __name__ == "__main__":
    for split in ["train", "val"]:
        dl = build_dataloader("data/cityscapes", split, num_workers=0)
        imgs, masks = next(iter(dl))
        print(f"\n--- {split} ---")
        print(f"images : {imgs.shape} {imgs.dtype} min={imgs.min():.3f} max={imgs.max():.3f}")
        print(f"masks  : {masks.shape} {masks.dtype} unique={torch.unique(masks).tolist()}")