"""
Script de preprocessing du dataset CityScapes.
"""

from pathlib import Path
import random

import numpy as np
from PIL import Image

split = "train"
laser_shape = 50

data_path = Path("data/cityscapes")
images_dir = data_path / "leftImg8bit" / split

images_paths = sorted(images_dir.glob("**/*_leftImg8bit.png"))
if not images_paths:
    raise FileNotFoundError(f"Aucune image dans {images_dir}")

dataset = []
missing = []
for img_path in images_paths:
    mask_path = Path(
        str(img_path)
        .replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png")
        .replace("leftImg8bit", "gtFine")
    )
    if not mask_path.exists():
        missing.append(str(mask_path))
        continue
    dataset.append((img_path, mask_path))

if missing:
    raise FileNotFoundError(f"{len(missing)} masque(s) manquant(s) dans le path {missing[0]}")

print(f"[CityscapesDataset] split={split} -> {len(dataset)} paires")

yy, xx = np.mgrid[0:laser_shape, 0:laser_shape]
center = (laser_shape - 1) / 2
sigma = laser_shape / 6
gaussian = np.exp(-(((xx - center) ** 2 + (yy - center) ** 2) / (2 * sigma ** 2))).astype(np.float32)
laser_disc = gaussian > 0.2

for img_path, mask_path in dataset:
    image = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))

    x = random.randint(laser_shape, image.shape[0] - 2 * laser_shape)
    y = random.randint(laser_shape, image.shape[1] - 2 * laser_shape)

    add = gaussian * 255
    region = image[x:x+laser_shape, y:y+laser_shape].astype(np.float32)
    region[..., 0] = np.clip(region[..., 0] + add, 0, 255)
    region[..., 1] = np.clip(region[..., 1] + add * 0.5, 0, 255)
    region[..., 2] = np.clip(region[..., 2] + add * 0.5, 0, 255)
    image[x:x+laser_shape, y:y+laser_shape] = region.astype(np.uint8)

    mask_region = mask[x:x+laser_shape, y:y+laser_shape]
    mask_region[laser_disc] = 19
    mask[x:x+laser_shape, y:y+laser_shape] = mask_region

    Image.fromarray(image).save(str(img_path).replace("_leftImg8bit.png", "_leftImg8bit_laser.png"))
    Image.fromarray(mask).save(str(mask_path).replace("_gtFine_labelTrainIds.png", "_gtFine_labelTrainIds_laser.png"))
