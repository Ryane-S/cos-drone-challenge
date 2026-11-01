"""Script de génération du dataset synthétique"""
from pathlib import Path

import numpy as np
from PIL import Image

from configs.config_cityscapes import setting_config_cityscapes
from preprocessing.laser_generation.synthetic_laser import generate_synthetic_laser

configs = setting_config_cityscapes()
splits = ["train", "val", "test"]

def main():
    for split in splits:
        # 1. Récupération des paires (image, mask) pour un split donné
        data_path = Path(configs.data_path)
        images_dir = data_path / "leftImg8bit" / split
        images_paths = sorted(images_dir.glob("**/*_leftImg8bit.png"))
        
        data = []
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
            data.append((img_path, mask_path))
        
            if missing:
                raise FileNotFoundError(f"{len(missing)} masque(s) manquant(s) dans le path {mask_path}")

        print(f"Split actuel : {split}")
        print(f"[CityscapesDataset] split={split} -> {len(data)} paires à traiter.")
        if len(data) == 0:
            continue

        # 2. Ajouter un faisceau laser sur l'image et son masque correspondant
        print(f"Generating synthetic {split} dataset !")

        rng = np.random.default_rng(45)

        for idx, paths in enumerate(data):
            image = np.array(Image.open(paths[0]).convert("RGB"), dtype=float) / 255
            mask = np.array(Image.open(paths[1]))

            seed = int(rng.integers(0, 2**32,))

            result, metadata = generate_synthetic_laser(image, seed=int(seed))

            if metadata["bbox"] is not None:
                i_min, i_max, j_min, j_max = metadata["bbox"]
                mask[i_min:i_max + 1, j_min:j_max + 1] = 19

            output_dir = data_path / "laser"
            rel_img = paths[0].relative_to(data_path)
            rel_mask = paths[1].relative_to(data_path)
            image_path_laser = output_dir / rel_img
            mask_path_laser = output_dir / rel_mask

            image_path_laser.parent.mkdir(parents=True, exist_ok=True)
            mask_path_laser.parent.mkdir(parents=True, exist_ok=True)

            image_uint8 = (result * 255).clip(0, 255).astype(np.uint8)

            print(f"{idx+1} Paire(s) traitée(s) !")

            print(f"Sauvegarde {image_path_laser}")
            Image.fromarray(image_uint8).save(image_path_laser)

            print(f"Sauvegarde {mask_path_laser}")
            Image.fromarray(mask).save(mask_path_laser)

if __name__ == '__main__':
    main()
