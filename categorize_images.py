"""Script qui génère un json contenant les occurences des différentes classes fusionnées dans le dataset."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from configs.config_cityscapes import setting_config_cityscapes

configs = setting_config_cityscapes()
splits = ["train", "val", "test"]
occurences = {}
save_path = "data/occurences.json"

def main():
    for split in splits:
        # 1. Récupération des paires (image, mask)
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

        # 2. Recenser les images par classe
        occurences[split] = {}

        for label in range(20):
            occurence_label = []
            for paths in (data):
                mask = np.array(Image.open(paths[1]))
                if label in mask:
                    occurence_label.append(str(paths[1]))
            occurences[split][label] = occurence_label
            print(f"Dans le dataset {split}, {len(occurence_label)} images contiennent le label {label}")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(occurences, f, ensure_ascii=False, indent=4)
    print(f"Occurences enregistrées dans {save_path}")

if __name__ == "__main__":
    main()