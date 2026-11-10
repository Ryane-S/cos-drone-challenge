"""Script de génération du dataset Cityscapes synthétique"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import tqdm

from configs.config_cityscapes import setting_config_cityscapes
from preprocessing.laser_generation.synthetic_laser import generate_synthetic_laser

configs = setting_config_cityscapes()
splits = ["train", "test"]

# Mapping trainId (0-18, 255) -> classe fusionnée (0-6, 19, 255)
TRAINID_TO_MERGED = {
    0: 0, 1: 0,                                    # flat
    2: 1, 3: 1, 4: 1,                              # construction
    5: 2, 6: 2, 7: 2,                              # urban_object
    8: 3, 9: 3,                                    # nature
    10: 4,                                         # sky
    11: 5, 12: 5,                                  # human
    13: 6, 14: 6, 15: 6, 16: 6, 17: 6, 18: 6,      # vehicle
    19: 19,                                        # laser
    255: 255,                                      # rest
}

MERGED_NAMES = {
    0: "flat",
    1: "construction",
    2: "urban_object",
    3: "nature",
    4: "sky",
    5: "human",
    6: "vehicle",
    19: "laser",
    255: "rest",
}


def main():
    rng = np.random.default_rng(45)

    for split in splits:
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
            raise FileNotFoundError(f"{len(missing)} masque(s) manquant(s), ex: {missing[0]}")

        if not data:
            print(f"[Cityscapes] split={split} : aucune paire à traiter.")
            continue

        laser_root = data_path / "laser"

        print(f"\n[Cityscapes] Génération du split {split} — {len(data)} paires")

        with tqdm.tqdm(
                total=len(data),
                desc=f"Cityscapes | {split}",
                unit="img",
                colour="cyan",
                dynamic_ncols=True,
                mininterval=0.2,
                bar_format=(
                        "{desc} |{bar}| {percentage:6.2f}% "
                        "| {n_fmt}/{total_fmt} "
                        "| écoulé: {elapsed} "
                        "| ETA: {remaining} "
                        "| {rate_noinv_fmt}"
                ),
        ) as progress:
            for img_path, mask_path in data:
                image = np.array(Image.open(img_path).convert("RGB"), dtype=float) / 255
                mask_original = np.array(Image.open(mask_path))

                seed = int(rng.integers(0, 2**32))
                result, metadata = generate_synthetic_laser(image, seed=seed)

                # Construction des masques
                # (a) laser_mask_alone : {0, 255}, uniquement le laser
                laser_mask_alone = np.zeros(mask_original.shape, dtype=np.uint8)

                # (b) laser_mask : copie du masque original avec laser=19 sur le contour
                laser_mask = mask_original.copy()

                if metadata["contour"]:
                    pts = np.array(
                        [(j, i) for (i, j) in metadata["contour"]],
                        dtype=np.int32,
                    ).reshape(-1, 1, 2)

                    cv2.fillPoly(laser_mask_alone, [pts], 255)
                    cv2.fillPoly(laser_mask, [pts], 19)

                    # Classes sous le laser (lues sur le masque ORIGINAL, avant modification)
                    classes_under = np.unique(mask_original[laser_mask_alone > 0])
                else:
                    classes_under = np.array([], dtype=np.uint8)

                # Détermination du dossier de sortie
                rel_img = img_path.relative_to(data_path)
                ville = rel_img.parent.name
                image_name = rel_img.stem.replace("_leftImg8bit", "")

                if ville == split:
                    ville = image_name.split("_")[0] if "_" in image_name else "_unknown"

                out_dir = laser_root / split / ville / image_name
                out_dir.mkdir(parents=True, exist_ok=True)

                # Sauvegarde images
                Image.fromarray((image * 255).clip(0, 255).astype(np.uint8)).save(out_dir / "original.png")
                Image.fromarray((result * 255).clip(0, 255).astype(np.uint8)).save(out_dir / "laser.png")

                # Sauvegarde masques
                Image.fromarray(mask_original).save(out_dir / "original_mask.png")
                Image.fromarray(laser_mask).save(out_dir / "laser_mask.png")
                Image.fromarray(laser_mask_alone).save(out_dir / "laser_mask_alone.png")

                # class.txt : classes fusionnées sous le laser
                merged_classes = sorted({TRAINID_TO_MERGED[int(c)] for c in classes_under})
                with open(out_dir / "class.txt", "w", encoding="utf-8") as f:
                    for c in merged_classes:
                        f.write(f"{c} {MERGED_NAMES[c]}\n")

                # metadata.json
                with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                progress.update(1)


if __name__ == "__main__":
    main()