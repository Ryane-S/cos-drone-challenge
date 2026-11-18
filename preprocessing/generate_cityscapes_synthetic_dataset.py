"""Script de génération du dataset Cityscapes synthétique (avec reprise)."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from configs.config_cityscapes import setting_config_cityscapes
from preprocessing.laser_generation.synthetic_laser import generate_synthetic_laser

configs = setting_config_cityscapes()
splits = ["train", "val", "test"]

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

# Tous les fichiers attendus dans un dossier d'image (metadata.json en dernier)
EXPECTED_IMAGES = (
    "original.png",
    "laser.png",
    "original_mask.png",
    "laser_mask.png",
    "laser_mask_alone.png",
)
EXPECTED_TEXT = ("class.txt", "metadata.json")


# Vérification d'intégrité
def is_png_valid(path: Path) -> bool:
    """Vérifie qu'un PNG existe, n'est pas vide et est lisible."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def is_text_valid(path: Path) -> bool:
    """Vérifie qu'un fichier texte existe et n'est pas vide."""
    return path.exists() and path.stat().st_size > 0


def is_metadata_valid(path: Path) -> bool:
    """Vérifie que metadata.json est un JSON valide avec les clés attendues."""
    if not is_text_valid(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return "seed" in data and "contour" in data
    except (json.JSONDecodeError, OSError):
        return False


def is_sample_complete(folder: Path) -> bool:
    """Vérifie qu'un dossier d'image contient tous les fichiers valides."""
    if not folder.is_dir():
        return False
    for name in EXPECTED_IMAGES:
        if not is_png_valid(folder / name):
            return False
    for name in EXPECTED_TEXT[:-1]:   # class.txt
        if not is_text_valid(folder / name):
            return False
    if not is_metadata_valid(folder / "metadata.json"):
        return False
    return True


# Utilitaires
def seed_for(img_path: Path, base_seed: int = 45) -> int:
    """Seed déterministe par image, indépendante de l'ordre de traitement."""
    digest = hashlib.sha256(str(img_path).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) ^ base_seed


def compute_out_dir(img_path: Path, data_path: Path, split: str, laser_root: Path) -> Path:
    """Construit le dossier de sortie pour une image donnée."""
    rel_img = img_path.relative_to(data_path)
    ville = rel_img.parent.name
    image_name = rel_img.stem.replace("_leftImg8bit", "")
    if ville == split:
        ville = image_name.split("_")[0] if "_" in image_name else "_unknown"
    return laser_root / split / ville / image_name


def main():
    for split in splits:
        data_path = Path(configs.data_path)
        images_dir = data_path / "leftImg8bit" / split
        images_paths = sorted(images_dir.glob("**/*_leftImg8bit.png"))

        # 1. Construction des paires (image, masque)
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
            raise FileNotFoundError(
                f"{len(missing)} masque(s) manquant(s), ex: {missing[0]}"
            )

        print(f"\n=== Split : {split} ===")
        print(f"[CityscapesDataset] {len(data)} paires à traiter.")

        if len(data) == 0:
            continue

        laser_root = data_path / "laser"

        # 2. Grouper par ville
        by_city = {}
        for img_path, mask_path in data:
            city = compute_out_dir(img_path, data_path, split, laser_root).parent.name
            by_city.setdefault(city, []).append((img_path, mask_path))

        # 3. Compteurs
        total = len(data)
        skipped = 0
        processed = 0

        # 4. Traiter ville par ville
        for city, pairs in sorted(by_city.items()):
            # 4a. Vérifier si la ville est déjà complète
            out_dirs = [
                compute_out_dir(img_path, data_path, split, laser_root)
                for img_path, _ in pairs
            ]
            if all(is_sample_complete(d) for d in out_dirs):
                print(f"[{split}] Ville '{city}' déjà complète "
                      f"({len(pairs)} images), skip.")
                skipped += len(pairs)
                continue

            # 4b. Traiter les images manquantes de cette ville
            print(f"[{split}] Ville '{city}' : {len(pairs)} image(s) à traiter.")

            for (img_path, mask_path), out_dir in zip(pairs, out_dirs):
                if is_sample_complete(out_dir):
                    skipped += 1
                    continue

                # --- Génération ---
                image = np.array(Image.open(img_path).convert("RGB"), dtype=float) / 255
                mask_original = np.array(Image.open(mask_path))

                seed = seed_for(img_path)
                result, metadata = generate_synthetic_laser(image, seed=seed)

                laser_mask_alone = np.zeros(mask_original.shape, dtype=np.uint8)
                laser_mask = mask_original.copy()

                if metadata["contour"]:
                    pts = np.array(
                        [(j, i) for (i, j) in metadata["contour"]],
                        dtype=np.int32,
                    ).reshape(-1, 1, 2)
                    cv2.fillPoly(laser_mask_alone, [pts], 255)
                    cv2.fillPoly(laser_mask, [pts], 19)
                    classes_under = np.unique(mask_original[laser_mask_alone > 0])
                else:
                    classes_under = np.array([], dtype=np.uint8)

                out_dir.mkdir(parents=True, exist_ok=True)

                # Écriture des images et masques
                Image.fromarray((image * 255).clip(0, 255).astype(np.uint8)).save(
                    out_dir / "original.png"
                )
                Image.fromarray((result * 255).clip(0, 255).astype(np.uint8)).save(
                    out_dir / "laser.png"
                )
                Image.fromarray(mask_original).save(out_dir / "original_mask.png")
                Image.fromarray(laser_mask).save(out_dir / "laser_mask.png")
                Image.fromarray(laser_mask_alone).save(out_dir / "laser_mask_alone.png")

                # class.txt
                merged_classes = sorted(
                    {TRAINID_TO_MERGED[int(c)] for c in classes_under}
                )
                with open(out_dir / "class.txt", "w", encoding="utf-8") as f:
                    for c in merged_classes:
                        f.write(f"{c} {MERGED_NAMES[c]}\n")

                # metadata.json EN DERNIER (marqueur de complétion)
                with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                processed += 1
                if processed % 100 == 0:
                    print(f"  [{split}] {processed} images générées...")

        # 5. Récapitulatif
        print(f"\n[{split}] Récapitulatif : "
              f"{total} total, {skipped} déjà faites, {processed} générées.")


if __name__ == "__main__":
    main()