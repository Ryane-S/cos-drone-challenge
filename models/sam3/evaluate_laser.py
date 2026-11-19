"""Chargement des données, inférence SAM et sauvegarde des résultats.

Les calculs sont délégués à laser_metrics.py.
Le détecteur est fourni par le script appelant : il n'est pas chargé ici.

Exports
-------
images.csv       : informations de chaque image et paramètres du laser.
per_image.csv    : résultats par image et prompt, avec classes support.
by_prompt.csv    : effectifs, moyennes et écarts-types par prompt.
by_class.csv     : mêmes statistiques par prompt et classe support.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from math import ceil

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from models.sam3.laser_metrics import LaserMetrics


TRAINID_TO_MERGED = {
    0: 0, 1: 0,                       # flat
    2: 1, 3: 1, 4: 1,                 # construction
    5: 2, 6: 2, 7: 2,                 # urban_object
    8: 3, 9: 3,                       # nature
    10: 4,                            # sky
    11: 5, 12: 5,                     # human
    13: 6, 14: 6, 15: 6,
    16: 6, 17: 6, 18: 6,              # vehicle
    255: 255,                         # rest
}

CLASS_NAMES = {
    0: "flat",
    1: "construction",
    2: "urban_object",
    3: "nature",
    4: "sky",
    5: "human",
    6: "vehicle",
    255: "rest",
}


def read_mask(path, binary=False):
    """Lit un masque indexé 2D ou un masque binaire 0/1 ou 0/255.

    Un binaire RGB est accepté si ses trois canaux sont identiques.
    Les visualisations RGB des classes Cityscapes sont refusées.
    """
    with Image.open(path) as image:
        mask = np.asarray(image).copy()

    if mask.ndim == 3 and binary and mask.shape[-1] in (3, 4):
        if not np.all(mask[..., :3] == mask[..., :1]):
            raise ValueError(f"{path} : le masque binaire RGB doit être gris.")
        mask = mask[..., 0]

    if mask.ndim != 2:
        raise ValueError(f"{path} : un masque indexé 2D est requis.")

    if binary:
        if not set(np.unique(mask)).issubset({0, 1, 255}):
            raise ValueError(f"{path} : valeurs binaires 0/1 ou 0/255 requises.")
        return mask > 0

    return mask


def support_classes(original, target, encoding):
    """Identifie les classes couvertes par le laser à la résolution native.

    encoding vaut 'trainid' pour les 19 classes initiales ou 'merged'
    si original_mask.png contient déjà les sept classes regroupées.
    Le masque original ne doit pas contenir la classe laser 19.
    """
    if original.shape != target.shape:
        raise ValueError("Le masque original et le masque laser sont désalignés.")

    ids = np.unique(original[target]).tolist()

    if encoding == "trainid":
        if not set(ids).issubset(TRAINID_TO_MERGED):
            raise ValueError(f"trainIds non reconnus : {ids}")
        ids = sorted({TRAINID_TO_MERGED[value] for value in ids})
    elif encoding != "merged":
        raise ValueError("original_mask_encoding doit valoir 'trainid' ou 'merged'.")

    if not set(ids).issubset(CLASS_NAMES):
        raise ValueError(f"Classes fusionnées non reconnues : {ids}")

    return [CLASS_NAMES[value] for value in ids]


def load_sample(path, root, config):
    """Retourne l'image CHW [0,1], la référence binaire et les informations.

    Image et cible sont redimensionnées ensemble à la résolution du config.
    Les classes support sont déterminées avant ce redimensionnement.
    Les métadonnées sont conservées pour images.csv, pas pour per_image.csv.
    """
    folder = path.parent
    native_target = read_mask(folder / "laser_mask_alone.png", binary=True)
    original_mask = read_mask(folder / "original_mask.png")

    supports = support_classes(
        original_mask,
        native_target,
        config.original_mask_encoding,
    )

    size = (config.input_size_w, config.input_size_h)

    with Image.open(path) as image:
        image = image.convert("RGB")
        native_width, native_height = image.size

        if native_target.shape != (native_height, native_width):
            raise ValueError(f"{folder} : l'image et le masque sont désalignés.")

        resized = image.resize(size, Image.Resampling.BICUBIC)
        image_array = np.asarray(resized).copy()

    target_image = Image.fromarray(native_target.astype(np.uint8))
    target_image = target_image.resize(size, Image.Resampling.NEAREST)
    target = np.asarray(target_image).astype(bool)

    metadata_path = folder / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    # Aplatit les dictionnaires, mais conserve les listes dans une seule cellule.
    flat_metadata = (
        pd.json_normalize(metadata, sep=".").iloc[0].to_dict()
        if metadata else {}
    )

    info = {
        "sample_id": folder.relative_to(root).as_posix(),
        "support_classes": supports,
        "class_txt": (folder / "class.txt").read_text(encoding="utf-8").strip(),
        "native_width": native_width,
        "native_height": native_height,
        "eval_width": size[0],
        "eval_height": size[1],
        "gt_pixels_native": int(native_target.sum()),
        "gt_pixels": int(target.sum()),
        "gt_disappeared_after_resize": bool(native_target.any() and not target.any()),
        **{f"meta.{key}": value for key, value in flat_metadata.items()},
    }

    tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255
    return tensor, target, info


def write_csv(frame, path):
    """Sauvegarde un tableau, en encodant explicitement les listes en JSON.

    Le tableau d'origine n'est pas modifié : les classes restent des listes
    en mémoire pour les regroupements statistiques.
    """
    frame = frame.copy()

    def serialize(value):
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)
        return value

    for column in frame:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(serialize)

    frame.to_csv(path, index=False)


def subplot_filename(sample_id):
    """Construit un nom plat à partir du chemin relatif de l'échantillon.

    Exemple :
        city/image_001 -> city__image_001.png

    Le fichier source s'appelant toujours laser.png, son nom seul
    ne permet pas de distinguer les échantillons.
    """
    parts = Path(sample_id).parts

    if not parts:
        raise ValueError("sample_id ne doit pas être vide.")

    # Double les underscores existants pour limiter les ambiguïtés
    # entre un séparateur de dossiers et le nom d'un dossier.
    encoded = [part.replace("_", "_u") for part in parts]
    return "__".join(encoded) + ".png"


def save_prediction_subplot(image_path, panels, output, sample_id):
    """Sauvegarde l'image originale et l'union des masques par prompt.

    panels contient des tuples :
        (nom_du_prompt, masque_fusionne, nombre_de_predictions)

    Aucun masque individuel ni sous-dossier par échantillon n'est créé.
    """
    with Image.open(image_path) as source:
        original = np.asarray(source.convert("RGB")).copy()

    n_panels = 1 + len(panels)
    n_columns = min(3, n_panels)
    n_rows = ceil(n_panels / n_columns)

    figure = Figure(
        figsize=(6 * n_columns, 3.8 * n_rows),
        constrained_layout=True,
    )
    FigureCanvasAgg(figure)

    axes = figure.subplots(
        n_rows,
        n_columns,
        squeeze=False,
    ).ravel()

    axes[0].imshow(original)
    axes[0].set_title("Image originale avec laser", fontsize=12)

    for axis, (prompt, mask, n_predictions) in zip(axes[1:], panels):
        if torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy()

        mask = np.asarray(mask, dtype=bool)

        if mask.ndim != 2:
            raise ValueError(
                f"{prompt!r} : un masque fusionné 2D est attendu, "
                f"reçu {mask.shape}."
            )

        axis.imshow(
            mask,
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )

        label = "prédiction" if n_predictions == 1 else "prédictions"
        axis.set_title(
            f"{prompt} — {n_predictions} {label}",
            fontsize=12,
            wrap=True,
        )

    for axis in axes:
        axis.axis("off")

    relative_sample = Path(sample_id)

    destination = (
            Path(output)
            / "masks"
            / relative_sample.parent
            / f"{relative_sample.name}.png"
    )

    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        figure.savefig(
            destination,
            dpi=150,
            facecolor="white",
        )
    finally:
        figure.clear()


def evaluate(config, detector):
    """Exécute l'évaluation et retourne le tableau des prédictions.

    Configuration attendue
    ----------------------
    data_path_synthetic, split, output, limit, prompts, metrics,
    input_size_w, input_size_h, original_mask_encoding, save_masks.

    Le détecteur doit exposer predict(image_tensor, prompt=...).
    La confiance est réglée dans le détecteur, pas dans cette fonction.
    """
    root = Path(config.data_path_synthetic) / config.split
    output = Path(config.output)
    paths = sorted(root.rglob("laser.png"))

    if config.limit is not None:
        if config.limit < 1:
            raise ValueError("limit doit être >= 1 ou None.")
        paths = paths[:config.limit]

    if not paths:
        raise FileNotFoundError(f"Aucune image laser.png dans {root}")

    if not config.prompts or len(set(config.prompts)) != len(config.prompts):
        raise ValueError("La liste de prompts doit être non vide, sans doublons.")

    if min(config.input_size_w, config.input_size_h) <= 0:
        raise ValueError("Les dimensions d'évaluation doivent être positives.")

    output.mkdir(parents=True, exist_ok=True)

    metrics = LaserMetrics(config.metrics)
    image_rows = []
    prediction_rows = []

    for index, path in enumerate(paths):
        image, target, info = load_sample(path, root, config)
        image_rows.append(info)

        panels = []

        for prompt in config.prompts:
            result = detector.predict(image, prompt=prompt)

            values, mask = metrics.update(
                result,
                target,
                prompt,
                info["support_classes"],
            )

            n_predictions = len(result["scores"])

            prediction_rows.append({
                "sample_id": info["sample_id"],
                "prompt": prompt,
                "support_classes": info["support_classes"],
                "n_predictions": n_predictions,
                **values,
            })

            if config.save_masks:
                # Copie indépendante : on conserve uniquement le masque fusionné,
                # pas l'ensemble des instances de chaque prompt.
                if torch.is_tensor(mask):
                    display_mask = (
                        mask.detach().cpu().numpy().astype(bool, copy=True)
                    )
                else:
                    display_mask = np.array(mask, dtype=bool, copy=True)

                panels.append((
                    str(prompt),
                    display_mask,
                    n_predictions,
                ))

            description = ", ".join(
                f"{name}={value:.3f}"
                for name, value in values.items()
            )

            print(
                f"[{index + 1}/{len(paths)}] "
                f"{prompt} : {description}"
            )

        # Une seule figure par image, une fois tous les prompts traités.
        if config.save_masks:
            save_prediction_subplot(
                image_path=path,
                panels=panels,
                output=output,
                sample_id=info["sample_id"],
            )

        # Checkpoint après chaque image complète.
        # Ne permet pas la reprise des états internes de la mAP.
        write_csv(
            pd.DataFrame(image_rows),
            output / "images.csv",
        )
        write_csv(
            pd.DataFrame(prediction_rows),
            output / "per_image.csv",
        )

    predictions = pd.DataFrame(prediction_rows)
    tables = metrics.tables(predictions)

    for name, table in tables.items():
        write_csv(table, output / f"{name}.csv")

    return predictions
