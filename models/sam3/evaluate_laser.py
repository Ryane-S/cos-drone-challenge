import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from models.sam3.laser_metrics import AVAILABLE, binary_metrics

MERGED = {0:0, 1:0, 2:1, 3:1, 4:1, 5:2, 6:2, 7:2, 8:3, 9:3,
          10:4, 11:5, 12:5, 13:6, 14:6, 15:6, 16:6, 17:6, 18:6,
          19:19, 255:255}
NAMES = {0:'flat', 1:'construction', 2:'urban_object', 3:'nature',
         4:'sky', 5:'human', 6:'vehicle', 255:'rest'}


def read_mask(path, binary=False):
    with Image.open(path) as image:
        mask = np.asarray(image).copy()
    if mask.ndim == 3:
        if not binary or mask.shape[2] not in (3, 4):
            raise ValueError(f'{path}: masque de classes indexé 2D requis, pas une image colorée.')
        if not np.all(mask[..., :3] == mask[..., :1]):
            raise ValueError(f'{path}: masque binaire RGB non gris.')
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(f'{path}: masque 2D requis.')
    if binary:
        if not set(np.unique(mask).tolist()) <= {0, 1, 255}:
            raise ValueError(f'{path}: valeurs binaires attendues (0/1 ou 0/255).')
        return mask > 0
    return mask


def support_classes(original, target, encoding):
    """Retourne les classes fusionnées réellement recouvertes par le laser."""

    if original.shape != target.shape:
        raise ValueError(
            "original_mask et laser_mask_alone doivent être alignés."
        )

    if encoding not in {"trainid", "merged"}:
        raise ValueError(
            f"Encodage original_mask invalide : {encoding!r}. "
            "Valeurs acceptées : 'trainid' ou 'merged'."
        )

    ids = np.unique(original[target]).tolist()

    if encoding == "trainid":
        unknown = sorted(set(ids) - set(MERGED))

        if unknown:
            raise ValueError(
                f"trainIds inconnus dans original_mask : {unknown}."
            )

        ids = sorted({MERGED[i] for i in ids})

    invalid = sorted(set(ids) - set(NAMES))

    if invalid:
        raise ValueError(
            f"Classes support invalides : {invalid}. "
            f"Encodage déclaré : {encoding!r}. "
            "Vérifier que l'encodage déclaré correspond au masque sauvegardé."
        )

    return [NAMES[i] for i in ids]


def laser_type(metadata):
    # STUB : ajouter ici ta nomenclature réelle des types de lasers.
    # Ne pas inventer un type à partir de paramètres continus sans convention.
    return str(metadata.get('laser_type') or metadata.get('type') or 'unspecified')


def flatten_metadata(value, prefix='meta'):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if key != 'contour':
                result.update(flatten_metadata(item, f'{prefix}.{key}'))
    elif isinstance(value, list):
        # Vecteurs courts (position, couleur...), pas les grands contours.
        for i, item in enumerate(value):
            result.update(flatten_metadata(item, f'{prefix}.{i}'))
    else:
        result[prefix] = value
    return result


def merge_masks(masks, shape):
    if masks is None:
        return np.zeros(shape, dtype=bool)
    masks = torch.as_tensor(masks).detach().cpu()
    if masks.numel() == 0:
        return np.zeros(shape, dtype=bool)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or tuple(masks.shape[-2:]) != tuple(shape):
        raise ValueError(f'Masques prédits : {tuple(masks.shape)}, attendu (N, {shape}).')
    if masks.dtype != torch.bool and not torch.all((masks == 0) | (masks == 1)):
        raise ValueError('Les masques SAM doivent être binaires après post-traitement.')
    return masks.bool().any(dim=0).numpy()


def summary(frame, keys, metrics):
    rows = []
    for group, subset in frame.groupby(keys, dropna=False, sort=True):
        if not isinstance(group, tuple):
            group = (group,)
        row = dict(zip(keys, group))
        row.update(n_images=len(subset),
                   n_no_prediction=int((subset.n_predictions == 0).sum()),
                   n_empty_gt=int((subset.gt_pixels == 0).sum()))
        for metric in metrics:
            row[f'{metric}_mean'] = subset[metric].mean()
            row[f'{metric}_std'] = subset[metric].std()
            row[f'{metric}_n_valid'] = int(subset[metric].count())
        tp, fp, fn = (int(subset[c].sum()) for c in ('tp', 'fp', 'fn'))
        denominators = dict(iou=tp+fp+fn, dice=2*tp+fp+fn,
                            precision=tp+fp, recall=tp+fn)
        for metric in metrics:
            numerator = 2*tp if metric == 'dice' else tp
            denominator = denominators[metric]
            row[f'{metric}_micro'] = numerator/denominator if denominator else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def export_results(rows, output, metrics):
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'per_image.csv', index=False)
    summary(frame, ['prompt'], metrics).to_csv(output / 'by_prompt.csv', index=False)
    summary(frame, ['prompt', 'laser_type'], metrics).to_csv(output / 'by_laser_type.csv', index=False)
    expanded = frame.copy()
    expanded['support_class'] = expanded.support_classes.map(json.loads)
    expanded = expanded.explode('support_class').dropna(subset=['support_class'])
    if not expanded.empty:
        summary(expanded, ['prompt', 'support_class'], metrics).to_csv(
            output / 'by_support_class.csv', index=False)
        summary(expanded, ['prompt', 'support_class', 'laser_type'], metrics).to_csv(
            output / 'by_support_and_type.csv', index=False)


def evaluate(config, detector):
    root = config.data_path_synthetic / config.split
    paths = sorted(root.rglob('laser.png'))

    if config.limit is not None:
        paths = paths[:config.limit]

    if not paths:
        raise FileNotFoundError(f'Aucune image laser.png dans {root}')

    config.output.mkdir(parents=True, exist_ok=True)

    # config = vars(config).copy()
    # config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    # config.update(empty_denominator='NaN',
    #               fusion='union of all accepted masks', scope='all image pixels',
    #               torch_version=torch.__version__)
    # (config.output / 'evaluation_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

    rows = []
    for index, path in enumerate(paths):
        folder = path.parent

        with Image.open(path) as image:
            image = image.convert('RGB')
            native_size = image.size
            target_native = read_mask(folder / 'laser_mask_alone.png', binary=True)

            if target_native.shape != (native_size[1], native_size[0]):
                raise ValueError(f'{folder}: image et GT non alignés.')
            print(folder)
            supports = support_classes(read_mask(folder / 'original_mask.png'),
                                       target_native, config.original_mask_encoding)
            size = config.input_size_w, config.input_size_h
            image_array = np.asarray(image.resize(size, Image.Resampling.BICUBIC)).copy()

        target = np.asarray(Image.fromarray(target_native.astype(np.uint8)).resize(
            size, Image.Resampling.NEAREST)
        ).astype(bool)

        metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        classes_text = (folder / 'class.txt').read_text(encoding='utf-8').strip()

        tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255
        for prompt_index, prompt in enumerate(config.prompts):
            result = detector.predict(tensor, prompt=prompt)
            pred = merge_masks(result['masks'], target.shape)
            values = binary_metrics(target, pred, config.metrics)

            rows.append(dict(
                sample_id=folder.relative_to(root).as_posix(),
                prompt=prompt,
                support_classes=json.dumps(supports),
                class_txt=classes_text,
                laser_type=laser_type(metadata), n_predictions=len(result['scores']),
                gt_pixels_native=int(target_native.sum()),
                gt_disappeared_after_resize=bool(target_native.any() and not target.any()),
                height=target.shape[0], width=target.shape[1],
                **values, **flatten_metadata(metadata)
            ))

            if config.save_masks:
                destination = config.output / 'masks' / folder.relative_to(root)
                destination.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pred.astype(np.uint8)*255).save(destination / f'prompt_{prompt_index:02d}.png')

            print(f'[{index+1}/{len(paths)}] {prompt}: '
                  + ', '.join(f'{m}={values[m]:.3f}' for m in config.metrics))
        # Checkpoint
        pd.DataFrame(rows).to_csv(config.output / 'per_image.csv', index=False)

    export_results(rows, config.output, config.metrics)
    return pd.DataFrame(rows)
