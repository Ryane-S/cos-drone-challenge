"""Script qui calcule des métriques de comparaison entre les détections des modèles et la vérité terrain."""
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import jaccard_score, f1_score

def compute_IoU(image_true, image_pred):
    """Calcule l'IoU entre le groundtruth et la détection."""
    return jaccard_score(image_true, image_pred)

def compute_dice(image_true, image_pred):
    """Calcule le Dice entre le groundtruth et la détection."""
    return f1_score(image_true, image_pred)

def generate_binary_laser_mask(mask_gt_path: Path):
    mask_gt = np.array(Image.open(mask_gt_path))
    mask_gt[mask_gt != 19] = 0
    mask_gt[mask_gt == 19] = 255
    Image.fromarray(mask_gt).save(mask_gt_path)
    return mask_gt

generate_binary_laser_mask("samples/frankfurt_000000_000576_gtFine_labelTrainIds.png")
