"""Script qui calcule des métriques de comparaison entre les détections des modèles et la vérité terrain."""
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import jaccard_score, f1_score, recall_score, precision_score


def _binarize(mask):
    """Convertit un masque {0, 255}, {0, 1} ou booléen en {0, 1}"""
    mask = np.asarray(mask)
    if mask.dtype == bool:
        return mask.astype(np.uint8)
    return (mask > 0).astype(np.uint8)


def compute_IoU(image_true, image_pred):
    """IoU (Intersection over Union) entre GT et prédiction."""
    return jaccard_score(
        _binarize(image_true).ravel(),
        _binarize(image_pred).ravel(),
        pos_label=1,
        zero_division=0,
    )


def compute_dice(image_true, image_pred):
    """Dice (F1) entre GT et prédiction."""
    return f1_score(
        _binarize(image_true).ravel(),
        _binarize(image_pred).ravel(),
        pos_label=1,
        zero_division=0,
    )


def compute_recall(image_true, image_pred):
    """Recall : proportion du laser réel détectée."""
    return recall_score(
        _binarize(image_true).ravel(),
        _binarize(image_pred).ravel(),
        pos_label=1,
        zero_division=0,
    )


def compute_precision(image_true, image_pred):
    """Precision : proportion des détections qui sont correctes."""
    return precision_score(
        _binarize(image_true).ravel(),
        _binarize(image_pred).ravel(),
        pos_label=1,
        zero_division=0,
    )


def compute_map_at_iou(image_true, image_pred, iou_thresholds=None):
    """mAP@[0.5:0.95] : moyenne des détections réussies à différents seuils IoU.

    Pour un masque binaire et une seule prédiction, on compte un succès
    (AP=1) à chaque seuil où l'IoU global dépasse le seuil, sinon échec
    (AP=0). La mAP est la moyenne des AP sur les seuils.

    Usage typique : évaluation objet (laser détecté, oui ou non).
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
    iou = compute_IoU(image_true, image_pred)
    hits = [1.0 if iou >= t else 0.0 for t in iou_thresholds]
    return float(np.mean(hits))


def compute_map_from_prob(image_true, image_prob, num_thresholds=101):
    """mAP pixel-level via la courbe Precision-Recall sur une carte de probabilités.

    Plus fine que `compute_map_at_iou`, mais nécessite une sortie continue
    (probabilités, logits sigmoïde, ou masques pondérés par leur score).

    Args:
        image_true : masque GT binaire (0/255 ou 0/1)
        image_prob : carte de probabilités float dans [0, 1], même shape
        num_thresholds : nombre de points sur la courbe PR
    """
    gt = _binarize(image_true).ravel().astype(bool)
    prob = np.asarray(image_prob).ravel()

    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    precisions = np.zeros(num_thresholds)
    recalls = np.zeros(num_thresholds)

    for i, t in enumerate(thresholds):
        pred = prob >= t
        tp = np.sum(pred & gt)
        fp = np.sum(pred & ~gt)
        fn = np.sum(~pred & gt)
        precisions[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recalls[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Courbe PR triée par recall croissant, puis aire sous la courbe
    order = np.argsort(recalls)
    recalls = recalls[order]
    precisions = precisions[order]
    return float(np.trapz(precisions, recalls))


def compute_all_metrics(image_true, image_pred, image_prob=None):
    """Calcule les 5 métriques en un seul appel."""
    metrics = {
        "iou": compute_IoU(image_true, image_pred),
        "dice": compute_dice(image_true, image_pred),
        "recall": compute_recall(image_true, image_pred),
        "precision": compute_precision(image_true, image_pred),
    }
    if image_prob is not None:
        metrics["map"] = compute_map_from_prob(image_true, image_prob)
    else:
        metrics["map"] = compute_map_at_iou(image_true, image_pred)
    return metrics