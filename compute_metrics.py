"""Script qui calcule des métriques de comparaison entre les détections des modèles et la vérité terrain."""

from sklearn.metrics import jaccard_score, f1_score

def compute_IoU(image_true, image_pred):
    """Calcule l'IoU entre le groundtruth et la détection."""
    return jaccard_score(image_true, image_pred)

def compute_dice(image_true, image_pred):
    """Calcule le Dice entre le groundtruth et la détection."""
    return f1_score(image_true, image_pred)