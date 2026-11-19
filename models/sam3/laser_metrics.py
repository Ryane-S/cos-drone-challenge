"""Calcul et agrégation des métriques de segmentation laser.

Métriques pixel : union de toutes les instances prédites.
Mask mAP : instances séparées, avec leurs scores de confiance.

Conventions
-----------
- Un dénominateur nul produit NaN.
- Les moyennes ignorent les NaN ; std utilise ddof=1.
- Pour la mAP, chaque image contient au maximum un laser réel.
- Les classes support définissent des sous-ensembles d'images,
  pas les catégories d'objets détectées (une seule : laser).
"""

import numpy as np
import pandas as pd
import torch


PIXEL_METRICS = ("iou", "dice", "precision", "recall")
AVAILABLE = (*PIXEL_METRICS, "map")


def binary_metrics(target, prediction, metrics=PIXEL_METRICS):
    """Calcule les métriques sélectionnées sur deux masques 2D booléens.

    Retourne uniquement les noms et valeurs des métriques demandées.
    Aucune moyenne entre images n'est effectuée ici.
    """
    target = np.asarray(target)
    prediction = np.asarray(prediction)

    if target.ndim != 2 or target.shape != prediction.shape:
        raise ValueError("Deux masques 2D de mêmes dimensions sont requis.")

    if not set(metrics).issubset(PIXEL_METRICS):
        raise ValueError(f"Métriques pixel disponibles : {PIXEL_METRICS}")

    target = target.astype(bool)
    prediction = prediction.astype(bool)

    true_positive = np.count_nonzero(target & prediction)
    false_positive = np.count_nonzero(~target & prediction)
    false_negative = np.count_nonzero(target & ~prediction)

    def divide(numerator, denominator):
        if denominator == 0:
            return float("nan")
        return float(numerator / denominator)

    values = {
        "iou": divide(
            true_positive,
            true_positive + false_positive + false_negative,
        ),
        "dice": divide(
            2 * true_positive,
            2 * true_positive + false_positive + false_negative,
        ),
        "precision": divide(
            true_positive,
            true_positive + false_positive,
        ),
        "recall": divide(
            true_positive,
            true_positive + false_negative,
        ),
    }

    return {name: values[name] for name in metrics}


def prediction_masks(result, shape):
    """Normalise les masques SAM en tenseur booléen CPU (N, H, W).

    Accepte les formes (N, H, W), (N, 1, H, W) et l'absence de masques.
    Vérifie également la correspondance entre masques et scores.
    """
    masks = result["masks"]

    if masks is None:
        masks = torch.empty((0, *shape), dtype=torch.bool)
    else:
        masks = torch.as_tensor(masks).detach().cpu()

    if masks.numel() == 0:
        masks = torch.empty((0, *shape), dtype=torch.bool)

    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    if masks.ndim != 3 or tuple(masks.shape[1:]) != tuple(shape):
        raise ValueError(
            f"Masques SAM mal dimensionnés : {tuple(masks.shape)}. "
            f"Forme attendue : (N, {shape[0]}, {shape[1]})."
        )

    if not torch.all((masks == 0) | (masks == 1)):
        raise ValueError("Des masques SAM binaires sont requis.")

    if len(masks) != len(result["scores"]):
        raise ValueError("Un score est requis pour chaque masque prédit.")

    return masks.bool().contiguous()


def summarize(frame, keys, metrics):
    """Résume les prédictions par prompt ou par prompt et classe support.

    n_images : nombre de lignes image/prompt du groupe.
    n_no_prediction : nombre de lignes sans instance prédite.
    *_mean, *_std : statistiques par image, hors valeurs NaN.
    """
    columns = [*keys, "n_images", "n_no_prediction"]
    columns += [f"{name}_{stat}" for name in metrics for stat in ("mean", "std")]
    rows = []

    # Une seule clé passée comme scalaire évite les différences de versions pandas.
    group_keys = keys[0] if len(keys) == 1 else keys

    for group, subset in frame.groupby(group_keys, dropna=False, sort=True):
        if len(keys) == 1:
            group = (group,)

        row = dict(zip(keys, group))
        row["n_images"] = len(subset)
        row["n_no_prediction"] = int((subset["n_predictions"] == 0).sum())

        for name in metrics:
            row[f"{name}_mean"] = subset[name].mean()
            row[f"{name}_std"] = subset[name].std(ddof=1)

        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


class LaserMetrics:
    """Centralise les calculs de toute une évaluation.

    Exemple
    -------
    evaluator = LaserMetrics(["iou", "dice", "map"])
    values, merged_mask = evaluator.update(result, target, prompt, supports)
    tables = evaluator.tables(predictions_dataframe)

    tables() se lance une seule fois à la fin de la boucle d'inférence.
    TorchMetrics est importé uniquement si la mAP est activée.
    """

    def __init__(self, metrics=PIXEL_METRICS):
        if not metrics or len(set(metrics)) != len(metrics):
            raise ValueError("Choisir une liste non vide de métriques distinctes.")

        if not set(metrics).issubset(AVAILABLE):
            raise ValueError(f"Métriques disponibles : {AVAILABLE}")

        self.pixel_metrics = tuple(name for name in metrics if name != "map")
        self.use_map = "map" in metrics
        self.map_groups = {}

    def update(self, result, target, prompt, supports):
        """Évalue une image pour un prompt et accumule ses instances pour la mAP.

        Parameters
        ----------
        result : dict
            Sortie du détecteur, contenant masks et scores.
        target : ndarray bool de forme (H, W)
            Vérité terrain binaire du laser.
        prompt : str
            Texte utilisé pour cette prédiction.
        supports : list[str]
            Classes Cityscapes recouvertes par le laser réel.

        Returns
        -------
        values : dict
            Métriques pixel sélectionnées pour cette image.
        merged : ndarray bool
            Union des instances prédites, utile pour la sauvegarde.
        """
        target = np.asarray(target, dtype=bool)
        if target.ndim != 2:
            raise ValueError("La vérité terrain doit être un masque 2D.")

        masks = prediction_masks(result, target.shape)
        merged = masks.any(dim=0).numpy()
        values = binary_metrics(target, merged, self.pixel_metrics)

        if self.use_map:
            scores = torch.as_tensor(result["scores"])
            scores = scores.detach().cpu().float().reshape(-1)

            if not torch.isfinite(scores).all():
                raise ValueError("Les scores contiennent des valeurs non finies.")

            self._update_map(masks, scores, target, prompt, supports)

        return values, merged

    def _update_map(self, masks, scores, target, prompt, supports):
        """Accumule les instances pour le prompt et chaque classe support."""
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

        target_tensor = torch.from_numpy(target.copy())

        if target.any():
            target_masks = target_tensor.unsqueeze(0)
        else:
            target_masks = torch.empty((0, *target.shape), dtype=torch.bool)

        prediction = {
            "masks": masks,
            "scores": scores,
            "labels": torch.zeros(len(masks), dtype=torch.int64),
        }
        reference = {
            "masks": target_masks,
            "labels": torch.zeros(len(target_masks), dtype=torch.int64),
        }

        groups = [("prompt", prompt, "")]
        groups += [("class", prompt, name) for name in sorted(set(supports))]

        for key in groups:
            if key not in self.map_groups:
                self.map_groups[key] = MeanAveragePrecision(
                    iou_type="segm",
                    backend="pycocotools",
                )

            self.map_groups[key].update([prediction], [reference])

    def tables(self, predictions):
        """Produit les synthèses par prompt et par classe support.

        Les métriques pixel sont résumées par mean et std.
        Si la mAP est activée, ses résultats sont ajoutés aux mêmes tableaux.

        Une image multiclasse contribue à chaque classe support concernée.
        """
        expanded = predictions.explode("support_classes")
        expanded = expanded.rename(
            columns={"support_classes": "support_class"}
        )
        expanded = expanded.dropna(subset=["support_class"])

        by_prompt = summarize(
            predictions,
            ["prompt"],
            self.pixel_metrics,
        )

        by_class = summarize(
            expanded,
            ["prompt", "support_class"],
            self.pixel_metrics,
        )

        if self.use_map:
            map_results = self._map_table(predictions, expanded)

            # On sélectionne uniquement les clés et les valeurs mAP.
            # Les effectifs sont déjà présents dans les synthèses.
            map_columns = ["map", "map_50", "map_75"]

            prompt_map = map_results.loc[
                map_results["group"] == "prompt",
                ["prompt", *map_columns],
            ]

            class_map = map_results.loc[
                map_results["group"] == "class",
                ["prompt", "support_class", *map_columns],
            ]

            by_prompt = by_prompt.merge(
                prompt_map,
                on="prompt",
                how="left",
                validate="one_to_one",
            )

            by_class = by_class.merge(
                class_map,
                on=["prompt", "support_class"],
                how="left",
                validate="one_to_one",
            )

        return {
            "by_prompt": by_prompt,
            "by_class": by_class,
        }

    def _map_table(self, predictions, expanded):
        """Calcule la mAP de chaque ensemble, sans moyenne artificielle par image."""
        rows = []

        for (group, prompt, support), metric in self.map_groups.items():
            values = metric.compute()

            if group == "prompt":
                subset = predictions[predictions["prompt"] == prompt]
            else:
                subset = expanded[
                    (expanded["prompt"] == prompt)
                    & (expanded["support_class"] == support)
                ]

            row = {
                "group": group,
                "prompt": prompt,
                "support_class": support,
                "n_images": len(subset),
                "n_no_prediction": int((subset["n_predictions"] == 0).sum()),
            }

            for name in ("map", "map_50", "map_75"):
                value = float(values[name])
                row[name] = value if value >= 0 else np.nan

            rows.append(row)
            metric.reset()

        self.map_groups.clear()
        return pd.DataFrame(rows)
