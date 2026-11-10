"""Métriques binaires par pixels, sans dépendance sklearn."""
import numpy as np

AVAILABLE = ('iou', 'dice', 'precision', 'recall')


def binary_metrics(target, prediction, metrics=AVAILABLE):
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.ndim != 2 or target.shape != prediction.shape:
        raise ValueError('Deux masques 2D de mêmes dimensions sont requis.')
    if not set(metrics) <= set(AVAILABLE):
        raise ValueError(f'Métriques disponibles : {AVAILABLE}')
    target, prediction = target.astype(bool), prediction.astype(bool)
    tp = int(np.count_nonzero(target & prediction))
    fp = int(np.count_nonzero(~target & prediction))
    fn = int(np.count_nonzero(target & ~prediction))
    tn = int(np.count_nonzero(~target & ~prediction))
    # Convention explicite : tout dénominateur nul donne NaN.
    div = lambda n, d: n / d if d else float('nan')
    values = dict(iou=div(tp, tp+fp+fn), dice=div(2*tp, 2*tp+fp+fn),
                  precision=div(tp, tp+fp), recall=div(tp, tp+fn))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                gt_pixels=tp+fn, pred_pixels=tp+fp,
                **{name: values[name] for name in metrics})
