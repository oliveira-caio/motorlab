import warnings

import numpy as np

from sklearn.metrics import balanced_accuracy_score


def compute(gts: dict, preds: dict, metric: str):
    if metric == "accuracy":
        return np.mean(
            [
                accuracy(gt, pred)
                for gt, pred in zip(gts.values(), preds.values())
            ]
        )
    elif metric == "mse":
        return np.mean(
            [mse(gt, pred) for gt, pred in zip(gts.values(), preds.values())]
        )
    elif metric == "correlation":
        return np.mean(
            [
                global_correlation(gt, pred)
                for gt, pred in zip(gts.values(), preds.values())
            ]
        ), np.mean(
            [
                local_correlation(gt, pred)
                for gt, pred in zip(gts.values(), preds.values())
            ]
        )
    else:
        raise ValueError(f"metric {metric} not implemented.")


def mse(gt: np.ndarray, pred: np.ndarray) -> float:
    return np.mean((gt - pred) ** 2).item()


def balanced_accuracy(gt: np.ndarray, pred: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="y_pred contains classes not in y_true",
            category=UserWarning,
            module="sklearn.metrics._classification",
        )
        return balanced_accuracy_score(gt, pred)


def accuracy(gt: np.ndarray, pred: np.ndarray) -> float:
    pred = pred.reshape(-1, pred.shape[-1])
    pred = pred.argmax(axis=-1)
    gt = gt.reshape(-1)
    return balanced_accuracy(gt, pred)


def global_correlation(gt: np.ndarray, pred: np.ndarray, reduce: bool = True):
    # gt, pred: (batch_size, seq_len, n_features)
    pred = pred.reshape(-1, pred.shape[-1])
    gt = gt.reshape(-1, gt.shape[-1])

    pred_centered = pred - np.nanmean(pred, axis=0)
    gt_centered = gt - np.nanmean(gt, axis=0)

    num = np.nansum(pred_centered * gt_centered, axis=0)
    den = np.sqrt(
        np.nansum(pred_centered**2, axis=0) * np.nansum(gt_centered**2, axis=0)
    )

    global_corr = num / den

    if reduce:
        global_corr = np.nanmean(global_corr).item()

    return global_corr


def local_correlation(gt: np.ndarray, pred: np.ndarray, reduce: bool = True):
    # gt, pred: (batch_size, seq_len, n_features)
    gt_centered = gt - np.nanmean(gt, axis=1, keepdims=True)
    pred_centered = pred - np.nanmean(pred, axis=1, keepdims=True)

    numerator = np.nansum(pred_centered * gt_centered, axis=1)
    denom = np.sqrt(
        np.nansum(pred_centered**2, axis=1) * np.nansum(gt_centered**2, axis=1)
    )

    corr = numerator / denom
    corr = np.where(denom != 0, corr, np.nan)

    local_corr = np.nanmean(corr, axis=0)  # (n_features,)

    if reduce:
        local_corr = np.nanmean(local_corr).item()

    return local_corr
