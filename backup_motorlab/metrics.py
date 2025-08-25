import warnings

import numpy as np

from sklearn.metrics import balanced_accuracy_score


def compute(gts: dict, preds: dict, metrics: dict):
    """
    Compute metrics over all sessions for each modality.

    Parameters
    ----------
    gts : dict
        Ground truth data structured as {session: {modality: np.ndarray}}.
    preds : dict
        Prediction data structured as {session: {modality: np.ndarray}}.
    metrics : dict
        Metrics to compute for each modality {modality: metric_name}.

    Returns
    -------
    dict
        Computed metric values {metric_name: value} where correlation is
        split into 'global_corr' and 'local_corr'.
    """
    results = {}

    for modality, metric_name in metrics.items():
        gt_arrays = [gts[session][modality] for session in gts.keys()]
        pred_arrays = [preds[session][modality] for session in preds.keys()]

        if metric_name == "accuracy":
            results[metric_name] = np.mean(
                [accuracy(gt, pred) for gt, pred in zip(gt_arrays, pred_arrays)]
            )
        elif metric_name == "mse":
            results[metric_name] = np.mean(
                [mse(gt, pred) for gt, pred in zip(gt_arrays, pred_arrays)]
            )
        elif metric_name == "correlation":
            global_corrs = []
            local_corrs = []
            for gt, pred in zip(gt_arrays, pred_arrays):
                global_corr = global_correlation(gt, pred)
                local_corr = local_correlation(gt, pred)
                global_corrs.append(global_corr)
                local_corrs.append(local_corr)

            results["global_corr"] = np.mean(global_corrs)
            results["local_corr"] = np.mean(local_corrs)
        else:
            raise ValueError(f"metric {metric_name} not implemented.")

    return results


def mse(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Compute mean squared error (MSE) between ground truth and predictions.

    Parameters
    ----------
    gt : np.ndarray
        Ground truth array.
    pred : np.ndarray
        Prediction array.

    Returns
    -------
    float
        Mean squared error.
    """
    return np.mean((gt - pred) ** 2).item()


def balanced_accuracy(
    gt: np.ndarray,
    pred: np.ndarray,
    group_by: str = "",
    include_sitting: bool = True,
) -> float:
    """
    Compute balanced accuracy between ground truth and predictions.

    Parameters
    ----------
    gt : np.ndarray
        Ground truth labels.
    pred : np.ndarray
        Predicted labels.
    group_by : str, optional
        If group_by = "x", computes balanced accuracy for x-axis predictions only.
        If group_by = "y", computes balanced accuracy for y-axis predictions only.
    include_sitting : bool, optional
        Whether to include the tiles in which the monkey is sitting (0, 1, 2, 12, 13, 14).

    Returns
    -------
    float
        Balanced accuracy score.
    """
    if not include_sitting:
        non_sitting_indices = ~np.isin(gt, [0, 1, 2, 12, 13, 14])
        gt = gt[non_sitting_indices]
        pred = pred[non_sitting_indices]

    if group_by == "x":
        # collapses the y-axis such that all y values are grouped by x.
        gt = gt % 3
        pred = pred % 3
    elif group_by == "y":
        # collapses the x-axis such that all x values are grouped by y.
        gt = gt // 3
        pred = pred // 3

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="y_pred contains classes not in y_true",
            category=UserWarning,
            module="sklearn.metrics._classification",
        )
        return balanced_accuracy_score(gt, pred)


def accuracy(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Compute accuracy (balanced accuracy) for classification predictions.

    Parameters
    ----------
    gt : np.ndarray
        Ground truth labels.
    pred : np.ndarray
        Prediction logits or probabilities.

    Returns
    -------
    float
        Balanced accuracy score.
    """
    pred = pred.reshape(-1, pred.shape[-1])
    pred = pred.argmax(axis=-1)
    gt = gt.reshape(-1)
    return balanced_accuracy(gt, pred)


def global_correlation(gt: np.ndarray, pred: np.ndarray, reduce: bool = True):
    """
    Compute global correlation between ground truth and predictions.

    Parameters
    ----------
    gt : np.ndarray
        Ground truth array.
    pred : np.ndarray
        Prediction array.
    reduce : bool, optional
        Whether to average over features. Default is True.

    Returns
    -------
    float or np.ndarray
        Global correlation value(s).
    """
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
    """
    Compute local (per-frame) correlation between ground truth and predictions.

    Parameters
    ----------
    gt : np.ndarray
        Ground truth array.
    pred : np.ndarray
        Prediction array.
    reduce : bool, optional
        Whether to average over features. Default is True.

    Returns
    -------
    float or np.ndarray
        Local correlation value(s).
    """
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
