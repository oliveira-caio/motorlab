import warnings

import numpy as np
import torch

from sklearn.metrics import balanced_accuracy_score


class MultiModalLoss(torch.nn.Module):
    def __init__(self, names: dict, weights: dict = dict()):
        super().__init__()
        registry = {
            "closed_mse": ClosedFormMSE(),
            "mse": torch.nn.MSELoss(),
            "poisson": torch.nn.PoissonNLLLoss(log_input=False, full=True),
        }
        self.modalities = list(names.keys())
        self.losses = torch.nn.ModuleDict(
            {modality: registry[name] for modality, name in names.items()}
        )
        self.weights = weights or {k: 1.0 for k in names}

    def forward(
        self,
        output: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ):
        total_loss = 0.0
        for modality in output:
            loss = self.losses[modality](output[modality], target[modality])
            total_loss += self.weights[modality] * loss
        return total_loss


class ClosedFormMSE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        output = output.reshape(-1, output.shape[-1])
        target = target.reshape(-1, target.shape[-1])
        n, d = output.shape
        ones = torch.ones(n, 1, device=output.device, dtype=output.dtype)
        X_aug = torch.cat([output, ones], dim=1)  # (n, d+1)
        XtX = X_aug.T @ X_aug
        reg = 1e-6 * torch.eye(
            XtX.shape[0], device=output.device, dtype=output.dtype
        )
        beta = torch.linalg.inv(XtX + reg) @ (X_aug.T @ target)  # (d+1, m)
        Y_hat = X_aug @ beta
        mse_loss = torch.mean((target - Y_hat) ** 2)
        return mse_loss


def compute(
    gts: dict[str, dict[str, np.ndarray]],
    preds: dict[str, dict[str, np.ndarray]],
    metrics: dict | None,
) -> dict[str, float]:
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
    if metrics is None:
        return dict()

    results = dict()

    for modality, metric_name in metrics.items():
        gt_arrays = [gts[session][modality] for session in gts.keys()]
        pred_arrays = [preds[session][modality] for session in preds.keys()]

        if metric_name == "accuracy":
            results[metric_name] = np.mean(
                [accuracy(gt, pred) for gt, pred in zip(gt_arrays, pred_arrays)]
            )
        elif metric_name == "rmse":
            results[metric_name] = np.mean(
                [rmse(gt, pred) for gt, pred in zip(gt_arrays, pred_arrays)]
            )
        elif metric_name == "correlation":
            results["global_corr"] = np.mean(
                [
                    np.mean(global_correlation(gt, pred))
                    for gt, pred in zip(gt_arrays, pred_arrays)
                ]
            )
            results["local_corr"] = np.mean(
                [
                    np.mean(local_correlation(gt, pred))
                    for gt, pred in zip(gt_arrays, pred_arrays)
                ]
            )
        else:
            warnings.warn(f"metric {metric_name} not implemented.")

    return results


def rmse(gt: np.ndarray, pred: np.ndarray) -> float:
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
    return np.sqrt(np.mean((gt - pred) ** 2).item())


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
    # shape of pred must be (..., n_classes)
    pred = pred.reshape(-1, pred.shape[-1])
    pred = pred.argmax(axis=-1)
    gt = gt.reshape(-1)
    return balanced_accuracy(gt, pred)


def global_correlation(
    gt: np.ndarray,
    pred: np.ndarray,
    reduce: bool = True,
) -> float | np.ndarray:
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
    # expected shape of gt and pred: (-1, n_features)
    pred_centered = pred - np.mean(pred, axis=0)
    gt_centered = gt - np.mean(gt, axis=0)

    num = np.sum(pred_centered * gt_centered, axis=0)
    den = np.sqrt(
        np.sum(pred_centered**2, axis=0) * np.sum(gt_centered**2, axis=0)
    )
    global_corr = np.where(den != 0, num / den, np.nan)

    if reduce:
        global_corr = np.nanmean(global_corr).item()

    return global_corr


def local_correlation(
    gt: np.ndarray,
    pred: np.ndarray,
    reduce: bool = True,
    sampling_rate: int = 20,
    subinterval_len: int = 1000,
) -> float | np.ndarray:
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
    subinterval_len = int(subinterval_len * sampling_rate / 1000)
    n_frames = gt.shape[0]
    n_splits = int(np.ceil(n_frames / subinterval_len))
    gt_splits = np.array_split(gt, n_splits, axis=0)
    pred_splits = np.array_split(pred, n_splits, axis=0)
    corrs = [
        global_correlation(gt_sub, pred_sub, reduce=False)
        for gt_sub, pred_sub in zip(gt_splits, pred_splits)
    ]
    corrs = np.array(corrs)  # shape: (n_splits, n_features)
    if reduce:
        return np.nanmean(corrs).item()
    else:
        return np.nanmean(corrs, axis=0)
