import warnings

import numpy as np

from sklearn.metrics import balanced_accuracy_score


def compute(gts, preds, metric):
    if metric == "accuracy":
        return accuracy(gts, preds)
    elif metric == "l2":
        pass
    elif metric == "correlation":
        return global_correlation(gts, preds, reduce=True), local_correlation(
            gts, preds, reduce=True
        )
    else:
        raise ValueError(f"metric {metric} not implemented.")


def balanced_accuracy(gt, pred):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="y_pred contains classes not in y_true",
            category=UserWarning,
            module="sklearn.metrics._classification",
        )
        return balanced_accuracy_score(gt, pred)


def accuracy(gts, preds):
    accs = []
    for session in preds:
        pred = preds[session]
        pred = pred.reshape(-1, pred.shape[-1])
        pred = pred.argmax(axis=-1)
        gt = gts[session]
        gt = gt.reshape(-1)
        accs.append(balanced_accuracy(gt, pred))
    return np.mean(accs)


def global_correlation(gts, preds, reduce=False):
    global_corr = {}

    for session in preds:
        pred = preds[session].reshape(-1, preds[session].shape[-1])
        gt = gts[session].reshape(-1, gts[session].shape[-1])

        pred_centered = pred - np.nanmean(pred, axis=0)
        gt_centered = gt - np.nanmean(gt, axis=0)

        numerator = np.nansum(pred_centered * gt_centered, axis=0)
        denominator = np.sqrt(
            np.nansum(pred_centered**2, axis=0)
            * np.nansum(gt_centered**2, axis=0)
        )

        global_corr[session] = numerator / denominator

    if reduce:
        return np.nanmean([np.nanmean(corr) for corr in global_corr.values()])

    return global_corr


def local_correlation(gts, preds, reduce=False):
    local_corr = {}

    for session in preds:
        preds_sess = np.stack(preds[session])  # (n_trials, n_time, n_channels)
        gts_sess = np.stack(gts[session])

        # Center predictions and targets per trial
        preds_centered = preds_sess - np.nanmean(
            preds_sess, axis=1, keepdims=True
        )
        gts_centered = gts_sess - np.nanmean(gts_sess, axis=1, keepdims=True)

        # Compute per-trial correlation numerators and denominators
        numerator = np.nansum(preds_centered * gts_centered, axis=1)
        denom = np.sqrt(
            np.nansum(preds_centered**2, axis=1)
            * np.nansum(gts_centered**2, axis=1)
        )

        # Trial-wise correlations (n_trials, n_channels)
        corr = numerator / denom

        # Average over trials for each channel
        local_corr[session] = np.nanmean(corr, axis=0)

    if reduce:
        return np.nanmean([np.nanmean(corr) for corr in local_corr.values()])

    return local_corr
