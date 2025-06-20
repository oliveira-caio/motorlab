import numpy as np


def compute(preds, gts, metric):
    if metric == "accuracy":
        return accuracy(preds, gts)
    elif metric == "l2":
        pass
    elif metric == "correlation":
        return global_correlation(preds, gts, reduce=True), local_correlation(
            preds, gts, reduce=True
        )
    else:
        raise ValueError(f"metric {metric} not implemented.")


def accuracy(preds, gts):
    accs = []
    for session in preds:
        pred = preds[session]
        pred = pred.reshape(-1, pred.shape[-1])
        pred = pred.argmax(axis=-1)
        gt = gts[session]
        gt = gt.reshape(-1)
        accs.append((pred == gt).mean().item())
    return np.mean(accs)


def l2():
    pass


def global_correlation(preds, gts, reduce=False):
    global_corr = {}

    for session in preds:
        pred = preds[session].reshape(-1, preds[session].shape[-1])
        gt = gts[session].reshape(-1, gts[session].shape[-1])

        pred_centered = pred - np.nanmean(pred, axis=0)
        gt_centered = gt - np.nanmean(gt, axis=0)

        numerator = np.nansum(pred_centered * gt_centered, axis=0)
        denominator = np.sqrt(
            np.nansum(pred_centered**2, axis=0) * np.nansum(gt_centered**2, axis=0)
        )

        global_corr[session] = numerator / denominator

    if reduce:
        return np.nanmean([np.nanmean(corr) for corr in global_corr.values()])

    return global_corr


def old_local_correlation(preds, gts, reduce=False):
    local_corr = dict()

    for session in preds:
        n_channels = gts[session][0].shape[-1]
        local_corr[session] = np.array(
            [
                np.nanmean(
                    [
                        np.corrcoef(pred[:, ch], gt[:, ch])[0, 1]
                        for pred, gt in zip(preds[session], gts[session])
                    ]
                )
                for ch in range(n_channels)
            ]
        )

    if reduce:
        sessions_mean_corr = [np.nanmean(local_corr[s]) for s in local_corr]
        return np.nanmean(sessions_mean_corr)

    return local_corr


def local_correlation(preds, gts, reduce=False):
    local_corr = {}

    for session in preds:
        preds_sess = np.stack(preds[session])  # (n_trials, n_time, n_channels)
        gts_sess = np.stack(gts[session])

        # Center predictions and targets per trial
        preds_centered = preds_sess - np.nanmean(preds_sess, axis=1, keepdims=True)
        gts_centered = gts_sess - np.nanmean(gts_sess, axis=1, keepdims=True)

        # Compute per-trial correlation numerators and denominators
        numerator = np.nansum(preds_centered * gts_centered, axis=1)
        denom = np.sqrt(
            np.nansum(preds_centered**2, axis=1) * np.nansum(gts_centered**2, axis=1)
        )

        # Trial-wise correlations (n_trials, n_channels)
        corr = numerator / denom

        # Average over trials for each channel
        local_corr[session] = np.nanmean(corr, axis=0)

    if reduce:
        return np.nanmean([np.nanmean(corr) for corr in local_corr.values()])

    return local_corr
