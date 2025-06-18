import numpy as np
import torch


def compute(preds, gts, metric):
    if metric == "accuracy":
        return accuracy(preds, gts)
    elif metric == "l2":
        pass
    elif metric == "correlation":
        return correlation(preds, gts)
    else:
        raise ValueError(f"metric {metric} not implemented.")


def accuracy(preds, gts):
    accs = []
    for session in preds:
        pred = preds[session]
        pred = pred.reshape(-1, pred.shape[-1])
        pred = pred.argmax(dim=-1)
        gt = gts[session]
        gt = gt.reshape(-1)
        accs.append((pred == gt).float().mean().item())
    return np.mean(accs)


def l2():
    pass


def correlation(preds, gts):
    corr = dict()
    for session in preds:
        corrs = torch.zeros(preds[session].shape[-1])
        for ch in range(preds[session].shape[-1]):
            pred = preds[session][..., ch].ravel()
            gt = gts[session][..., ch].ravel()
            corrs[ch] = torch.corrcoef(torch.stack([pred, gt]))[0, 1]
        corr[session] = torch.nanmean(corrs)
    final_corr = torch.nanmean(torch.tensor([corr[session] for session in corr]))
    return final_corr
