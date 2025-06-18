import numpy as np
import torch

from mymodels import metrics


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def iterate(
    model,
    dataloaders,
    in_modalities,
    out_modalities,
    loss_fns,
    optimizer,
    scheduler,
    mode,
    config,
):
    is_train = mode == "train"
    model.train() if is_train else model.eval()
    gts = dict()
    preds = dict()
    track_metrics = dict()
    losses = []

    with torch.set_grad_enabled(is_train):
        for session in dataloaders:
            gts[session] = []
            preds[session] = []
            for d in dataloaders[session]:
                x = torch.cat([d[m] for m in in_modalities], dim=-1).to(device)
                y = torch.cat([d[m] for m in out_modalities], dim=-1).to(device)

                pred = model(x, session)
                gts[session].append(y)
                preds[session].append(pred)

                if is_train:
                    optimizer.zero_grad()
                    loss = loss_fns[session](pred, y)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())

        stacked_preds = {session: torch.cat(preds[session], dim=0) for session in preds}
        stacked_gts = {session: torch.cat(gts[session], dim=0) for session in gts}
        track_metrics[config["metric"]] = metrics.compute(
            stacked_preds, stacked_gts, config["metric"]
        )

        if is_train:
            scheduler.step()
            track_metrics["loss"] = np.mean(losses)
            grad_norm = sum(
                p.grad.norm().item() for p in model.parameters() if p.grad is not None
            )
            track_metrics["grad_norm"] = grad_norm

    return track_metrics, preds, gts
