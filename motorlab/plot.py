import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn

from . import metrics


def confusion_matrix(
    gts, preds, group=None, include_sitting=False, save_path=None
):
    def group_y(x):
        return x // 3

    def group_x(x):
        return x % 3

    n_sessions = len(gts)
    n_cols = 2
    n_rows = int(np.ceil(n_sessions / n_cols))

    if group == "x":
        tiles_vec = range(3)
        coord_labels = [str(i) for i in tiles_vec]
        col_factor, row_factor = 5.5, 4
    elif group == "y":
        tiles_vec = range(5) if include_sitting else range(1, 4)
        coord_labels = [str(i) for i in tiles_vec]
        col_factor, row_factor = 5.5, 4
    else:
        tiles_vec = range(15) if include_sitting else range(3, 12)
        coord_labels = [f"({i % 3}, {i // 3})" for i in tiles_vec]
        col_factor, row_factor = 9, 6

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(col_factor * n_cols, row_factor * n_rows),
        sharex=True,
        sharey=True,
    )

    for idx, session in enumerate(gts):
        row, col = divmod(idx, n_cols)
        ax = axs[row, col] if n_rows > 1 else axs[col]

        if group == "x":
            gt = group_x(gts[session])
            pred = group_x(preds[session])
        elif group == "y":
            gt = group_y(gts[session])
            pred = group_y(preds[session])
        else:
            gt = gts[session]
            pred = preds[session]

        conf_mat = sklearn.metrics.confusion_matrix(
            gt, pred, labels=tiles_vec, normalize="true"
        )
        acc = metrics.balanced_accuracy(gt, pred)

        sns.heatmap(
            conf_mat,
            cmap="Blues",
            ax=ax,
            xticklabels=coord_labels,
            yticklabels=coord_labels,
        )
        ax.set_title(f"{session} | acc: {acc:.2f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", labelrotation=0)

    for i in range(n_sessions, n_rows * n_cols):
        fig.delaxes(axs.flatten()[i])

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.tight_layout()
    plt.show()
