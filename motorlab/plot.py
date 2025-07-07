import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn

from . import metrics
from . import poses
from . import room


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


def room_histogram2d(
    gts, preds, concat=False, experiment="gbyk", colorbar=True, save_path=None
):
    if concat:
        gts = {
            "all": np.concatenate(list(gts.values()), axis=0),
        }

        preds = {
            "all": np.concatenate(list(preds.values()), axis=0),
        }

    x_bins = np.linspace(0, room.x_size, room.x_divisions + 1)
    y_bins = np.linspace(0, room.y_size, room.y_divisions + 1)
    sessions = list(gts.keys())

    ncols = 3 if len(sessions) > 2 else len(sessions)
    nrows = (len(sessions) + ncols - 1) // ncols
    pad = 2 if colorbar else 0
    fig, outer_axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * room.x_divisions + pad, nrows * room.y_divisions),
    )
    outer_axes = np.array(outer_axes).reshape(nrows, ncols)

    for idx, session in enumerate(sessions):
        gt = gts[session]
        pred = preds[session]
        acc = metrics.balanced_accuracy(
            room.extract_tiles(gt), room.extract_tiles(pred)
        )

        i, j = divmod(idx, ncols)
        outer_ax = outer_axes[i, j]
        outer_ax.axis("off")
        outer_ax.set_title(
            f"{session} | acc: {acc:.2f}",
            fontsize=9 if len(sessions) > 2 else 12,
        )

        # create inset axes that fill the outer subplot
        inner_axes = np.empty(
            (room.y_divisions, room.x_divisions), dtype=object
        )
        for x in range(room.x_divisions):
            for y in range(room.y_divisions):
                left = x / room.x_divisions
                bottom = (room.y_divisions - 1 - y) / room.y_divisions
                width = 1 / room.x_divisions
                height = 1 / room.y_divisions
                ax = outer_ax.inset_axes([left, bottom, width, height])
                inner_axes[y, x] = ax

        inner_axes = inner_axes[::-1]  # to align (0,0) at bottom-left

        for i in range(room.x_divisions):
            for j in range(room.y_divisions):
                ax = inner_axes[j, i]

                in_tile = (
                    (gt[:, 0] >= x_bins[i])
                    & (gt[:, 0] < x_bins[i + 1])
                    & (gt[:, 1] >= y_bins[j])
                    & (gt[:, 1] < y_bins[j + 1])
                )
                if not np.any(in_tile):
                    ax.axis("off")
                    continue

                preds_in_tile = pred[in_tile]
                pred_heatmap, _, _ = np.histogram2d(
                    preds_in_tile[:, 0],
                    preds_in_tile[:, 1],
                    bins=[x_bins, y_bins],
                    density=True,
                )

                img = ax.imshow(
                    pred_heatmap.T,
                    origin="lower",
                    extent=[0, room.x_size, 0, room.y_size],
                    cmap="RdPu",
                    aspect="auto",
                )
                ax.set_xticks([])
                ax.set_yticks([])

    # turn off unused outer axes
    for ax in outer_axes.ravel()[len(sessions) :]:
        ax.axis("off")

    if colorbar:
        fig.colorbar(
            img, ax=outer_axes, orientation="vertical", location="right"
        )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()
