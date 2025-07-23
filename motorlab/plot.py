import warnings

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import seaborn as sns
import umap

from lipstick import GifMaker, update_fig
from sklearn.metrics import confusion_matrix as sklearn_cm

from motorlab import data, intervals, metrics, room, utils


def confusion_matrix(
    gts: dict,
    preds: dict,
    group: str | None = None,
    include_sitting: bool = False,
    concat: bool = False,
    save_path: str | None = None,
) -> None:
    """
        Plot confusion matrices for predictions vs. ground truth for each session.

        Parameters
        ----------
        gts,
        preds,
        group=None,
        include_sitting=False,
        concat=False,
        save_path=None,
    ):
            Whether to include sitting tiles. Default is False.
        concat : bool, optional
            Whether to concatenate all sessions. Default is False.
        save_path : str, optional
            Path to save the figure. Default is None.
    """

    def group_y(x):
        return x // 3

    def group_x(x):
        return x % 3

    if concat:
        gts = {
            "all": np.concatenate(list(gts.values()), axis=0),
        }

        preds = {
            "all": np.concatenate(list(preds.values()), axis=0),
        }

    if not include_sitting:
        to_filter = [0, 1, 2, 12, 13, 14]
        for session in gts:
            indices = np.where(np.isin(gts[session], to_filter))[0]
            gts[session] = np.delete(gts[session], indices, axis=0)
            preds[session] = np.delete(preds[session], indices, axis=0)

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

        acc = metrics.balanced_accuracy(gt, pred)
        conf_mat = sklearn_cm(gt, pred, labels=tiles_vec, normalize="true")

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


def room_histogram2d(gts, preds, concat=False, colorbar=True, save_path=None):
    """
    Plot 2D histograms of predicted vs. ground truth positions for each session.

    Parameters
    ----------
    gts : dict
        Ground truth positions for each session.
    preds : dict
        Predicted positions for each session.
    concat : bool, optional
        Whether to concatenate all sessions. Default is False.
    colorbar : bool, optional
        Whether to show colorbar. Default is True.
    save_path : str, optional
        Path to save the figure. Default is None.
    """
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

    img = None
    for idx, session in enumerate(sessions):
        gt = gts[session]
        pred = preds[session]
        acc = metrics.balanced_accuracy(
            room.get_tiles(gt[:, 0], gt[:, 1]),
            room.get_tiles(pred[:, 0], pred[:, 1]),
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

    if colorbar and img is not None:
        fig.colorbar(
            img, ax=outer_axes, orientation="vertical", location="right"
        )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()


def poses3d(
    data: np.ndarray,
    experiment: str,
    save_path: str | None = None,
    return_fig: bool = False,
    fps: int = 20,
) -> list[plt.Figure] | None:
    """
        Plot or animate 3D human pose(s) using a given skeleton structure.

        Parameters
        ----------
        data,
        experiment,
        save_path=None,
        return_fig=False,
        fps=20,
    ):
        return_fig : bool, optional
            If True, returns list of Figures instead of displaying/saving. Default is False.
        fps : int, optional
            Frames per second for GIF export if `data` is 3D. Default is 20.

        Returns
        -------
        list[plt.Figure] | None
            List of Figures if `return_fig` is True, otherwise None.
    """

    def draw_pose(pose):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for s, e in utils.SKELETON:
            x = [
                pose[utils.KEYPOINTS[experiment][s], 0],
                pose[utils.KEYPOINTS[experiment][e], 0],
            ]
            y = [
                pose[utils.KEYPOINTS[experiment][s], 1],
                pose[utils.KEYPOINTS[experiment][e], 1],
            ]
            z = [
                pose[utils.KEYPOINTS[experiment][s], 2],
                pose[utils.KEYPOINTS[experiment][e], 2],
            ]
            ax.plot(x, y, z, color="blue")

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-1, 1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=15, azim=70)
        ax.set_box_aspect([1, 1, 1])
        ax.grid(False)
        return fig, ax

    duration, n_3d_keypoints = (
        data.shape if len(data.shape) > 1 else (1, len(data))
    )
    if duration > 500:
        duration = 300
        warnings.warn(
            "too many frames, the gif will be truncated to 300 frames only."
        )

    data = np.atleast_3d(data.reshape(duration, n_3d_keypoints // 3, 3))
    figs = [draw_pose(pose) for pose in data]

    if return_fig:
        return figs

    if save_path and len(figs) > 1:
        with GifMaker(save_path, fps=fps) as gif:
            for fig, _ in figs:
                gif.add(fig)
        return
    elif save_path:
        fig, ax = draw_pose(data[0])
        fig.savefig(save_path, bbox_inches="tight")
        return

    for fig, ax in figs:
        update_fig(fig, ax)


def com(data_dir, sessions=None, ncols=3, homing=False, save_path=None):
    """
    Plot center of mass (COM) trajectories for each session.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing session data.
    sessions : list of str, optional
        List of session names. Default is None (use directory name).
    ncols : int, optional
        Number of columns in subplot grid. Default is 3.
    homing : bool, optional
        Whether to plot homing intervals. Default is False.
    save_path : str, optional
        Path to save the figure. Default is None.
    """
    data_dir = Path(data_dir)
    if sessions is None:
        sessions = [data_dir.name]
        data_dir = data_dir.parent

    n_sessions = len(sessions)
    ncols = min(ncols, n_sessions)
    nrows = (n_sessions + ncols - 1) // ncols

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 3.5, nrows * 3.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axs = axs.flat

    for i, s in enumerate(sessions):
        poses_dir = data_dir / s / "poses"
        poses_ = data.load_from_memmap(poses_dir)
        poses_ = poses_.reshape(-1, 21, 3)
        com = poses_[:, utils.COM_KEYPOINTS_IDXS["gbyk"], :2].mean(axis=1)

        trials_dir = data_dir / s / "trials"
        tiles = None
        try:
            tiles = data.load_tiles(data_dir, s, "gbyk", "tiles")
        except Exception:
            pass
        if homing:
            _intervals = (
                intervals.get_homing_intervals(trials_dir, tiles)
                if tiles is not None
                else []
            )
        else:
            _intervals = (
                intervals.get_trials_intervals(trials_dir, tiles)
                if tiles is not None
                else []
            )

        for start, end in _intervals:
            axs[i].plot(com[start:end, 0], com[start:end, 1], color="b")
            axs[i].set_title(s)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.tight_layout()
    plt.show()


def umap3d(
    data,
    conditions,
    plot_title="",
    plot_filename="",
    show=True,
):
    """
    Plot 3D UMAP embedding for different conditions.

    Parameters
    ----------
    data : np.ndarray
        Data to embed and plot.
    conditions : dict
        Dictionary mapping condition labels to intervals.
    plot_title : str, optional
        Title for the plot. Default is "".
    plot_filename : str, optional
        Path to save the plot as HTML. Default is "".
    show : bool, optional
        Whether to display the plot. Default is True.
    """
    color_palette = dict(
        zip(conditions.keys(), sns.color_palette("tab10", len(conditions)))
    )
    fig = go.Figure()

    for cond_label, cond_intervals in conditions.items():
        data_aligned = utils.align_intervals(data, cond_intervals)
        data_aligned = np.nanmean(data_aligned, axis=0)

        reducer = umap.UMAP(n_components=3, random_state=0)
        data_transformed = reducer.fit_transform(data_aligned)

        color = tuple(int(255 * c) for c in color_palette[cond_label])
        fig.add_trace(
            go.Scatter3d(
                x=data_transformed[:, 0],
                y=data_transformed[:, 1],
                z=data_transformed[:, 2],
                mode="lines",
                name=cond_label,
                line=dict(color=f"rgb{color}"),
            )
        )

    fig.update_layout(
        title=plot_title,
        scene=dict(
            xaxis_title="UMAP1",
            yaxis_title="UMAP2",
            zaxis_title="UMAP3",
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
            bgcolor="rgba(0,0,0,0)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=10, r=10, t=40, b=10),
    )

    if plot_filename is not None:
        fig.write_html(plot_filename)
    if show:
        fig.show()
