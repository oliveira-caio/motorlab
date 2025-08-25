import matplotlib.pyplot as plt
import numpy as np


from motorlab import keypoints, room
from matplotlib.lines import Line2D


def poses3d(
    data: np.ndarray,
    experiment: str,
    skeleton_type: str = "normal",
    optional_data: np.ndarray | None = None,
    ax: plt.Axes | None = None,
) -> None:
    """Visualize 3D poses."""
    assert data.ndim == 1, f"poses_data must be a 1D array, shape: {data.shape}"
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=10, azim=10)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(False)

    skeleton = keypoints.get_skeleton(skeleton_type)
    data = np.reshape(data, (-1, 3))

    for start, end in skeleton:
        if optional_data is not None:
            optional_data = np.reshape(optional_data, (-1, 3))
            start_idx = keypoints.to_idx(start, experiment)
            end_idx = keypoints.to_idx(end, experiment)
            x_axis = optional_data[start_idx, 0], optional_data[end_idx, 0]
            y_axis = optional_data[start_idx, 1], optional_data[end_idx, 1]
            z_axis = optional_data[start_idx, 2], optional_data[end_idx, 2]
            ax.plot(x_axis, y_axis, z_axis, color="red")

        start_idx = keypoints.to_idx(start, experiment)
        end_idx = keypoints.to_idx(end, experiment)
        x_axis = data[start_idx, 0], data[end_idx, 0]
        y_axis = data[start_idx, 1], data[end_idx, 1]
        z_axis = data[start_idx, 2], data[end_idx, 2]
        ax.plot(x_axis, y_axis, z_axis, color="black")

    if optional_data is not None:
        legend_elements = [
            Line2D([0], [0], color="black", lw=2, label="Ground truth"),
            Line2D([0], [0], color="red", lw=2, label="Prediction"),
        ]
        ax.legend(handles=legend_elements)


def center_of_mass(data, optional_data=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    x_size, y_size = room.get_size()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(0, x_size)
    ax.set_ylim(0, y_size)
    label = "" if optional_data is None else "Ground truth"

    if optional_data is not None:
        ax.plot(
            optional_data[:, 0],
            optional_data[:, 1],
            label="Prediction",
            color="red",
        )

    ax.plot(data[:, 0], data[:, 1], label=label, color="black")
    ax.legend(loc="lower right", ncols=1)


def spike_count(data, sampling_rate, optional_data=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spike count")
    label = "" if optional_data is None else "Ground truth"
    time = np.linspace(0, len(data) / sampling_rate, len(data))

    if optional_data is not None:
        ax.plot(
            time,
            optional_data,
            label="Prediction",
            color="red",
        )

    ax.plot(time, data, label=label, color="black")
    ax.legend()


def make_outputs_fig(gts, preds, cfg):
    output_modalities = list(preds[next(iter(preds))].keys())
    n_modalities = len(output_modalities)
    ncols = min(2, n_modalities)
    figs = []

    for session in preds.keys():
        n_intervals = len(next(iter(preds[session].values())))
        min_intervals = min(5, n_intervals)
        fig, axs = plt.subplots(
            nrows=min_intervals,
            ncols=ncols,
            figsize=(6 * ncols, 6 * min_intervals),
        )
        fig.suptitle(session)
        axs = axs.flatten()

        for interval_idx in range(min_intervals):
            for i, modality in enumerate(output_modalities):
                correct_idx = interval_idx * ncols + i
                if modality == "poses":
                    fig.delaxes(axs[correct_idx])
                    ax3d = fig.add_subplot(
                        min_intervals, ncols, correct_idx + 1, projection="3d"
                    )
                    axs[correct_idx] = ax3d
                    poses3d(
                        gts[session][modality][interval_idx][
                            cfg["sampling_rate"]
                        ],
                        experiment=cfg["experiment"],
                        skeleton_type=cfg["skeleton_type"],
                        ax=ax3d,
                        optional_data=preds[session][modality][interval_idx][
                            cfg["sampling_rate"]
                        ],
                    )
                elif modality == "location":
                    center_of_mass(
                        gts[session][modality][interval_idx],
                        ax=axs[correct_idx],
                        optional_data=preds[session][modality][interval_idx],
                    )
                elif modality == "spike_count":
                    spike_count(
                        gts[session][modality][interval_idx][..., 0],
                        sampling_rate=cfg["sampling_rate"],
                        ax=axs[correct_idx],
                        optional_data=preds[session][modality][interval_idx][
                            ..., 0
                        ],
                    )

                axs[correct_idx].set_title(
                    f"Interval {interval_idx} - {modality}"
                )

        figs.append(fig)
        plt.close()

    return figs


def make_inputs_fig(
    inputs: dict,
    cfg: dict,
):
    input_modalities = list(inputs[next(iter(inputs))].keys())
    n_modalities = len(input_modalities)
    ncols = min(2, n_modalities)
    figs = []

    for session in inputs.keys():
        n_intervals = len(next(iter(inputs[session].values())))
        min_intervals = min(5, n_intervals)
        fig, axs = plt.subplots(
            nrows=min_intervals,
            ncols=ncols,
            figsize=(6 * ncols, 6 * min_intervals),
        )
        fig.suptitle(session)
        axs = axs.flatten()

        for interval_idx in range(min_intervals):
            for i, modality in enumerate(input_modalities):
                correct_idx = interval_idx * ncols + i
                if modality == "poses":
                    fig.delaxes(axs[correct_idx])
                    ax3d = fig.add_subplot(
                        min_intervals, ncols, correct_idx + 1, projection="3d"
                    )
                    axs[correct_idx] = ax3d
                    poses3d(
                        inputs[session][modality][interval_idx][
                            cfg["sampling_rate"]
                        ],
                        experiment=cfg["experiment"],
                        skeleton_type=cfg["skeleton_type"],
                        ax=ax3d,
                    )
                elif modality == "location":
                    center_of_mass(
                        inputs[session][modality][interval_idx],
                        ax=axs[correct_idx],
                    )
                elif modality == "spike_count":
                    spike_count(
                        inputs[session][modality][interval_idx][..., 0],
                        sampling_rate=cfg["sampling_rate"],
                        ax=axs[correct_idx],
                    )

                axs[correct_idx].set_title(
                    f"Interval {interval_idx} - {modality}"
                )

        figs.append(fig)
        plt.close()

    return figs
