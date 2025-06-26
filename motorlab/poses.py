from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from lipstick import GifMaker, update_fig

from . import data
from . import utils


skeleton = [
    ["S_tail", "E_tail"],
    ["S_tail", "L_hip"],
    ["S_tail", "R_hip"],
    ["R_knee", "R_hip"],
    ["R_knee", "R_ankle"],
    ["L_knee", "L_hip"],
    ["L_knee", "L_ankle"],
    ["R_elbow", "R_shoulder"],
    ["R_elbow", "R_wrist"],
    ["L_elbow", "L_shoulder"],
    ["L_elbow", "L_wrist"],
    ["neck", "S_tail"],
    ["neck", "head"],
    ["neck", "nose"],
    ["neck", "L_shoulder"],
    ["neck", "R_shoulder"],
    ["neck", "L_ear"],
    ["neck", "R_ear"],
    ["nose", "L_eye"],
    ["nose", "R_eye"],
    ["nose", "L_ear"],
    ["nose", "R_ear"],
    ["head", "L_ear"],
    ["head", "R_ear"],
    ["head", "L_eye"],
    ["head", "R_eye"],
]

keypoints = {
    "gbyk": {
        "E_tail": 0,
        "L_ankle": 1,
        "L_ear": 2,
        "L_elbow": 3,
        "L_eye": 4,
        "L_hip": 5,
        "L_knee": 6,
        "L_shoulder": 7,
        "L_wrist": 8,
        "R_ankle": 9,
        "R_ear": 10,
        "R_elbow": 11,
        "R_eye": 12,
        "R_hip": 13,
        "R_knee": 14,
        "R_shoulder": 15,
        "R_wrist": 16,
        "S_tail": 17,
        "head": 18,
        "neck": 19,
        "nose": 20,
    },
    "old_gbyk": {
        "L_wrist": 0,
        "L_elbow": 1,
        "L_shoulder": 2,
        "R_wrist": 3,
        "R_elbow": 4,
        "R_shoulder": 5,
        "L_ankle": 6,
        "L_knee": 7,
        "L_hip": 8,
        "R_ankle": 9,
        "R_knee": 10,
        "R_hip": 11,
        "E_tail": 12,
        "S_tail": 13,
        "neck": 14,
        "head": 15,
        "L_ear": 16,
        "R_ear": 17,
        "L_eye": 18,
        "R_eye": 19,
        "nose": 20,
    },
    "pg": {
        "neck": 0,
        "spine": 1,
        "head": 2,
        "L_ear": 3,
        "R_ear": 4,
        "L_eye": 5,
        "R_eye": 6,
        "nose": 7,
        "L_shoulder": 8,
        "L_elbow": 9,
        "L_wrist": 10,
        "L_upperArm": 11,
        "L_lowerArm": 12,
        "R_shoulder": 13,
        "R_elbow": 14,
        "R_wrist": 15,
        "R_upperArm": 16,
        "R_lowerArm": 17,
        "L_hip": 18,
        "L_knee": 19,
        "L_ankle": 20,
        "L_upperLeg": 21,
        "L_lowerLeg": 22,
        "R_hip": 23,
        "R_knee": 24,
        "R_ankle": 25,
        "R_upperLeg": 26,
        "R_lowerLeg": 27,
        "S_tail": 28,
        "M_tail": 29,
        "E_tail": 30,
    },
}

com_keypoints = {
    "gbyk": [
        keypoints["gbyk"]["L_hip"],
        keypoints["gbyk"]["R_hip"],
        keypoints["gbyk"]["L_shoulder"],
        keypoints["gbyk"]["R_shoulder"],
        keypoints["gbyk"]["neck"],
        keypoints["gbyk"]["S_tail"],
    ],
    "pg": [
        keypoints["pg"]["L_hip"],
        keypoints["pg"]["R_hip"],
        keypoints["pg"]["L_shoulder"],
        keypoints["pg"]["R_shoulder"],
        keypoints["pg"]["neck"],
        keypoints["pg"]["S_tail"],
    ],
}

extra_keypoints = {
    "gbyk": [
        keypoints["gbyk"]["L_eye"],
        keypoints["gbyk"]["R_eye"],
        keypoints["gbyk"]["head"],
    ],
    "pg": [
        keypoints["pg"]["spine"],
        keypoints["pg"]["head"],
        keypoints["pg"]["L_eye"],
        keypoints["pg"]["R_eye"],
        keypoints["pg"]["L_upperArm"],
        keypoints["pg"]["L_lowerArm"],
        keypoints["pg"]["R_upperArm"],
        keypoints["pg"]["R_lowerArm"],
        keypoints["pg"]["L_upperLeg"],
        keypoints["pg"]["L_lowerLeg"],
        keypoints["pg"]["R_upperLeg"],
        keypoints["pg"]["R_lowerLeg"],
        keypoints["pg"]["M_tail"],
    ],
}


def compute_speed(poses, experiment, speed_repr):
    duration, n_3d_keypoints = poses.shape

    if speed_repr == "kps_vec":
        speed = np.zeros_like(poses)
        speed[1:] = np.diff(poses, axis=0)
    elif speed_repr == "com_vec":
        poses = poses.reshape(duration, n_3d_keypoints // 3, 3)
        com = poses[:, com_keypoints[experiment]].mean(axis=1)
        speed = np.zeros_like(com)
        speed[1:] = np.diff(com, axis=0)
    elif speed_repr == "kps_mag":
        speed = np.zeros((duration, 1))
        speed[1:] = np.linalg.norm(
            np.diff(poses, axis=0), axis=1, keepdims=True
        )
    elif speed_repr == "com_mag":
        poses = poses.reshape(duration, n_3d_keypoints // 3, 3)
        com = poses[:, com_keypoints[experiment]].mean(axis=1)
        speed = np.zeros((com.shape[0], 1))
        speed[1:] = np.linalg.norm(np.diff(com, axis=0), axis=1, keepdims=True)
    else:
        raise ValueError(f"unknown speed representation: {speed_repr}.")

    return speed


def compute_acceleration(speed):
    accel = np.zeros_like(speed)
    accel[1:] = np.diff(speed, axis=0)
    return accel


def compute_com(poses, experiment):
    duration, n_3d_keypoints = poses.shape
    poses = poses.reshape(duration, n_3d_keypoints // 3, 3)
    com = poses[:, com_keypoints[experiment], :2].mean(axis=1)
    return com


def change_repr(poses, experiment, body_repr):
    # normalizing factor for the egocentric poses. it keeps the coordinates within a good numerical range.
    norm_factor = 10

    duration, n_3d_keypoints = poses.shape
    new_poses = poses.reshape(duration, n_3d_keypoints // 3, 3).copy()
    l_ear = new_poses[:, [keypoints[experiment]["L_ear"]]]
    r_ear = new_poses[:, [keypoints[experiment]["R_ear"]]]
    middle_head = 0.5 * (l_ear + r_ear)

    if body_repr in {"allocentric", "centered"}:
        new_poses -= np.tile(middle_head, (1, n_3d_keypoints // 3, 1))
        new_poses = np.reshape(new_poses, (duration, n_3d_keypoints))
        if body_repr == "allocentric":
            middle_head = middle_head[:, 0, :].reshape(-1, 3)
            new_poses = np.append(new_poses, middle_head, axis=1)
    elif body_repr == "egocentric":
        nose = new_poses[:, [keypoints[experiment]["nose"]]]
        x_axis = norm_factor * (r_ear - middle_head).squeeze()
        y_axis = norm_factor * (nose - middle_head).squeeze()
        inner_product = np.sum(x_axis * y_axis, axis=1, keepdims=True)
        y_axis = y_axis - inner_product * x_axis
        z_axis = np.cross(x_axis, y_axis)
        # change of basis matrices: from room coords to head-centered coords.
        cob_matrices = np.linalg.pinv(
            np.stack((x_axis, y_axis, z_axis), axis=2)
        )
        new_poses -= np.tile(middle_head, (1, n_3d_keypoints // 3, 1))
        new_poses = np.einsum("nij,nkj->nki", cob_matrices, new_poses)
        new_poses = np.reshape(new_poses, (duration, n_3d_keypoints))
    else:
        raise ValueError(f"invalid body representation: {body_repr}.")

    return new_poses


def exclude_keypoints(poses, experiment, keypoints_to_exclude=None):
    duration, n_3d_keypoints = poses.shape
    poses = np.reshape(poses, (duration, n_3d_keypoints // 3, 3))
    if keypoints_to_exclude is None:
        keypoints_to_exclude = extra_keypoints[experiment]
    poses = np.delete(poses, keypoints_to_exclude, axis=1)
    poses = np.reshape(poses, (duration, -1))
    return poses


def normalize_poses(poses, POSES_DIR):
    POSES_DIR = Path(POSES_DIR)
    with open(POSES_DIR / "meta" / "min.npy", "rb") as f:
        min_coord = np.load(f).item()
    with open(POSES_DIR / "meta" / "max.npy", "rb") as f:
        max_coord = np.load(f).item()
    return 2 * (poses - min_coord) / (max_coord - min_coord) - 1


def plot(
    poses,
    experiment,
    save_path=None,
    return_fig=False,
    fps=20,
):
    """
    Plot or animate 3D human pose(s) using a given skeleton structure.

    Parameters:
    - poses (ndarray): Array of shape (J, 3) or (T, J, 3) representing 3D joint positions.
    - save_path (str, optional): If given and ends with .gif, saves output GIF here.
    - return_fig (bool, optional): If True, returns list of Figures instead of displaying/saving.
    - fps (int, optional): Frames per second for GIF export if `poses` is 3D.

    Returns:
    - matplotlib.figure.Figure, list of Figures, or None.
    """

    def draw_pose(pose):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for s, e in skeleton:
            x = [
                pose[keypoints[experiment][s], 0],
                pose[keypoints[experiment][e], 0],
            ]
            y = [
                pose[keypoints[experiment][s], 1],
                pose[keypoints[experiment][e], 1],
            ]
            z = [
                pose[keypoints[experiment][s], 2],
                pose[keypoints[experiment][e], 2],
            ]
            ax.plot(x, y, z, color="blue")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=15, azim=70)
        ax.set_box_aspect([1, 1, 1])
        ax.grid(False)
        return fig, ax

    duration, n_3d_keypoints = poses.shape
    poses = np.atleast_3d(poses.reshape(duration, n_3d_keypoints // 3, 3))
    figs = [draw_pose(pose) for pose in poses]

    if return_fig:
        return figs

    if save_path and len(figs) > 1:
        with GifMaker(save_path, fps=fps) as gif:
            for fig in figs:
                gif.add(fig)
        return
    elif save_path:
        fig, ax = draw_pose(poses[0])
        fig.savefig(save_path, bbox_inches="tight")
        return

    for fig, ax in figs:
        update_fig(fig, ax)


def plot_com(DATA_DIR, sessions=None, ncols=3, homing=False):
    DATA_DIR = Path(DATA_DIR)
    if sessions is None:
        sessions = [DATA_DIR.name]
        DATA_DIR = DATA_DIR.parent

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
        POSES_DIR = DATA_DIR / s / "poses"
        poses = data.load_from_memmap(POSES_DIR)
        poses = poses.reshape(-1, 21, 3)
        com = poses[:, com_keypoints["gbyk"], :2].mean(axis=1)

        TRIALS_DIR = DATA_DIR / s / "trials"
        if homing:
            intervals = utils.extract_homing_intervals(TRIALS_DIR)
        else:
            intervals = utils.extract_trials_intervals(TRIALS_DIR)

        for start, end in intervals:
            axs[i].plot(com[start:end, 0], com[start:end, 1], color="b")
            axs[i].set_title(s)

    fig.tight_layout()
    plt.show()
