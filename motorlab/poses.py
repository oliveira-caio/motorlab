import warnings

from pathlib import Path

import numpy as np

from . import utils


def compute_speed(poses, representation, experiment):
    duration, n_3d_keypoints = poses.shape

    if representation == "kps_vec":
        speed = np.zeros_like(poses)
        speed[1:] = np.diff(poses, axis=0)
    elif representation == "com_vec":
        poses = poses.reshape(duration, n_3d_keypoints // 3, 3)
        com = poses[:, com_keypoints_idxs[experiment]].mean(axis=1)
        speed = np.zeros_like(com)
        speed[1:] = np.diff(com, axis=0)
    elif representation == "kps_mag":
        speed = np.zeros((duration, 1))
        speed[1:] = np.linalg.norm(
            np.diff(poses, axis=0), axis=1, keepdims=True
        )
    elif representation == "com_mag":
        poses = poses.reshape(duration, n_3d_keypoints // 3, 3)
        com = poses[:, com_keypoints_idxs[experiment]].mean(axis=1)
        speed = np.zeros((com.shape[0], 1))
        speed[1:] = np.linalg.norm(np.diff(com, axis=0), axis=1, keepdims=True)
    else:
        raise ValueError(f"unknown speed representation: {representation}.")

    return speed


def compute_acceleration(speed):
    accel = np.zeros_like(speed)
    accel[1:] = np.diff(speed, axis=0)
    return accel


def compute_com(poses, experiment: str):
    duration, n_3d_keypoints = poses.shape
    poses = poses.reshape(duration, n_3d_keypoints // 3, 3)
    com = poses[:, com_keypoints_idxs[experiment], :2].mean(axis=1)
    return com


def change_representation(data, representation, experiment):
    if representation == "allocentric":
        new_poses = compute_allocentric(data, experiment)
    elif representation == "centered":
        new_poses = compute_centered(data, experiment)
    elif representation == "egocentric":
        new_poses = compute_egocentric(data, experiment)
    elif representation == "distances":
        new_poses = compute_distances(data, experiment)
    elif representation == "angles":
        new_poses = compute_angles(data, experiment)
    elif representation == "trunk":
        new_poses = compute_trunk(data, experiment)
    else:
        raise ValueError(f"invalid body representation: {representation}.")
    return new_poses


def compute_trunk(data, experiment):
    n_frames, n_3d_keypoints = data.shape
    trunk_poses = data.reshape(n_frames, n_3d_keypoints // 3, 3).copy()

    neck = trunk_poses[:, keypoints_dict[experiment]["neck"]]
    s_tail = trunk_poses[:, keypoints_dict[experiment]["s_tail"]]
    r_wrist = trunk_poses[:, keypoints_dict[experiment]["r_wrist"]]

    trunk_poses = trunk_poses - neck[:, None, :]

    x_axis = s_tail - neck
    x_axis /= np.linalg.norm(x_axis, axis=-1, keepdims=True)

    v = neck - r_wrist
    z_axis = v - np.sum(v * x_axis, axis=-1, keepdims=True) * x_axis
    z_axis /= np.linalg.norm(z_axis, axis=-1, keepdims=True)

    y_axis = np.cross(x_axis, z_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=-1)
    R = np.transpose(R, (0, 2, 1))
    trunk_poses = np.einsum("nij,nkj->nki", R, trunk_poses)
    trunk_poses = trunk_poses.reshape(n_frames, n_3d_keypoints)

    return trunk_poses


def compute_distances(data, experiment):
    n_frames, n_3d_keypoints = data.shape
    data_copy = data.reshape(n_frames, n_3d_keypoints // 3, 3).copy()
    dist_poses = np.empty((n_frames, len(neckless_skeleton)), dtype=np.float32)
    dist_poses = np.stack(
        [
            np.linalg.norm(
                data_copy[:, keypoints_dict[experiment][s]]
                - data_copy[:, keypoints_dict[experiment][e]],
                axis=1,
            )
            for s, e in neckless_skeleton
        ],
        axis=1,
    )
    return dist_poses


def compute_centered(data, experiment):
    n_frames, n_3d_keypoints = data.shape
    centered_poses = data.reshape(n_frames, n_3d_keypoints // 3, 3).copy()

    r_ear = centered_poses[:, [keypoints_dict[experiment]["r_ear"]]]
    l_ear = centered_poses[:, [keypoints_dict[experiment]["l_ear"]]]
    middle_head = 0.5 * (l_ear + r_ear)

    centered_poses = centered_poses - middle_head
    centered_poses = np.reshape(centered_poses, (n_frames, n_3d_keypoints))

    return centered_poses


def compute_allocentric(data, experiment):
    n_frames, n_3d_keypoints = data.shape
    allo_poses = data.reshape(n_frames, n_3d_keypoints // 3, 3).copy()

    r_ear = allo_poses[:, [keypoints_dict[experiment]["r_ear"]]]
    l_ear = allo_poses[:, [keypoints_dict[experiment]["l_ear"]]]
    middle_head = 0.5 * (l_ear + r_ear)

    allo_poses = allo_poses - middle_head
    allo_poses = np.reshape(allo_poses, (n_frames, n_3d_keypoints))
    middle_head = middle_head[:, 0, :].reshape(-1, 3)
    allo_poses = np.append(allo_poses, middle_head, axis=1)

    return allo_poses


def compute_egocentric(data, experiment):
    n_frames, n_3d_keypoints = data.shape
    ego_poses = data.reshape(n_frames, n_3d_keypoints // 3, 3).copy()

    r_ear = ego_poses[:, [keypoints_dict[experiment]["r_ear"]]]
    l_ear = ego_poses[:, [keypoints_dict[experiment]["l_ear"]]]
    nose = ego_poses[:, [keypoints_dict[experiment]["nose"]]]
    middle_head = 0.5 * (l_ear + r_ear)

    ego_poses = ego_poses - middle_head

    norm_factor = 10
    x_axis = norm_factor * (r_ear - middle_head).squeeze()
    y_axis = norm_factor * (nose - middle_head).squeeze()

    inner_product = np.sum(x_axis * y_axis, axis=1, keepdims=True)
    y_axis = y_axis - inner_product * x_axis
    z_axis = np.cross(x_axis, y_axis)

    # change basis from room coords to head-centered coords.
    basis = np.stack((x_axis, y_axis, z_axis), axis=2)
    cob_matrices = np.linalg.inv(basis)
    ego_poses = np.einsum("nij,nkj->nki", cob_matrices, ego_poses)

    # for kp in ["head", "nose", "r_ear", "l_ear", "l_eye", "r_eye", "neck"]:
    #     new_poses[:, keypoints_dict[experiment][kp]] = new_poses[
    #         :, keypoints_dict[experiment][kp]
    #     ].mean(axis=0)

    ego_poses = np.reshape(ego_poses, (n_frames, n_3d_keypoints))

    return ego_poses


def exclude_keypoints(data, keypoints, experiment):
    if not keypoints:
        return data
    indices = [keypoints_dict[experiment][kp] for kp in keypoints]
    n_frames, n_3d_keypoints = data.shape
    data = np.reshape(data, (n_frames, n_3d_keypoints // 3, 3))
    data = np.delete(data, indices, axis=1)
    data = np.reshape(data, (n_frames, -1))
    return data


def zero_keypoints(poses, keypoints, experiment):
    if not keypoints:
        return poses
    indices = [keypoints_dict[experiment][kp] for kp in keypoints]
    duration, n_3d_keypoints = poses.shape
    poses = np.reshape(poses, (duration, n_3d_keypoints // 3, 3))
    poses[:, indices, :] = 0.0
    poses = np.reshape(poses, (duration, -1))
    return poses


def normalize_poses(poses, POSES_DIR):
    POSES_DIR = Path(POSES_DIR)
    with open(POSES_DIR / "meta" / "min.npy", "rb") as f:
        min_coord = np.load(f).item()
    with open(POSES_DIR / "meta" / "max.npy", "rb") as f:
        max_coord = np.load(f).item()
    return 2 * (poses - min_coord) / (max_coord - min_coord) - 1


def preprocess(
    data, config, experiment="gbyk", intervals=None, pcs_to_exclude=None
):
    """preprocess poses data before training the model.

    WARNING: the order of the functions matter.
        - you CAN'T exclude the keypoints before changing the representation because the indices for the head keypoints are fixed and are used to change the representation, for example.
        - excluding the keypoints AND projecting to PCA may lead to weird results.
        - excluding
    """

    if "representation" in config:
        data = change_representation(
            data,
            config["representation"],
            experiment,
        )

    if "keypoints_to_exclude" in config:
        data = exclude_keypoints(
            data,
            config["keypoints_to_exclude"],
            experiment,
        )

    if config.get("project_to_pca", False):
        if intervals is None:
            warnings.warn(
                "intervals not provided, it'll do PCA on the entire session."
            )
        data = utils.project_to_pca(
            data,
            intervals,
            config.get("divide_variance", False),
        )[0]

    if pcs_to_exclude is not None:
        data = np.delete(data, pcs_to_exclude, axis=1)

    return data


def compute_elevation_angle(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    dz = p2[..., 2] - p1[..., 2]
    dx = p2[..., 0] - p1[..., 0]
    dy = p2[..., 1] - p1[..., 1]
    horizontal_dist = np.sqrt(dx**2 + dy**2)
    return np.arctan2(dz, horizontal_dist)


def compute_joint_angle(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    v1_normalized = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-8)
    v2_normalized = v2 / (np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(v1_normalized * v2_normalized, axis=-1)
    tmp = np.arccos(np.clip(dot, -1.0, 1.0))  # angle in radians
    return tmp


def compute_angles(
    poses: np.ndarray,
    experiment: str,
):
    n_frames = poses.shape[0]
    n_keypoints = len(keypoints_dict[experiment])
    poses = poses.reshape(n_frames, n_keypoints, 3)

    joint_angles = []
    for a, b, c in joint_angles_idxs:
        pa = poses[:, keypoints_dict[experiment][a]]
        pb = poses[:, keypoints_dict[experiment][b]]
        pc = poses[:, keypoints_dict[experiment][c]]
        v1 = pa - pb
        v2 = pc - pb
        ang = compute_joint_angle(v1, v2)
        joint_angles.append(ang)
    joint_angles = np.stack(joint_angles, axis=1)

    elevation_angles = []
    for a, b in SKELETON:
        p1 = poses[:, keypoints_dict[experiment][a]]
        p2 = poses[:, keypoints_dict[experiment][b]]
        elev = compute_elevation_angle(p1, p2)
        elevation_angles.append(elev)
    elevation_angles = np.stack(elevation_angles, axis=1)

    return np.concatenate([joint_angles, elevation_angles], axis=1)
