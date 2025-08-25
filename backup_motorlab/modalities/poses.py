import warnings

from pathlib import Path

import numpy as np
import yaml

from motorlab import utils


REPRESENTATIONS = [
    "allocentric",
    "centered",
    "egocentric",
    "trunk",
    "head",
    "pc",
    "loose",
    "medium",
    "strict",
    "draconian",
]


def load(session_dir: Path | str, sampling_rate: int) -> np.ndarray:
    """
    Load poses data for a session.

    Parameters
    ----------
    session_dir : Path or str
        Path to the session directory
    sampling_rate : int
        Sampling rate for the data

    Returns
    -------
    np.ndarray
        Poses data array
    """
    session_dir = Path(session_dir)
    poses_dir = session_dir / "poses"

    with open(poses_dir / "meta.yml", "r") as f:
        meta_data = yaml.safe_load(f)

    if "is_mem_mapped" in meta_data and meta_data["is_mem_mapped"]:
        poses_data = np.memmap(
            poses_dir / "data.mem",
            dtype=meta_data["dtype"],
            mode="r",
            shape=(meta_data["n_timestamps"], meta_data["n_signals"]),
        )
        poses_data = np.array(poses_data)
    else:
        poses_data = np.load(poses_dir / "data.npy")

    if sampling_rate <= meta_data["sampling_rate"]:
        factor = meta_data["sampling_rate"] // sampling_rate
        duration = meta_data["n_timestamps"] - (
            meta_data["n_timestamps"] % factor
        )
        poses_data = poses_data[:duration].reshape(
            -1, factor, meta_data["n_signals"]
        )
        poses_data = np.nanmean(poses_data, axis=1)
    else:
        warnings.warn(
            f"Upsampling not implemented. Original sampling rate is: {meta_data['sampling_rate']}"
        )

    return poses_data.astype(meta_data["dtype"])


def load_com(
    session_dir: Path | str,
    sampling_rate: int,
) -> np.ndarray:
    """
    Load center of mass data for a session.

    Parameters
    ----------
    session_dir : Path or str
        Path to the session directory
    sampling_rate : int
        Sampling rate for the data

    Returns
    -------
    np.ndarray
        Center of mass coordinates
    """
    session_dir = Path(session_dir)

    with open(session_dir / "poses" / "meta.yml", "r") as f:
        meta_data = yaml.safe_load(f)

    com_path = session_dir / "poses" / "meta" / "com.npy"
    com = np.load(com_path)

    factor = meta_data["sampling_rate"] // sampling_rate
    duration = len(com) - (len(com) % factor)
    com = com[:duration].reshape(-1, factor, com.shape[1])
    com = np.nanmean(com, axis=1)
    com = com.astype(meta_data["dtype"])

    return com


def compute_speed(data, representation):
    """
    Compute speed from data using the specified representation.

    Parameters
    ----------
    data : np.ndarray
        Input data array (poses for kps_*, COM for com_*).
    representation : str
        Type of speed representation ('kps_vec', 'com_vec', 'kps_mag', 'com_mag').

    Returns
    -------
    np.ndarray
        Speed array.
    """
    if "vec" in representation:
        speed = np.zeros_like(data)
        speed[1:] = np.diff(data, axis=0)
    elif "mag" in representation:
        duration = data.shape[0]
        speed = np.zeros((duration, 1))
        speed[1:] = np.linalg.norm(np.diff(data, axis=0), axis=1, keepdims=True)
    else:
        raise ValueError(f"unknown speed representation: {representation}.")
    return speed


def compute_acceleration(speed):
    """
    Compute acceleration from speed data.

    Parameters
    ----------
    speed : np.ndarray
        Speed array.

    Returns
    -------
    np.ndarray
        Acceleration array.
    """
    accel = np.zeros_like(speed)
    accel[1:] = np.diff(speed, axis=0)
    return accel


def set_representation(representation: str, old_gbyk: bool) -> dict:
    """Set up the representation for poses."""
    prefix = "old_" if old_gbyk else ""
    pcs_path = Path("artifacts/tables/analysis_pca")
    poses_config = dict()
    poses_config["representation"] = representation

    if representation == "trunk":
        poses_config["keypoints_to_exclude"] = [
            "e_tail",
            "head",
            "l_ear",
            "l_eye",
            "r_ear",
            "r_eye",
            "nose",
        ]
    elif representation == "head":
        poses_config["keypoints_to_exclude"] = [
            "e_tail",
            "s_tail",
            "l_hip",
            "l_knee",
            "l_ankle",
            "r_hip",
            "r_knee",
            "r_ankle",
            "l_shoulder",
            "l_elbow",
            "l_wrist",
            "r_shoulder",
            "r_elbow",
            "r_wrist",
        ]
    elif representation == "pc":
        poses_config["project_to_pca"] = True
    elif representation == "loose":
        poses_config["project_to_pca"] = True
        file = pcs_path / f"{prefix}loose.yml"
        poses_config["pcs_to_exclude"] = utils.data.load_yaml(file)
    elif representation == "medium":
        poses_config["project_to_pca"] = True
        file = pcs_path / f"{prefix}medium.yml"
        poses_config["pcs_to_exclude"] = utils.data.load_yaml(file)
    elif representation == "strict":
        poses_config["project_to_pca"] = True
        file = pcs_path / f"{prefix}strict.yml"
        poses_config["pcs_to_exclude"] = utils.data.load_yaml(file)
    elif representation == "draconian":
        poses_config["project_to_pca"] = True
        file = pcs_path / f"{prefix}draconian.yml"
        poses_config["pcs_to_exclude"] = utils.data.load_yaml(file)
    else:
        raise ValueError(f"Unknown representation: {representation}")

    return poses_config


def change_representation(data, representation, experiment):
    """
    Change the body representation of pose data.

    Parameters
    ----------
    data : np.ndarray
        Pose data array.
    representation : str
        Target representation ('allocentric', 'centered', 'egocentric', 'distances', 'angles', 'trunk').
    experiment : str
        Experiment name (for keypoint selection).

    Returns
    -------
    np.ndarray
        Transformed pose data.
    """
    if representation == "allocentric":
        new_poses = compute_allocentric(data, experiment)
    elif representation == "centered":
        new_poses = compute_centered(data, experiment)
    elif representation in {
        "egocentric",
        "pc",
        "loose",
        "medium",
        "strict",
        "draconian",
    }:
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
    """
    Compute trunk-centered pose representation.

    Parameters
    ----------
    data : np.ndarray
        Pose data array.
    experiment : str
        Experiment name (for keypoint selection).

    Returns
    -------
    np.ndarray
        Trunk-centered pose data.
    """
    n_frames, n_3d_keypoints = data.shape
    trunk_poses = data.reshape(n_frames, n_3d_keypoints // 3, 3).copy()

    neck = trunk_poses[:, utils.KEYPOINTS[experiment]["neck"]]
    s_tail = trunk_poses[:, utils.KEYPOINTS[experiment]["s_tail"]]
    r_wrist = trunk_poses[:, utils.KEYPOINTS[experiment]["r_wrist"]]

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
    """
    Compute pairwise distances between skeleton segments (neckless skeleton).

    Parameters
    ----------
    data : np.ndarray
        Pose data array.
    experiment : str
        Experiment name (for keypoint selection).

    Returns
    -------
    np.ndarray
        Distances for each skeleton segment.
    """
    n_frames, n_3d_keypoints = data.shape
    data_copy = data.reshape(n_frames, n_3d_keypoints // 3, 3).copy()
    dist_poses = np.empty(
        (n_frames, len(utils.get_neckless_skeleton())), dtype=np.float32
    )
    dist_poses = np.stack(
        [
            np.linalg.norm(
                data_copy[:, utils.KEYPOINTS[experiment][s]]
                - data_copy[:, utils.KEYPOINTS[experiment][e]],
                axis=1,
            )
            for s, e in utils.get_neckless_skeleton()
        ],
        axis=1,
    )
    return dist_poses


def compute_centered(data, experiment):
    """
    Center pose data at the midpoint between left and right ears.

    Parameters
    ----------
    data : np.ndarray
        Pose data array.
    experiment : str
        Experiment name (for keypoint selection).

    Returns
    -------
    np.ndarray
        Centered pose data.
    """
    n_frames, n_3d_keypoints = data.shape
    centered_poses = data.reshape(n_frames, n_3d_keypoints // 3, 3).copy()

    r_ear = centered_poses[:, [utils.keypoints.map_to_idx("r_ear", experiment)]]
    l_ear = centered_poses[:, [utils.keypoints.map_to_idx("l_ear", experiment)]]
    middle_head = 0.5 * (l_ear + r_ear)

    centered_poses = centered_poses - middle_head
    centered_poses = np.reshape(centered_poses, (n_frames, n_3d_keypoints))

    return centered_poses


def compute_allocentric(data, experiment):
    """
    Compute allocentric pose representation (room-centered, with head position appended).

    Parameters
    ----------
    data : np.ndarray
        Pose data array.
    experiment : str
        Experiment name (for keypoint selection).

    Returns
    -------
    np.ndarray
        Allocentric pose data with head position.
    """
    n_frames, n_3d_keypoints = data.shape
    allo_poses = data.reshape(n_frames, n_3d_keypoints // 3, 3).copy()

    r_ear = allo_poses[:, [utils.keypoints.map_to_idx("r_ear", experiment)]]
    l_ear = allo_poses[:, [utils.keypoints.map_to_idx("l_ear", experiment)]]
    middle_head = 0.5 * (l_ear + r_ear)

    allo_poses = allo_poses - middle_head
    allo_poses = np.reshape(allo_poses, (n_frames, n_3d_keypoints))
    middle_head = middle_head[:, 0, :].reshape(-1, 3)
    allo_poses = np.append(allo_poses, middle_head, axis=1)

    return allo_poses


def compute_egocentric(data, experiment):
    """
    Compute egocentric pose representation (head-centered, rotated to head axes).

    Parameters
    ----------
    data : np.ndarray
        Pose data array.
    experiment : str
        Experiment name (for keypoint selection).

    Returns
    -------
    np.ndarray
        Egocentric pose data.
    """
    n_frames, n_3d_keypoints = data.shape
    ego_poses = data.reshape(n_frames, n_3d_keypoints // 3, 3).copy()

    r_ear = ego_poses[:, [utils.keypoints.map_to_idx("r_ear", experiment)]]
    l_ear = ego_poses[:, [utils.keypoints.map_to_idx("l_ear", experiment)]]
    nose = ego_poses[:, [utils.keypoints.map_to_idx("nose", experiment)]]
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
    #     new_poses[:, utils.KEYPOINTS[experiment][kp]] = new_poses[
    #         :, utils.KEYPOINTS[experiment][kp]
    #     ].mean(axis=0)

    ego_poses = np.reshape(ego_poses, (n_frames, n_3d_keypoints))

    return ego_poses


def exclude_keypoints(data, keypoints, experiment):
    """
    Exclude specified keypoints from pose data.

    Parameters
    ----------
    data : np.ndarray
        Pose data array.
    keypoints : list[str]
        List of keypoint names to exclude.
    experiment : str
        Experiment name (for keypoint selection).

    Returns
    -------
    np.ndarray
        Pose data with specified keypoints removed.
    """
    if not keypoints:
        return data
    indices = [utils.keypoints.map_to_idx(kp, experiment) for kp in keypoints]
    n_frames, n_3d_keypoints = data.shape
    data = np.reshape(data, (n_frames, n_3d_keypoints // 3, 3))
    data = np.delete(data, indices, axis=1)
    data = np.reshape(data, (n_frames, -1))
    return data


def zero_keypoints(poses, keypoints, experiment):
    """
    Set specified keypoints to zero in pose data.

    Parameters
    ----------
    poses : np.ndarray
        Pose data array.
    keypoints : list[str]
        List of keypoint names to zero out.
    experiment : str
        Experiment name (for keypoint selection).

    Returns
    -------
    np.ndarray
        Pose data with specified keypoints zeroed.
    """
    if not keypoints:
        return poses
    indices = [utils.keypoints.map_to_idx(kp, experiment) for kp in keypoints]
    duration, n_3d_keypoints = poses.shape
    poses = np.reshape(poses, (duration, n_3d_keypoints // 3, 3))
    poses[:, indices, :] = 0.0
    poses = np.reshape(poses, (duration, -1))
    return poses


def normalize_poses(poses, POSES_DIR):
    """
    Normalize pose data to [-1, 1] using min/max from metadata.

    Parameters
    ----------
    poses : np.ndarray
        Pose data array.
    POSES_DIR : str or Path
        Directory containing min/max metadata.

    Returns
    -------
    np.ndarray
        Normalized pose data.
    """
    POSES_DIR = Path(POSES_DIR)
    with open(POSES_DIR / "meta" / "min.npy", "rb") as f:
        min_coord = np.load(f).item()
    with open(POSES_DIR / "meta" / "max.npy", "rb") as f:
        max_coord = np.load(f).item()
    return 2 * (poses - min_coord) / (max_coord - min_coord) - 1


def preprocess(
    data: np.ndarray,
    config: dict,
    experiment: str = "gbyk",
    intervals: list[list[int]] | None = None,
    pcs_to_exclude: np.ndarray | None = None,
) -> np.ndarray:
    """
    Preprocess pose data before training the model.

    WARNING: The order of the functions matters.
        - You CAN'T exclude the keypoints before changing the representation because the indices for the head keypoints are fixed and are used to change the representation, for example.
        - Excluding the keypoints AND projecting to PCA may lead to weird results.

    Parameters
    ----------
    data : np.ndarray
        Pose data array.
    config : dict
        Configuration dictionary for preprocessing.
    experiment : str, optional
        Experiment name. Default is 'gbyk'.
    intervals : np.ndarray, optional
        Intervals for PCA projection. Default is None.
    pcs_to_exclude : np.ndarray, optional
        Principal components to exclude. Default is None.

    Returns
    -------
    np.ndarray
        Preprocessed pose data.
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
        data = utils.data.project_to_pca(
            data,
            intervals,
            config.get("divide_variance", False),
        )[0]

    if pcs_to_exclude is not None:
        data = np.delete(data, pcs_to_exclude, axis=1)

    return data


def compute_elevation_angle(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Compute the elevation angle between two points.

    Parameters
    ----------
    p1 : np.ndarray
        First point(s), shape (..., 3).
    p2 : np.ndarray
        Second point(s), shape (..., 3).

    Returns
    -------
    np.ndarray
        Elevation angle(s) in radians.
    """
    dz = p2[..., 2] - p1[..., 2]
    dx = p2[..., 0] - p1[..., 0]
    dy = p2[..., 1] - p1[..., 1]
    horizontal_dist = np.sqrt(dx**2 + dy**2)
    return np.arctan2(dz, horizontal_dist)


def compute_joint_angle(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Compute the angle between two vectors.

    Parameters
    ----------
    v1 : np.ndarray
        First vector(s).
    v2 : np.ndarray
        Second vector(s).

    Returns
    -------
    np.ndarray
        Angle(s) in radians.
    """
    v1_normalized = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-8)
    v2_normalized = v2 / (np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(v1_normalized * v2_normalized, axis=-1)
    tmp = np.arccos(np.clip(dot, -1.0, 1.0))  # angle in radians
    return tmp


def compute_angles(
    poses: np.ndarray,
    experiment: str,
) -> np.ndarray:
    """
    Compute joint and elevation angles for all frames.

    Parameters
    ----------
    poses : np.ndarray
        Pose data array.
    experiment : str
        Experiment name (for keypoint selection).

    Returns
    -------
    np.ndarray
        Concatenated joint and elevation angles for each frame.
    """
    n_frames = poses.shape[0]
    poses = poses.reshape(n_frames, -1, 3)
    keypoints_angles = utils.keypoints.get_keypoints_angles()
    skeleton = utils.keypoints.get_skeleton(experiment)

    joint_angles = []
    for a, b, c in keypoints_angles:
        pa = poses[:, utils.keypoints.map_to_idx(a, experiment)]
        pb = poses[:, utils.keypoints.map_to_idx(b, experiment)]
        pc = poses[:, utils.keypoints.map_to_idx(c, experiment)]
        v1 = pa - pb
        v2 = pc - pb
        ang = compute_joint_angle(v1, v2)
        joint_angles.append(ang)
    joint_angles = np.stack(joint_angles, axis=1)

    elevation_angles = []
    for a, b in skeleton:
        p1 = poses[:, utils.keypoints.map_to_idx(a, experiment)]
        p2 = poses[:, utils.keypoints.map_to_idx(a, experiment)]
        elev = compute_elevation_angle(p1, p2)
        elevation_angles.append(elev)
    elevation_angles = np.stack(elevation_angles, axis=1)

    return np.concatenate([joint_angles, elevation_angles], axis=1)
