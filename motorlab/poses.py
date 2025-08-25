import pathlib
import warnings

import numpy as np
import yaml

from motorlab import keypoints, room, utils


def load(session_dir: str | pathlib.Path, sampling_rate: int):
    session_dir = pathlib.Path(session_dir)
    poses_dir = session_dir / "poses"

    with open(poses_dir / "meta.yml", "r") as f:
        meta = yaml.safe_load(f)

    poses_data = np.memmap(
        poses_dir / "data.mem",
        dtype=meta["dtype"],
        mode="r",
        shape=(meta["n_timestamps"], meta["n_signals"]),
    )

    if sampling_rate < meta["sampling_rate"]:
        factor = meta["sampling_rate"] // sampling_rate
        duration = meta["n_timestamps"] - (meta["n_timestamps"] % factor)
        poses_data = poses_data[:duration].reshape(
            -1, factor, meta["n_signals"]
        )
        poses_data = np.nanmean(poses_data, axis=1)
    elif sampling_rate == meta["sampling_rate"]:
        pass
    else:
        warnings.warn(
            f"Upsampling not implemented. Original sampling rate is: {meta['sampling_rate']}"
        )

    return poses_data.astype(meta["dtype"])


def load_com(session_dir: str | pathlib.Path, sampling_rate: int):
    session_dir = pathlib.Path(session_dir)
    poses_dir = session_dir / "poses"

    with open(poses_dir / "meta.yml", "r") as f:
        meta = yaml.safe_load(f)

    com_data = np.load(poses_dir / "meta/com.npy")

    if sampling_rate <= meta["sampling_rate"]:
        factor = meta["sampling_rate"] // sampling_rate
        duration = meta["n_timestamps"] - (meta["n_timestamps"] % factor)
        com_data = com_data[:duration].reshape(-1, factor, 3)
        com_data = np.nanmean(com_data, axis=1)
    else:
        warnings.warn(
            f"Upsampling not implemented. Original sampling rate is: {meta['sampling_rate']}"
        )

    return com_data.astype(meta["dtype"])


def _compute_allocentric(data, experiment):
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
    processed_data = data.reshape(len(data), -1, 3).copy()

    r_ear = processed_data[:, [keypoints.to_idx("r_ear", experiment)]]
    l_ear = processed_data[:, [keypoints.to_idx("l_ear", experiment)]]
    middle_head = 0.5 * (l_ear + r_ear)

    processed_data = processed_data - middle_head
    processed_data = np.reshape(processed_data, (len(data), -1))
    middle_head = middle_head[:, 0, :].reshape(-1, 3)
    processed_data = np.append(processed_data, middle_head, axis=1)

    return processed_data


def _compute_centered(data, experiment):
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
    processed_data = data.reshape(len(data), -1, 3).copy()

    r_ear = processed_data[:, [keypoints.to_idx("r_ear", experiment)]]
    l_ear = processed_data[:, [keypoints.to_idx("l_ear", experiment)]]
    middle_head = 0.5 * (l_ear + r_ear)

    processed_data = processed_data - middle_head
    processed_data = np.reshape(processed_data, (len(data), -1))

    return processed_data


def _compute_egocentric(data, experiment):
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
    processed_data = data.reshape(len(data), -1, 3).copy()

    r_ear = processed_data[:, [keypoints.to_idx("r_ear", experiment)]]
    l_ear = processed_data[:, [keypoints.to_idx("l_ear", experiment)]]
    nose = processed_data[:, [keypoints.to_idx("nose", experiment)]]
    middle_head = 0.5 * (l_ear + r_ear)

    processed_data = processed_data - middle_head

    norm_factor = 10
    x_axis = norm_factor * (r_ear - middle_head).squeeze()
    y_axis = norm_factor * (nose - middle_head).squeeze()

    inner_product = np.sum(x_axis * y_axis, axis=1, keepdims=True)
    y_axis = y_axis - inner_product * x_axis
    z_axis = np.cross(x_axis, y_axis)

    # change basis from room coords to head-centered coords.
    basis = np.stack((x_axis, y_axis, z_axis), axis=2)
    cob_matrices = np.linalg.inv(basis)
    processed_data = np.einsum("nij,nkj->nki", cob_matrices, processed_data)

    # for kp in ["head", "nose", "r_ear", "l_ear", "l_eye", "r_eye", "neck"]:
    #     new_poses[:, utils.KEYPOINTS[experiment][kp]] = new_poses[
    #         :, utils.KEYPOINTS[experiment][kp]
    #     ].mean(axis=0)

    processed_data = np.reshape(processed_data, (len(data), -1))

    return processed_data


def _compute_trunk(data, experiment):
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
    processed_data = data.reshape(len(data), -1, 3).copy()

    neck = processed_data[:, keypoints.to_idx("neck", experiment)]
    s_tail = processed_data[:, keypoints.to_idx("s_tail", experiment)]
    r_wrist = processed_data[:, keypoints.to_idx("r_wrist", experiment)]

    processed_data = processed_data - neck[:, None, :]

    x_axis = s_tail - neck
    x_axis /= np.linalg.norm(x_axis, axis=-1, keepdims=True)

    v = neck - r_wrist
    z_axis = v - np.sum(v * x_axis, axis=-1, keepdims=True) * x_axis
    z_axis /= np.linalg.norm(z_axis, axis=-1, keepdims=True)

    y_axis = np.cross(x_axis, z_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=-1)
    R = np.transpose(R, (0, 2, 1))
    processed_data = np.einsum("nij,nkj->nki", R, processed_data)
    processed_data = processed_data.reshape(len(data), -1)

    return processed_data


def _compute_distances(
    data: np.ndarray,
    experiment: str,
    skeleton_type: str = "normal",
):
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
    neckless_skeleton = keypoints.get_neckless_skeleton(skeleton_type)
    processed_data = data.reshape(len(data), -1, 3)
    processed_data = np.stack(
        [
            np.linalg.norm(
                processed_data[:, keypoints.to_idx(s, experiment)]
                - processed_data[:, keypoints.to_idx(e, experiment)],
                axis=1,
            )
            for s, e in neckless_skeleton
        ],
        axis=1,
    ).astype(np.float32)
    return processed_data


def change_coordinates(
    data: np.ndarray,
    coordinate_system: str,
    experiment: str,
    skeleton_type: str = "normal",
) -> np.ndarray:
    """
    Change the body representation of pose data.

    Parameters
    ----------
    data : np.ndarray
        Pose data array.
    representation : str
        Target representation ('allocentric', 'centered', 'egocentric', 'distances', 'trunk').
    experiment : str
        Experiment name (for keypoint selection).

    Returns
    -------
    np.ndarray
        Transformed pose data.
    """
    if coordinate_system == "allocentric":
        processed_data = _compute_allocentric(data, experiment)
    elif coordinate_system == "centered":
        processed_data = _compute_centered(data, experiment)
    elif coordinate_system == "egocentric":
        processed_data = _compute_egocentric(data, experiment)
    elif coordinate_system == "distances":
        processed_data = _compute_distances(data, experiment, skeleton_type)
    elif coordinate_system == "trunk":
        processed_data = _compute_trunk(data, experiment)
    else:
        processed_data = data
        warnings.warn(
            f"Unknown coordinate system: {coordinate_system}, using raw data."
        )
    return processed_data


def exclude_keypoints(
    data: np.ndarray,
    keypoints_list: list[str],
    experiment: str,
) -> np.ndarray:
    """
    Exclude specified keypoints from pose data.

    Parameters
    ----------
    data : np.ndarray
        Pose data array.
    keypoints_list : list[str]
        List of keypoint names to exclude.
    experiment : str
        Experiment name (for keypoint selection).

    Returns
    -------
    np.ndarray
        Pose data with specified keypoints removed.
    """
    if not keypoints_list:
        warnings.warn("poses.exclude_keypoints got an empty keypoints list")
        return data
    processed_data = data.copy()
    indices = [keypoints.to_idx(kp, experiment) for kp in keypoints_list]
    processed_data = np.reshape(processed_data, (len(data), -1, 3))
    processed_data = np.delete(processed_data, indices, axis=1)
    processed_data = np.reshape(processed_data, (len(data), -1))
    return processed_data


def residualize(data: np.ndarray, session_dir: str | pathlib.Path):
    poses_meta_dir = pathlib.Path(session_dir) / "poses" / "meta"
    processed_data = data.copy()
    orth_proj = np.load(poses_meta_dir / "orthogonal_projection.npy")
    processed_data = (orth_proj @ processed_data.T).T
    return processed_data


def preprocess(
    data: np.ndarray,
    cfg: dict,
    session_dir: str | pathlib.Path,
) -> np.ndarray:
    if not cfg:
        warnings.warn("poses.preprocess got an empty config")
        return data

    session_dir = pathlib.Path(session_dir)
    experiment = session_dir.parent.name
    coordinates = cfg.get("coordinates", "egocentric")
    processed_data = data.copy()

    processed_data = change_coordinates(
        processed_data,
        coordinates,
        experiment,
        skeleton_type=cfg.get("skeleton_type", "normal"),
    )

    if "project_to_pcs" in cfg:
        pcs_filename = f"pcs_{coordinates}_{cfg['project_to_pcs']}.pkl"
        pcs_path = pathlib.Path(session_dir) / "poses" / "meta" / pcs_filename
        processed_data = utils.project_to_pc(processed_data, pcs_path)

    if cfg.get("residualize", False):
        processed_data = residualize(processed_data, session_dir)

    if "keypoints_to_exclude" in cfg:
        processed_data = exclude_keypoints(
            processed_data,
            cfg["keypoints_to_exclude"],
            experiment,
        )

    if "dims_to_exclude" in cfg:
        if len(cfg["dims_to_exclude"]) > 0:
            processed_data = np.delete(
                processed_data,
                cfg["dims_to_exclude"],
                axis=1,
            )
        else:
            warnings.warn(
                f"poses.preprocess: no dimension excluded during preprocessing of session: {session_dir}."
            )

    return processed_data


def preprocess_com(data: np.ndarray, config: dict) -> np.ndarray:
    if not config:
        warnings.warn("poses.preprocess_com got an empty config")
        return data

    processed_data = data.copy()

    if "representation" in config:
        if config["representation"] == "com":
            processed_data = processed_data[..., :2]
        elif config["representation"] == "tiles":
            processed_data = room.compute_tiles(
                processed_data[:, 0], processed_data[:, 1]
            )

    return processed_data
