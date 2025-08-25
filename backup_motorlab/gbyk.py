"""Export tooling for the GBYK dataset.

For comprehensive documentation of the GBYK dataset structure, data organization,
and preprocessing details, see the gbyk.md file in the repository root.
"""

import os
import shutil

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split

from motorlab import keypoints, room


def _extract_hdf5_strings(references: list, file: h5py.File) -> list:
    """Extract string data from HDF5 file references.

    Args:
        references: List of HDF5 references to string data
        file: Open HDF5 file object

    Returns:
        List of decoded strings extracted from the references
    """
    return ["".join((chr(x[0]) for x in file[ref[0]])) for ref in references]


def export_target(session_dir: Path | str) -> None:
    """Create target signal data indicating reward direction for each time frame.

    This function generates a target signal array where each timepoint is labeled
    with the reward direction: 0 before cue, 1 for left reward, 2 for right reward.
    The target signal is based on trial interval data and cue timing information.

    Args:
        session_dir: Path to the session directory containing trial interval files

    Returns:
        None
    """
    session_dir = Path(session_dir)
    target_dir = session_dir / "target"
    target_dir.mkdir(exist_ok=True, parents=True)

    with (session_dir / "spikes" / "meta.yml").open("r") as f:
        meta_responses = yaml.safe_load(f)
        target = np.zeros(meta_responses["n_timestamps"])

    intervals_dir = session_dir / "intervals"
    for interval in intervals_dir.iterdir():
        with interval.open("r") as f:
            interval_info = yaml.safe_load(f)
            cue_idx = interval_info["cue_frame_idx"]
            end_idx = (
                interval_info["first_frame_idx"] + interval_info["num_frames"]
            )
            target[cue_idx:end_idx] = 1 if interval_info["reward"] == "R" else 2

    mmap = np.memmap(
        target_dir / "data.mem",
        dtype="float32",
        mode="w+",
        shape=target.shape,
    )
    mmap[:] = target[:]
    mmap.flush()

    ### meta
    with open(target_dir / "meta.yml", "w") as f:
        meta = {
            "dtype": "float32",
            "end_time": len(target),
            "is_mem_mapped": True,
            "modality": "sequence",
            "n_signals": target.shape[-1] if len(target.shape) > 1 else 1,
            "n_timestamps": len(target),
            "sampling_rate": 1000,
            "start_time": 0,
        }
        yaml.dump(meta, f)


def _load_and_filter_trials(path: Path) -> pd.DataFrame:
    """Load trial data from CSV and apply initial filtering.

    Args:
        path: Path to the CSV file containing trial data

    Returns:
        Filtered DataFrame with successful non-feedback trials
    """
    trials_info = pd.read_csv(path)
    trials_info = trials_info[
        (trials_info["outcome"] == "success")
        & (trials_info["block"] != "feedback")
    ]
    trials_info.columns = trials_info.columns.map(str.lower)

    columns_to_keep = [
        "choice",
        "go_abstime",
        "reward",
        "block",
        "trialend_abstime",
        "cuestart_abstime",
    ]
    trials_info = trials_info[columns_to_keep]
    columns_to_rename = {
        "go_abstime": "start",
        "block": "type",
        "trialend_abstime": "end",
        "cuestart_abstime": "cuestart",
    }
    trials_info = trials_info.rename(columns=columns_to_rename)
    return trials_info.reset_index(drop=True)


def _process_reward(trials_info: pd.DataFrame) -> pd.DataFrame:
    """Map reward values to L and R labels.

    Args:
        trials_info: DataFrame with trial information

    Returns:
        DataFrame with processed reward column
    """
    # Set the reward column: if reward == 1.8, use the choice value; otherwise, flip the choice ("L" <-> "R")
    trials_info["reward"] = np.where(
        trials_info["reward"] == 1.8,
        trials_info["choice"],
        trials_info["choice"].map({"L": "R", "R": "L"}),
    )
    return trials_info


def _create_homing_intervals(
    trials_info: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """Create inter-trial homing intervals between consecutive trials.

    Args:
        trials_info: DataFrame with trial information

    Returns:
        DataFrame with added homing interval rows
    """
    new_rows = []

    for i in range(len(trials_info) - 1):
        row_i = trials_info.iloc[i]
        row_i_plus_1 = trials_info.iloc[i + 1]
        if (row_i_plus_1["start"] - row_i["end"]) >= threshold:
            continue
        new_row = {
            "type": "homing",
            "choice": row_i["choice"],
            "reward": row_i["choice"],
            "start": row_i["end"],
            "end": row_i_plus_1["start"],
            "cuestart": row_i["end"],
            "cueend": row_i["end"],
        }
        new_rows.append(new_row)

    new_rows_df = pd.DataFrame(new_rows)
    trials_info = pd.concat([trials_info, new_rows_df], ignore_index=True)
    return trials_info.sort_values("start").reset_index(drop=True)


def _assign_cue_time(trials_info: pd.DataFrame) -> pd.DataFrame:
    """Assign cue timing based on type type.

    Args:
        trials_info: DataFrame with trial information

    Returns:
        DataFrame with added cue timing and frame count columns
    """
    precue_mask = trials_info["type"] == "precue"
    gbyk_mask = trials_info["type"] == "gbyk"
    feedback_mask = trials_info["type"] == "feedback"
    homing_mask = trials_info["type"] == "homing"

    cue = np.empty(len(trials_info))
    cue[precue_mask] = trials_info.loc[precue_mask, "start"]
    cue[gbyk_mask] = trials_info.loc[gbyk_mask, "cuestart"]
    cue[feedback_mask] = trials_info.loc[feedback_mask, "end"]
    cue[homing_mask] = trials_info.loc[homing_mask, "start"]

    trials_info["cue"] = cue

    return trials_info


def _assign_tiers(trials_info: pd.DataFrame) -> pd.DataFrame:
    """Assign train/test/validation tiers to trials based on conditions.

    Args:
        trials_info: DataFrame with trial information

    Returns:
        DataFrame with added tier assignments
    """
    trials_info["tier"] = ""
    conditions = trials_info.groupby(["choice", "type"]).groups

    for condition, indices in conditions.items():
        indices_list = list(indices)

        train_indices, temp_indices = train_test_split(
            indices_list, test_size=0.4, random_state=0
        )
        test_indices, val_indices = train_test_split(
            temp_indices, test_size=0.5, random_state=0
        )

        trials_info.loc[train_indices, "tier"] = "train"
        trials_info.loc[test_indices, "tier"] = "test"
        trials_info.loc[val_indices, "tier"] = "validation"

        print(
            f"Condition {condition}: {len(train_indices)} train, {len(test_indices)} test, {len(val_indices)} validation"
        )

    return trials_info


def export_intervals(session_dir: Path | str, homing_threshold: int) -> None:
    """Process trial data and create interval files for a session.

    This function reads trial data from CSV, filters for successful trials,
    creates inter-trial "homing" intervals, splits data into train/test/validation
    sets, and saves individual trial interval files as YAML.

    Args:
        session_dir: Path to the directory where intervals will be created
        homing_threshold: Threshold (in milliseconds) for including the homing
        interval between trials in the dataset

    Returns:
        None
    """
    session_dir = Path(session_dir)
    session_name = session_dir.name
    experiment_dir = session_dir.parent

    intervals_dir = session_dir / "intervals"
    intervals_dir.mkdir(exist_ok=True, parents=True)

    trials_info = _load_and_filter_trials(
        experiment_dir / f"{session_name}.csv"
    )
    trials_info = _process_reward(trials_info)
    trials_info = _create_homing_intervals(
        trials_info, threshold=homing_threshold
    )
    trials_info = _assign_cue_time(trials_info)
    trials_info = _assign_tiers(trials_info)

    trials_info["duration"] = trials_info["end"] - trials_info["start"]

    trials_info.to_csv(intervals_dir / "data.csv")

    with open(intervals_dir / "meta.yml", "w") as f:
        meta = {
            "modality": "labeled_interval",
            "sampling_rate": 1000,
        }
        yaml.dump(meta, f)


def _transform_coordinates(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    denoised_format: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_size, y_size = room.get_size()

    if denoised_format:
        x_axis = y_size * (x_axis + (x_size / y_size / 2))
        y_axis = y_size * y_axis
        z_axis = y_size * z_axis
    else:
        x_axis = -x_axis + x_size
        y_axis = y_axis
        z_axis = z_axis

    return x_axis, y_axis, z_axis


def export_poses(
    session_dir: Path | str,
    denoised_format: bool,
) -> None:
    """Extract and save pose/motion tracking data from HDF5 file.

    This function reads 3D coordinate data for body keypoints and center of mass
    from MATLAB files, applies coordinate transformations, and saves the data
    in memory-mapped format along with keypoint and skeleton metadata.

    Args:
        session_dir: Path to the session directory where pose data will be saved
        denoised_format: Whether to use the denoised format (skips first 5 joints) or new format

    Returns:
        None

    Note:
        The x-axis is flipped for new format data to correct coordinate system conversion.
    """
    session_dir = Path(session_dir)
    session_name = session_dir.name
    experiment_dir = session_dir.parent

    poses_dir = session_dir / "poses"
    poses_dir.mkdir(exist_ok=True, parents=True)

    meta_dir = poses_dir / "meta"
    meta_dir.mkdir(exist_ok=True, parents=True)

    file_path = experiment_dir / f"{session_name}.mat"
    dataset = h5py.File(file_path)

    ### center of mass
    with open(meta_dir / "com.npy", "wb") as f:
        x_com = np.array(dataset["spikes"]["Traj"]["x"][0])
        y_com = np.array(dataset["spikes"]["Traj"]["y"][0])
        z_com = np.array(dataset["spikes"]["Traj"]["z"][0])
        x_com, y_com, z_com = _transform_coordinates(
            x_com,
            y_com,
            z_com,
            denoised_format,
        )
        com = np.stack([x_com, y_com, z_com], axis=0)
        com = np.transpose(com)
        com = com.astype(np.float32)
        np.save(f, com)

    ### coords
    x_coords = np.array(dataset["spikes"]["Body"]["x"])
    y_coords = np.array(dataset["spikes"]["Body"]["y"])
    z_coords = np.array(dataset["spikes"]["Body"]["z"])

    if denoised_format:
        # first 5 keypoints are metadata for the old (denoised) datasets.
        x_coords = x_coords[5:]
        y_coords = y_coords[5:]
        z_coords = z_coords[5:]

    x_coords, y_coords, z_coords = _transform_coordinates(
        x_coords,
        y_coords,
        z_coords,
        denoised_format,
    )

    # after next line the shape is (n_joints, 3, n_frames)
    coords = np.stack([x_coords, y_coords, z_coords], axis=1)
    coords = np.reshape(coords, (-1, coords.shape[-1]))
    coords = np.transpose(coords)
    coords = coords.astype(np.float32)
    mmap = np.memmap(
        os.path.join(poses_dir, "data.mem"),
        dtype="float32",
        mode="w+",
        shape=coords.shape,
    )
    mmap[:] = coords
    mmap.flush()

    ### meta
    with open(poses_dir / "meta.yml", "w") as f:
        meta = {
            "dtype": "float32",
            "end_time": len(coords),
            "is_mem_mapped": True,
            "modality": "sequence",
            "n_signals": coords.shape[-1],
            "n_timestamps": len(coords),
            "sampling_rate": 100,
            "start_time": 0,
        }
        yaml.dump(meta, f)

    with open(meta_dir / "keypoints.yml", "w") as f:
        if denoised_format:
            yaml.safe_dump(keypoints.get("old_gbyk"), f)
        else:
            yaml.safe_dump(keypoints.get("gbyk"), f)

    with open(meta_dir / "skeleton.npy", "wb") as f:
        np.save(f, keypoints.get_skeleton("normal"))

    with open(meta_dir / "skeleton_reduced.npy", "wb") as f:
        np.save(f, keypoints.get_skeleton("reduced"))

    assert len(com) == len(coords), f"{len(com)}, {len(coords)}"


def export_spikes(
    session_dir: Path | str,
    spike_count_sampling_rate: int,
) -> None:
    """Extract and save spike data and spike count data from HDF5 file.

    This function reads neural spike data from a MATLAB (.mat) file once,
    converts it to a memory-mapped format for efficient access, and creates
    both the raw spikes data and binned spike count data. This avoids loading
    the large spike data twice.

    Args:
        session_dir: Path to the directory where spike data will be saved
        spike_count_sampling_rate: Target spike count sampling rate

    Returns:
        None
    """
    session_dir = Path(session_dir)
    session_name = session_dir.name
    experiment_dir = session_dir.parent

    spikes_dir = session_dir / "spikes"
    spikes_dir.mkdir(exist_ok=True, parents=True)

    spikes_meta_dir = spikes_dir / "meta"
    spikes_meta_dir.mkdir(exist_ok=True, parents=True)

    spike_count_dir = session_dir / "spike_count"
    spike_count_dir.mkdir(exist_ok=True, parents=True)

    spike_count_meta_dir = spike_count_dir / "meta"
    spike_count_meta_dir.mkdir(exist_ok=True, parents=True)

    dataset = h5py.File(f"{experiment_dir}/{session_name}.mat")

    ### spikes
    spikes = np.array(dataset["spikes"]["session"][:])

    ### save raw spike training
    spikes_mmap = np.memmap(
        spikes_dir / "data.mem",
        dtype="float32",
        mode="w+",
        shape=spikes.shape,
    )
    spikes_mmap[:] = spikes[:]
    spikes_mmap.flush()

    ### save spikes metadata
    spikes_sampling_rate = 1000
    with open(spikes_dir / "meta.yml", "w") as f:
        meta = {
            "dtype": "float32",
            "end_time": len(spikes),
            "is_mem_mapped": True,
            "modality": "event",
            "n_signals": spikes.shape[-1],
            "n_timestamps": len(spikes),
            "sampling_rate": spikes_sampling_rate,
            "start_time": 0,
        }
        yaml.dump(meta, f)

    ### save brain areas data
    with open(spikes_meta_dir / "areas.npy", "wb") as f:
        areas = np.array(
            _extract_hdf5_strings(dataset["spikes"]["array_labels"], dataset)
        )
        array_code = dataset["spikes"]["array_code"][:].ravel() - 1  # 1-based
        np.save(f, np.char.lower(areas[array_code.astype(int)]))

    assert len(array_code) == spikes.shape[-1], (
        f"{len(array_code)}, {spikes.shape[-1]}"
    )

    ### create spike count data from loaded spikes
    period = int(spikes_sampling_rate / spike_count_sampling_rate)
    duration = len(spikes) - (len(spikes) % period)
    spike_count = (
        spikes[:duration].reshape(len(spikes) // period, period, -1).sum(axis=1)
    )

    ### Save spike count data
    spike_count_mmap = np.memmap(
        spike_count_dir / "data.mem",
        dtype="float32",
        mode="w+",
        shape=spike_count.shape,
    )
    spike_count_mmap[:] = spike_count[:]
    spike_count_mmap.flush()

    ### save spike count metadata
    with open(spike_count_dir / "meta.yml", "w") as f:
        meta = {
            "dtype": "float32",
            "end_time": len(spike_count),
            "is_mem_mapped": True,
            "modality": "sequence",
            "n_signals": spike_count.shape[-1],
            "n_timestamps": len(spike_count),
            "sampling_rate": spike_count_sampling_rate,
            "start_time": 0,
        }
        yaml.dump(meta, f)

    ### copy brain areas metadata to spike count directory
    shutil.copytree(spikes_meta_dir, spike_count_meta_dir, dirs_exist_ok=True)
