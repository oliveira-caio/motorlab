"""Preprocessing tooling for the GBYK dataset.

The data is organized as follows:

## sessions
- each session is a different dataset
- folders inside each session: spikes, spike count, poses, target, and trials
- trials is not a modality, it's just for me to keep track of information when evaluating the model

### spikes
- `data.mem`: timing of each spike of each neuron for the full session. shape: `(n_frames, n_neurons)`
- `meta.yml`: dictionary with metadata, see `metadata` below
- `meta/areas.npy`: array with the brain areas as strings

### spike count
- `data.mem`: spike count (50ms bins) for each neuron for the full session. shape: `(n_frames, n_neurons)`
- `meta.yml`: dictionary with metadata, see `metadata` below
- `meta/areas.npy`: np.array with the brain areas as strings

### poses
- `data.mem`: $(x, y, z)$ coordinates of each keypoint for the full session. shape: `(n_frames, 3*n_keypoints)`
- `meta.yml`: dictionary with metadata, see `metadata` below
- `meta/com.npy`: $(x, y, z)$ coordinate of the center of mass (com) for the full session. shape: `(n_frames, 3)`
- `meta/keypoints.npy`: name of the keypoints tracked
- `meta/skeleton.npy`: adjacency list for all the keypoints

### target
- `data.mem`: something like a np.array with the target information for each frame. 0 before the cue is shown, 1 after the cue was showed if it indicates left and 2 if it indicates right. shape: `(n_frames, 1)`
- `meta.yml`: dictionary with metadata, see `metadata` below

### metadata (meta.yml files)
- `dtype`: necessary to load `.mem` files
- `is_mem_mapped`: if it's `.mem` or `.npy`
- `modality`: sequence or trial
- `n_signals`: number of keypoints, neurons etc
- `n_timestamps`: number of frames
- `sampling_rate`: 100 for poses; 1000 for spikes, target and trials; 20 for spike count
- `start_time`: 0 in my case, i think

### trials
- trial_start: when each trial starts
- trial_end: when each trial ends
- toc (time of commitment): when the monkey committed to a target
- trial_type: precue, gbyk, feedback, homing
- walk_start: when the monkey starts walking
- walk_end: when the monkey stops walking
- cue_start: time of the signal that shows the reward
- cue_end: end of the signal
- side: L, R
- reward: L, R (can be different from side if the monkey ignores the highest reward)

##### to keep in mind
- figure out how to parse the toc.
- how walk_start and walk_end are represented? relative to trial start?
- how is walk_start/mt_on and walk_end/mt_off encoded?
- as of 20250715, the x-axis must be flipped because Irene converted wrongly from Vlad's format.
"""

import os
import shutil

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split

from motorlab import room, utils


def _extract_hdf5_strings(references: list, file: h5py.File) -> list:
    """Extract string data from HDF5 file references.

    Args:
        references: List of HDF5 references to string data
        file: Open HDF5 file object

    Returns:
        List of decoded strings extracted from the references
    """
    return ["".join((chr(x[0]) for x in file[ref[0]])) for ref in references]


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
        "cueend_abstime",
    ]
    trials_info = trials_info[columns_to_keep]
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
    trials_info: pd.DataFrame, threshold: float = 60000
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
        if (
            row_i_plus_1["go_abstime"] - row_i["trialend_abstime"]
        ) >= threshold:
            continue
        new_row = {
            "block": "homing",
            "choice": row_i["choice"],
            "reward": row_i["choice"],
            "go_abstime": row_i["trialend_abstime"],
            "trialend_abstime": row_i_plus_1["go_abstime"],
            "cuestart_abstime": row_i["trialend_abstime"],
            "cueend_abstime": row_i["trialend_abstime"],
        }
        new_rows.append(new_row)

    new_rows_df = pd.DataFrame(new_rows)
    trials_info = pd.concat([trials_info, new_rows_df], ignore_index=True)
    return trials_info.sort_values("go_abstime").reset_index(drop=True)


def _assign_cue_time(trials_info: pd.DataFrame) -> pd.DataFrame:
    """Assign cue timing based on block type.

    Args:
        trials_info: DataFrame with trial information

    Returns:
        DataFrame with added cue timing and frame count columns
    """
    precue_mask = trials_info["block"] == "precue"
    gbyk_mask = trials_info["block"] == "gbyk"
    feedback_mask = trials_info["block"] == "feedback"
    homing_mask = trials_info["block"] == "homing"

    cue = np.empty(len(trials_info))
    cue[precue_mask] = trials_info.loc[precue_mask, "go_abstime"]
    cue[gbyk_mask] = trials_info.loc[gbyk_mask, "cuestart_abstime"]
    cue[feedback_mask] = trials_info.loc[feedback_mask, "trialend_abstime"]
    cue[homing_mask] = trials_info.loc[homing_mask, "go_abstime"]

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
    conditions = trials_info.groupby(["choice", "block"]).groups

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


def intervals(session_dir: Path | str, threshold: int = 60000) -> None:
    """Process trial data and create interval files for a session.

    This function reads trial data from CSV, filters for successful trials,
    creates inter-trial "homing" intervals, splits data into train/test/validation
    sets, and saves individual trial interval files as YAML.

    Args:
        session_dir: Path to the session directory where interval files will be created
        threshold: Unused parameter (kept for backward compatibility)

    Returns:
        None
    """
    session_dir = Path(session_dir)
    session_name = session_dir.name
    data_dir = session_dir.parent
    intervals_dir = session_dir / "intervals"
    intervals_dir.mkdir(exist_ok=True, parents=True)

    trials_info = _load_and_filter_trials(data_dir / f"{session_name}.csv")
    trials_info = _process_reward(trials_info)
    trials_info = _create_homing_intervals(trials_info, threshold=threshold)
    trials_info = _assign_cue_time(trials_info)
    trials_info = _assign_tiers(trials_info)

    trials_info["num_frames"] = (
        trials_info["trialend_abstime"] - trials_info["go_abstime"]
    )

    for idx in range(len(trials_info)):
        with (intervals_dir / f"{idx:03d}.yml").open("w") as f:
            data = {
                "side": trials_info.loc[idx, "choice"],
                "cue_frame_idx": int(trials_info.loc[idx, "cue"]),
                "first_frame_idx": int(trials_info.loc[idx, "go_abstime"]),
                "num_frames": int(trials_info.loc[idx, "num_frames"]),
                # "walk_start": trials_info.loc[idx, "walk_start"],
                # "walk_end": trials_info.loc[idx, "walk_end"],
                "reward": trials_info.loc[idx, "reward"],
                "type": trials_info.loc[idx, "block"],
                "tier": trials_info.loc[idx, "tier"],
            }
            yaml.dump(data, f)


def spikes(session_dir: Path | str, sampling_rate: int = 20) -> None:
    """Extract and save spike data and spike count data from HDF5 file.

    This function reads neural spike data from a MATLAB (.mat) file once,
    converts it to a memory-mapped format for efficient access, and creates
    both the raw spikes data and binned spike count data. This avoids loading
    the large spike data twice.

    Args:
        session_dir: Path to the session directory where spike data will be saved
        sampling_rate: Target sampling rate for spike counts (default: 20Hz)

    Returns:
        None
    """
    session_dir = Path(session_dir)
    data_dir = session_dir.parent

    spikes_dir = session_dir / "spikes"
    spikes_dir.mkdir(exist_ok=True, parents=True)

    spike_count_dir = session_dir / "spike_count"
    spike_count_dir.mkdir(exist_ok=True, parents=True)

    spikes_meta_dir = spikes_dir / "meta"
    spikes_meta_dir.mkdir(exist_ok=True, parents=True)

    spike_count_meta_dir = spike_count_dir / "meta"
    spike_count_meta_dir.mkdir(exist_ok=True, parents=True)

    filename = session_dir.name
    dataset = h5py.File(f"{data_dir}/{filename}.mat")

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
    with open(spikes_dir / "meta.yml", "w") as f:
        meta = {
            "dtype": "float32",
            "end_time": len(spikes),
            "is_mem_mapped": True,
            "modality": "sequence",
            "n_signals": spikes.shape[-1],
            "n_timestamps": len(spikes),
            "sampling_rate": 1000,
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
    period = int(1000 / sampling_rate)
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
            "sampling_rate": sampling_rate,
            "start_time": 0,
        }
        yaml.dump(meta, f)

    ### copy brain areas metadata to spike count directory
    shutil.copytree(spikes_meta_dir, spike_count_meta_dir, dirs_exist_ok=True)


def poses(session_dir: Path | str, old_format: bool = False) -> None:
    """Extract and save pose/motion tracking data from HDF5 file.

    This function reads 3D coordinate data for body keypoints and center of mass
    from MATLAB files, applies coordinate transformations, and saves the data
    in memory-mapped format along with keypoint and skeleton metadata.

    Args:
        session_dir: Path to the session directory where pose data will be saved
        old_format: Whether to use the old format (skips first 5 joints) or new format

    Returns:
        None

    Note:
        The x-axis is flipped for new format data to correct coordinate system conversion.
    """
    session_dir = Path(session_dir)
    data_dir = session_dir.parent
    poses_dir = session_dir / "poses"
    poses_dir.mkdir(exist_ok=True, parents=True)
    meta_dir = poses_dir / "meta"
    meta_dir.mkdir(exist_ok=True, parents=True)

    file_name = session_dir.name
    file_path = Path(f"{data_dir}/{file_name}.mat")
    dataset = h5py.File(file_path)

    ### coords
    # first five joints are useless for the denoised datasets
    if old_format:
        x_com = np.array(dataset["spikes"]["Traj"]["x"][0])
        x_coords = np.array(dataset["spikes"]["Body"]["x"][5:])
        y_coords = np.array(dataset["spikes"]["Body"]["y"][5:])
        z_coords = np.array(dataset["spikes"]["Body"]["z"][5:])
    else:
        x_com = -np.array(dataset["spikes"]["Traj"]["x"][0]) + room.x_size
        x_coords = -np.array(dataset["spikes"]["Body"]["x"]) + room.x_size
        y_coords = np.array(dataset["spikes"]["Body"]["y"])
        z_coords = np.array(dataset["spikes"]["Body"]["z"])

    ### center of mass
    with open(meta_dir / "com.npy", "wb") as f:
        y_com = np.array(dataset["spikes"]["Traj"]["y"][0])
        z_com = np.array(dataset["spikes"]["Traj"]["z"][0])
        com = np.stack([x_com, y_com, z_com], axis=0).T
        np.save(f, com)

    # after next line the shape is (n_joints, 3, n_frames)
    coords = np.stack([x_coords, y_coords, z_coords], axis=1)
    coords = np.reshape(coords, (-1, coords.shape[-1])).T
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

    with open(meta_dir / "keypoints.npy", "wb") as f:
        if old_format:
            np.save(f, list(utils.KEYPOINTS["old_gbyk"].keys()))
        else:
            np.save(f, list(utils.KEYPOINTS["gbyk"].keys()))

    with open(meta_dir / "skeleton.npy", "wb") as f:
        np.save(f, list(utils.SKELETON["normal"]))

    with open(meta_dir / "skeleton_reduced.npy", "wb") as f:
        np.save(f, list(utils.SKELETON["reduced"]))

    assert len(com) == len(coords), f"{len(com)}, {len(coords)}"


def target(session_dir: Path | str) -> None:
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

    trials_dir = session_dir / "trials"
    for trial in trials_dir.iterdir():
        with trial.open("r") as f:
            trial_info = yaml.safe_load(f)
            cue_idx = trial_info["cue_frame_idx"]
            end_idx = trial_info["first_frame_idx"] + trial_info["num_frames"]
            target[cue_idx:end_idx] = 1 if trial_info["reward"] == "R" else 2

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
