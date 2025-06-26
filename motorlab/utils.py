import random

from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from motorlab import data


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def filter_sitting(tiles, intervals, low=3, high=11):
    """
    Filters intervals to keep only the subintervals where ts values are within [low, high].

    Parameters:
    - tiles: 1D numpy array of ints representing the tiles (values from 0 to 14)
    - intervals: list of (start, end) tuples
    - low, high: inclusive bounds for filtering

    Returns:
    - filtered_intervals: list of (start, end) tuples where values are within [low, high]
    """
    filtered_intervals = []
    for start, end in intervals:
        segment = tiles[start:end]
        mask = (segment >= low) & (segment <= high)
        idx = np.where(mask)[0]
        filtered_intervals.append([start + idx[0], start + idx[-1] + 1])
    return filtered_intervals


def balance_intervals(tiles, intervals, middle_range=(5, 9), alpha=1.2):
    """
    Trims each interval by removing overrepresented values at the edges.

    Parameters:
    - tiles: 1D numpy array of integers (e.g. in range 0–14)
    - intervals: list of (start, end) tuples defining regions in `tiles`
    - middle_range: inclusive range (low, high) used to define "balanced" values
    - alpha: maximum allowed count multiplier relative to the middle median

    Returns:
    - List of trimmed (start, end) tuples
    """
    trimmed_intervals = []
    mid_values = set(range(middle_range[0], middle_range[1] + 1))

    for start, end in intervals:
        while start < end:
            segment = tiles[start:end]
            counts = Counter(segment)

            mid_counts = [
                count for val, count in counts.items() if val in mid_values
            ]
            median_count = (
                np.median(mid_counts)
                if mid_counts
                else np.median(list(counts.values()))
            )
            threshold = alpha * median_count

            if all(c <= threshold for c in counts.values()):
                break

            left, right = tiles[start], tiles[end - 1]
            left_count = counts[left]
            right_count = counts[right]

            if left_count > threshold and left_count >= right_count:
                start += 1
            elif right_count > threshold:
                end -= 1
            else:
                break

        if start < end:
            trimmed_intervals.append((start, end))

    return trimmed_intervals


def extract_homing_intervals(TRIALS_DIR, fps=20):
    TRIALS_DIR = Path(TRIALS_DIR)
    frame_duration = 1000 / fps
    threshold = 60 * fps
    start_trials = []
    end_trials = []

    for trial in sorted(TRIALS_DIR.iterdir()):
        with trial.open("rb") as f:
            trial_info = yaml.safe_load(f)
            # to exclude trials, check the trial type here.
            start = int(trial_info["first_frame_idx"] / frame_duration)
            end = start + int(trial_info["num_frames"] / frame_duration)
            start_trials.append(start)
            end_trials.append(end)

    # the threshold is needed so we know the monkey returned to the starting position "right after" the trial. i plotted the histogram of homing duration and 60 seconds looks like a good threshold.
    intervals = [
        [e, s]
        for e, s in zip(end_trials, start_trials[1:])
        if s - e <= threshold
    ]

    return intervals


def extract_trials_intervals(TRIALS_DIR, fps=20):
    TRIALS_DIR = Path(TRIALS_DIR)
    frame_duration = 1000 / fps
    intervals = []

    for trial in sorted(TRIALS_DIR.iterdir()):
        with trial.open("rb") as f:
            trial_info = yaml.safe_load(f)
            ### to exclude trials, check the trial type here.
            start = int(trial_info["first_frame_idx"] / frame_duration)
            end = start + int(trial_info["num_frames"] / frame_duration)
            intervals.append([start, end])

    return intervals


def extract_intervals(config):
    test_intervals = dict()
    train_intervals = dict()
    valid_intervals = dict()
    percent = 20

    for session in config["sessions"]:
        TRIALS_DIR = Path(f"{config['DATA_DIR']}/{session}/trials")
        intervals = extract_trials_intervals(TRIALS_DIR)

        if config["homing"]:
            intervals += extract_homing_intervals(TRIALS_DIR)

        if config.get("filter", "") == "sitting":
            tiles = data.load_tiles(
                config["DATA_DIR"], session, config["experiment"]
            )
            intervals = filter_sitting(tiles, intervals)

        shuffled_intervals = random.sample(intervals, k=len(intervals))
        n_intervals = len(shuffled_intervals)

        test_intervals[session] = shuffled_intervals[: n_intervals // percent]
        train_intervals[session] = shuffled_intervals[
            n_intervals // percent : -n_intervals // percent
        ]
        valid_intervals[session] = shuffled_intervals[-n_intervals // percent :]

    return test_intervals, train_intervals, valid_intervals


def compute_tile_distribution(tiles, intervals=None):
    n_of_tiles = 15
    if intervals:
        tiles = np.concatenate([tiles[s:e] for s, e in intervals])
    counts = np.array([(tiles == t).sum() for t in range(n_of_tiles)])
    distr = counts / counts.sum()
    return distr


def list_modalities(modalities):
    if modalities == "all":
        return ["poses", "speed", "acceleration", "spike_count"]
    elif modalities == "spike_count":
        return ["spike_count"]
    elif modalities == "poses":
        return ["poses"]
    elif modalities == "kinematics":
        return ["poses", "speed", "acceleration"]
    elif modalities == "poses_spikes":
        return ["poses", "spike_count"]
    elif modalities == "position":
        return ["position"]
    else:
        raise ValueError(f"unknown modalities: {modalities}.")


def compute_weights(tiles, intervals):
    # i'm using the poses and the valid intervals instead of the tiles itself because the distribution will probably change considerably when we compare the full session and only the valid time points.
    tile_distr = compute_tile_distribution(tiles, intervals)
    weights = torch.tensor(
        np.where(tile_distr > 0.0, 1.0 - tile_distr, 0.0), dtype=torch.float32
    ).to(device)
    return weights


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.use_deterministic_algorithms(True)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_params(model):
    # Core
    core_params = sum(p.numel() for p in model.core.parameters())

    # Embedding per session
    embedding_params = sum(
        sum(p.numel() for p in v.parameters())
        for v in model.embedding.linear.values()
    )

    # Readout per session
    readout_params = sum(
        sum(p.numel() for p in v.parameters())
        for v in model.readout.linear.values()
    )

    return {
        "core": core_params,
        "embedding": embedding_params,
        "readout": readout_params,
    }


def params_per_session(model):
    # Core (shared across sessions)
    core_param_count = sum(p.numel() for p in model.core.parameters())

    # Embedding and readout per session
    session_names = model.embedding.linear.keys()
    result = {}

    for session in session_names:
        emb_count = sum(
            p.numel() for p in model.embedding.linear[session].parameters()
        )
        read_count = sum(
            p.numel() for p in model.readout.linear[session].parameters()
        )
        total = emb_count + core_param_count + read_count

        result[session] = {
            "embedding": 100 * emb_count / total,
            "core": 100 * core_param_count / total,
            "readout": 100 * read_count / total,
            "total_params": total,
        }

    return result
