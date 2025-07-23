import random

from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from motorlab import data, utils


class LabeledInterval:
    """
    Represents a labeled interval with start, end, type, and direction.

    Parameters
    ----------
    start : int
        Start index of the interval.
    end : int
        End index of the interval.
    type : str
        Type of the interval (e.g., 'trial', 'homing').
    direction : str
        Direction label (e.g., 'L', 'R').
    """

    def __init__(self, start: int, end: int, type: str, direction: str):
        self.start = start
        self.end = end
        self.type = type
        self.direction = direction

    def __repr__(self):
        return (
            f"Interval({self.start}, {self.end}, {self.type}, {self.direction})"
        )

    def set_tier(self, tier: str):
        """
        Set the tier attribute for the interval.

        Parameters
        ----------
        tier : str
            Tier label (e.g., 'train', 'test', 'validation').
        """
        self.tier = tier

    def filter_sitting(self, tiles, low=3, high=11):
        """
        Filter the interval to keep only subintervals where tile values are within [low, high].

        Parameters
        ----------
        tiles : np.ndarray
            1D array of tile indices (values from 0 to 14).
        low : int, optional
            Lower bound for filtering (inclusive). Default is 3.
        high : int, optional
            Upper bound for filtering (inclusive). Default is 11.
        """
        segment = tiles[self.start : self.end]
        mask = (segment >= low) & (segment <= high)
        idx = np.where(mask)[0]
        self.start += idx[0] if len(idx) > 0 else 0
        self.end = self.start + len(idx) if len(idx) > 0 else self.end

    def balance_intervals(self, tiles, middle_range=(5, 9), alpha=1.2):
        """
        Trim the interval by removing overrepresented values at the edges.

        Parameters
        ----------
        tiles : np.ndarray
            1D array of tile indices (e.g., in range 0--14).
        middle_range : tuple of int, optional
            Inclusive range (low, high) used to define 'balanced' values. Default is (5, 9).
        alpha : float, optional
            Maximum allowed count multiplier relative to the middle median. Default is 1.2.
        """
        mid_values = set(range(middle_range[0], middle_range[1] + 1))
        start, end = self.start, self.end

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

        self.start, self.end = start, end


def get_homing_intervals(trials_dir, tiles, fps=20):
    """
    Extract homing intervals between trials based on tile transitions.

    Parameters
    ----------
    trials_dir : str or Path
        Directory containing trial YAML files.
    tiles : np.ndarray
        Array of tile indices for the session.
    fps : int, optional
        Frames per second. Default is 20.

    Returns
    -------
    list of LabeledInterval
        List of homing intervals as LabeledInterval objects.
    """
    trials_dir = Path(trials_dir)
    frame_duration = 1000 / fps
    threshold = 60 * fps
    start_trials = []
    end_trials = []
    intervals = []

    for trial in sorted(trials_dir.iterdir()):
        with trial.open("rb") as f:
            trial_info = yaml.safe_load(f)
            start = int(trial_info["first_frame_idx"] / frame_duration)
            end = start + int(trial_info["num_frames"] / frame_duration)
            start_trials.append(start)
            end_trials.append(end)

    # the threshold is needed so we know the monkey returned to the starting position "right after" the trial. i plotted the histogram of homing duration and 60 seconds looks like a good threshold.
    for e, s in zip(end_trials, start_trials[1:]):
        if tiles[e] != 12 or tiles[e] != 14:
            continue
        if s - e <= threshold:
            direction = "R" if tiles[e] == 14 else "L"
            intervals.append(LabeledInterval(e, s, "homing", direction))

    return intervals


def get_trials_intervals(trials_dir, tiles, fps=20):
    """
    Extract trial intervals from trial YAML files.

    Parameters
    ----------
    trials_dir : str or Path
        Directory containing trial YAML files.
    tiles : np.ndarray
        Array of tile indices for the session.
    fps : int, optional
        Frames per second. Default is 20.

    Returns
    -------
    list of LabeledInterval
        List of trial intervals as LabeledInterval objects.
    """
    trials_dir = Path(trials_dir)
    frame_duration = 1000 / fps
    intervals = []

    for trial in sorted(trials_dir.iterdir()):
        with trial.open("rb") as f:
            trial_info = yaml.safe_load(f)
            ### to exclude trials, check the trial type here.
            start = int(trial_info["first_frame_idx"] / frame_duration)
            end = start + int(trial_info["num_frames"] / frame_duration)
            intervals.append(
                LabeledInterval(
                    start, end, trial_info["type"], trial_info["choice"]
                )
            )

    return intervals


def get_tiers_intervals(
    data_dir: str | Path,
    sessions: list[str],
    experiment: str,
    include_trial: bool = True,
    include_homing: bool = False,
    include_sitting: bool = True,
    shuffle: bool = True,
    seed: int = 0,
    percent_split: int = 20,
) -> tuple[
    dict[str, list[LabeledInterval]],
    dict[str, list[LabeledInterval]],
    dict[str, list[LabeledInterval]],
]:
    """
    Split intervals into test, train, and validation sets by group.

    Parameters
    ----------
    data_dir : str or Path
        Path to the data directory.
    sessions : list of str
        List of session names.
    experiment : str
        Experiment name.
    include_trial : bool, optional
        Whether to include trial intervals. Default is True.
    include_homing : bool, optional
        Whether to include homing intervals. Default is False.
    include_sitting : bool, optional
        Whether to include sitting intervals. Default is True.
    shuffle : bool, optional
        Whether to shuffle intervals. Default is True.
    seed : int, optional
        Random seed for shuffling. Default is 0.
    percent_split : int, optional
        Percentage of data to use for test and validation (each). Default is 20.

    Returns
    -------
    tuple of dict
        (test_intervals, train_intervals, valid_intervals) for each session.
    """
    data_dir = Path(data_dir)
    intervals = get_intervals(
        data_dir,
        sessions,
        experiment,
        include_trial=include_trial,
        include_homing=include_homing,
        include_sitting=include_sitting,
        shuffle=shuffle,
        seed=seed,
    )

    test_intervals = {session: [] for session in sessions}
    train_intervals = {session: [] for session in sessions}
    valid_intervals = {session: [] for session in sessions}

    for session in sessions:
        session_intervals = intervals[session]

        groups = {}
        for interval in session_intervals:
            key = (interval.type, interval.direction)
            if key not in groups:
                groups[key] = []
            groups[key].append(interval)

        for group_intervals in groups.values():
            n_total = len(group_intervals)
            # at least 1 if group exists
            n_test = max(1, n_total * percent_split // 100)
            n_valid = max(1, n_total * percent_split // 100)

            test_intervals[session].extend(group_intervals[:n_test])
            valid_intervals[session].extend(
                group_intervals[n_test : n_test + n_valid]
            )
            train_intervals[session].extend(group_intervals[n_test + n_valid :])

    return test_intervals, train_intervals, valid_intervals


def get_intervals(
    data_dir: str | Path,
    sessions: list[str],
    experiment: str,
    include_trial: bool = True,
    include_homing: bool = False,
    include_sitting: bool = True,
    shuffle: bool = False,
    seed: int = 0,
) -> dict[str, list[LabeledInterval]]:
    """
    Get intervals for each session, optionally including trial, homing, and sitting intervals.

    Parameters
    ----------
    data_dir : str or Path
        Path to the data directory.
    sessions : list of str
        List of session names.
    experiment : str
        Experiment name.
    include_trial : bool, optional
        Whether to include trial intervals. Default is True.
    include_homing : bool, optional
        Whether to include homing intervals. Default is False.
    include_sitting : bool, optional
        Whether to include sitting intervals. Default is True.
    shuffle : bool, optional
        Whether to shuffle intervals. Default is False.
    seed : int, optional
        Random seed for shuffling. Default is 0.

    Returns
    -------
    dict
        Dictionary mapping session names to lists of LabeledInterval objects.
    """
    data_dir = Path(data_dir)
    intervals = {session: [] for session in sessions}

    for session in sessions:
        tiles = data.load_tiles(data_dir, session, experiment, "tiles")
        trials_dir = data_dir / session / "trials"
        intervals_ = []

        if include_trial:
            intervals_ += get_trials_intervals(trials_dir, tiles)

        if include_homing:
            intervals_ += get_homing_intervals(trials_dir, tiles)

        if not include_sitting:
            intervals_ = [
                interval.filter_sitting(tiles) for interval in intervals_
            ]

        if shuffle:
            utils.fix_seed(seed)
            intervals_ = random.sample(intervals_, k=len(intervals_))

        intervals[session] = intervals_

    return intervals
