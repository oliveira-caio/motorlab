import random

from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from motorlab import data, utils


class LabeledInterval:
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
        self.tier = tier

    def filter_sitting(self, tiles, low=3, high=11):
        """
        Filters intervals to keep only the subintervals where tiles values are within [low, high].

        Parameters:
        - tiles: 1D numpy array of ints representing the tiles (values from 0 to 14)
        - low, high: inclusive bounds for filtering
        """
        segment = tiles[self.start : self.end]
        mask = (segment >= low) & (segment <= high)
        idx = np.where(mask)[0]
        self.start += idx[0] if len(idx) > 0 else 0
        self.end = self.start + len(idx) if len(idx) > 0 else self.end

    def balance_intervals(self, tiles, middle_range=(5, 9), alpha=1.2):
        """Trims the interval by removing overrepresented values at the edges.

        Parameters:
        - tiles: 1D numpy array of integers (e.g. in range 0--14)
        - middle_range: inclusive range (low, high) used to define "balanced" values
        - alpha: maximum allowed count multiplier relative to the middle median
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
