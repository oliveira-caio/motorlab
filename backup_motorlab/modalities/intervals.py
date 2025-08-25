from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from motorlab.modalities import location


MIDDLE_RANGE = (3, 11)


class LabeledInterval:
    """
    Represents a labeled interval with start/end frames and metadata.

    Attributes
    ----------
    start : int
        Start frame index
    end : int
        End frame index
    num_frames : int
        Number of frames in the interval
    cue_frame_idx : int
        Frame index when cue was presented
    reward : str
        Reward type
    side : str
        Side information
    tier : str
        Data tier (train/test/validation)
    type : str
        Interval type (trial/homing/etc)
    """

    def __init__(
        self,
        start: int,
        end: int,
        cue_frame_idx: int,
        reward: str,
        side: str,
        tier: str,
        type: str,
    ):
        self.start = start
        self.end = end
        self.num_frames = end - start
        self.cue_frame_idx = cue_frame_idx
        self.reward = reward
        self.side = side
        self.tier = tier
        self.type = type

    def __repr__(self) -> str:
        return (
            f"LabeledInterval(start={self.start}, end={self.end}, "
            f"type={self.type}, tier={self.tier})"
        )

    def __iter__(self):
        """Allow unpacking like (start, end) = interval."""
        yield self.start
        yield self.end

    def filter_sitting(self, tiles: np.ndarray):
        """
        Filter interval to keep only frames where tile values are within the sitting range.

        Parameters
        ----------
        tiles : np.ndarray
            1D array of tile indices

        Returns
        -------
        tuple[int, int]
            Filtered (start, end) frame indices
        """
        low, high = MIDDLE_RANGE
        segment = tiles[self.start : self.end]
        mask = (segment >= low) & (segment <= high)
        valid_indices = np.where(mask)[0]
        self.start += valid_indices[0]
        self.end = self.start + len(valid_indices)

    def balance(self, tiles: np.ndarray):
        """
        Balance interval by removing overrepresented tile values at the edges.

        Trims frames from the beginning and end to ensure each trial has roughly
        the same number of frames per tile value.

        Parameters
        ----------
        tiles : np.ndarray
            1D array of tile indices

        Returns
        -------
        tuple[int, int]
            Balanced (start, end) frame indices
        """
        mid_values = set(range(MIDDLE_RANGE[0], MIDDLE_RANGE[1] + 1))

        while self.start < self.end:
            segment = tiles[self.start : self.end]
            counts = Counter(segment)

            # Calculate threshold based on median count of middle values
            mid_counts = [
                count for val, count in counts.items() if val in mid_values
            ]
            median_count = (
                np.median(mid_counts)
                if mid_counts
                else np.median(list(counts.values()))
            )
            threshold = 1.2 * median_count

            # Check if balancing is complete
            if all(count <= threshold for count in counts.values()):
                break

            # Remove from the edge with higher overrepresentation
            left_val, right_val = tiles[self.start], tiles[self.end - 1]
            left_count, right_count = counts[left_val], counts[right_val]

            if left_count > threshold and left_count >= right_count:
                self.start += 1
            elif right_count > threshold:
                self.end -= 1
            else:
                break


def _include_interval(
    interval_type: str, include_trial: bool, include_homing: bool
) -> bool:
    """Check if interval should be included based on type and flags."""
    if not include_homing and interval_type == "homing":
        return False
    if not include_trial and interval_type != "homing":
        return False
    return True


def load(session_dir: Path | str, query: dict, sampling_rate: int) -> list:
    session_dir = Path(session_dir)

    with open(session_dir / "meta.yml", "r") as f:
        meta_data = yaml.safe_load(f)

    intervals_df = pd.read_csv(session_dir / "data.csv")

    filtered_df = intervals_df[
        (intervals_df[list(query)] == pd.Series(query)).all(axis=1)
    ]
    factor = meta_data["sampling_rate"] // sampling_rate
    filtered_df["start"] = (filtered_df["start"] // factor).astype(int)
    filtered_df["end"] = (filtered_df["end"] // factor).astype(int)
    intervals_data = list(zip(filtered_df["start"], filtered_df["end"]))

    return intervals_data


def old_load(
    data_dir: str | Path,
    session: str,
    experiment: str,
    include_trial: bool = True,
    include_homing: bool = True,
    include_sitting: bool = True,
    balance_intervals: bool = False,
) -> list[LabeledInterval]:
    """
    Load intervals for a single session.

    Parameters
    ----------
    data_dir : str or Path
        Path to the intervals directory.
    session : str
        Session name.
    experiment : str
        Experiment name.
    include_trial : bool, optional
        Whether to include trial intervals. Default is True.
    include_homing : bool, optional
        Whether to include homing intervals. Default is True.
    include_sitting : bool, optional
        Whether to include sitting intervals. Default is True.
    balance_intervals : bool, optional
        Whether to balance intervals. Default is False.

    Returns
    -------
    dict
        Dictionary mapping session names to lists of LabeledInterval objects.
    """
    data_dir = Path(data_dir)
    experiment_dir = data_dir / experiment
    period = MODALITY_SAMPLING_RATE // RUN_SAMPLING_RATE

    intervals_dir = data_dir / experiment / session / "intervals"
    intervals = []
    tiles = location.load(
        experiment_dir,
        session,
        representation="tiles",
    )

    for interval_file in sorted(intervals_dir.iterdir()):
        with interval_file.open("r") as f:
            interval_info = yaml.safe_load(f)

        if not _include_interval(
            interval_info["type"],
            include_trial,
            include_homing,
        ):
            continue

        # Convert frame indices to sampling rate
        start = interval_info["first_frame_idx"] // period
        num_frames = interval_info["num_frames"] // period
        end = start + num_frames

        interval = LabeledInterval(
            start=start,
            end=end,
            cue_frame_idx=interval_info["cue_frame_idx"] // period,
            reward=interval_info["reward"],
            side=interval_info["side"],
            tier=interval_info["tier"],
            type=interval_info["type"],
        )

        if not include_sitting:
            interval.filter_sitting(tiles)
        if balance_intervals:
            interval.balance(tiles)

        intervals.append(interval)

    return intervals


def load_by_tiers(
    data_dir: str | Path,
    session: str,
    experiment: str,
    include_trial: bool = True,
    include_homing: bool = True,
    include_sitting: bool = True,
    balance_intervals: bool = False,
) -> tuple[list[LabeledInterval], list[LabeledInterval], list[LabeledInterval]]:
    """
    Load intervals for a single session and split by tiers (test, train, validation).

    Parameters
    ----------
    data_dir : str or Path
        Path to the data directory.
    session : str
        Session name.
    experiment : str
        Experiment name.
    include_trial : bool, optional
        Whether to include trial intervals. Default is True.
    include_homing : bool, optional
        Whether to include homing intervals. Default is True.
    include_sitting : bool, optional
        Whether to include sitting intervals. Default is True.
    balance_intervals : bool, optional
        Whether to balance intervals. Default is False.

    Returns
    -------
    tuple of list
        (test_intervals, train_intervals, valid_intervals) for the session.
    """
    intervals = load(
        data_dir=data_dir,
        session=session,
        experiment=experiment,
        include_trial=include_trial,
        include_homing=include_homing,
        include_sitting=include_sitting,
        balance_intervals=balance_intervals,
    )

    test_intervals = [
        interval for interval in intervals if interval.tier == "test"
    ]
    train_intervals = [
        interval for interval in intervals if interval.tier == "train"
    ]
    valid_intervals = [
        interval for interval in intervals if interval.tier == "validation"
    ]

    return test_intervals, train_intervals, valid_intervals


def load_all_by_tiers(
    data_dir: str | Path,
    sessions: list[str],
    experiment: str,
    include_trial: bool = True,
    include_homing: bool = True,
    include_sitting: bool = True,
    balance_intervals: bool = False,
) -> tuple[
    dict[str, list[LabeledInterval]],
    dict[str, list[LabeledInterval]],
    dict[str, list[LabeledInterval]],
]:
    """
    Load intervals and split by tiers (test, train, validation).

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
        Whether to include homing intervals. Default is True.
    include_sitting : bool, optional
        Whether to include sitting intervals. Default is True.
    balance_intervals : bool, optional
        Whether to balance intervals. Default is False.

    Returns
    -------
    tuple of dict
        (test_intervals, train_intervals, valid_intervals) for each session.
    """
    test_intervals = {}
    train_intervals = {}
    valid_intervals = {}

    for session in sessions:
        test, train, valid = load_by_tiers(
            data_dir=data_dir,
            session=session,
            experiment=experiment,
            include_trial=include_trial,
            include_homing=include_homing,
            include_sitting=include_sitting,
            balance_intervals=balance_intervals,
        )
        test_intervals[session] = test
        train_intervals[session] = train
        valid_intervals[session] = valid

    return test_intervals, train_intervals, valid_intervals


def load_all(
    data_dir: str | Path,
    sessions: list[str],
    experiment: str,
    include_trial: bool = True,
    include_homing: bool = True,
    include_sitting: bool = True,
    balance_intervals: bool = False,
) -> dict[str, list[LabeledInterval]]:
    """
    Load all intervals for each session with optional filtering and processing.

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
        Whether to include homing intervals. Default is True.
    include_sitting : bool, optional
        Whether to include sitting intervals. Default is True.
    balance_intervals : bool, optional
        Whether to balance intervals. Default is False.

    Returns
    -------
    dict
        Dictionary mapping session names to lists of LabeledInterval objects.
    """
    intervals = {}

    for session in sessions:
        intervals[session] = load(
            data_dir=data_dir,
            session=session,
            experiment=experiment,
            include_trial=include_trial,
            include_homing=include_homing,
            include_sitting=include_sitting,
            balance_intervals=balance_intervals,
        )

    return intervals


def align(
    data: np.ndarray,
    intervals: list[list[int]],
    overlap: int = 10,
    method: str = "end",
) -> np.ndarray:
    """
    Align intervals of data to a common length, either by start or end.

    This function takes a data array of shape (n_frames, n_features) and a list
    of intervals, where each interval is a (start, end) pair. It returns a new
    array of shape (n_intervals, max_len, n_features), where each interval is
    aligned either by its start or end.

    If aligning by 'start', all intervals are assumed to start at the same
    time point, and the array is padded with NaNs at the end as needed.
    If aligning by 'end', all intervals are assumed to end at the same time
    point, and the array is padded with NaNs at the beginning.

    To determine max_len, the function finds the maximum interval length that
    is shared by at least `overlap + 1` intervals. This strategy reduces
    variance when averaging over intervals.

    Parameters
    ----------
    data : np.ndarray
        Data array of shape (n_frames, n_features).
    intervals : list of list of int
        List of (start, end) intervals.
    overlap : int, optional
        Minimum number of intervals with the same length to consider for max_len. Default is 10.
    method : {'start', 'end'}, optional
        'start' to align by start, 'end' to align by end. Default is 'end'.

    Returns
    -------
    np.ndarray
        Aligned data of shape (n_intervals, max_len, n_features) with NaNs for padding.
    """
    lengths = [e - s + 1 for s, e in intervals]
    lengths = np.array(lengths)
    max_len = max(
        [
            length
            for length in lengths
            if np.sum(lengths >= length) >= (overlap + 1)
        ]
    )

    aligned_data = np.full((len(intervals), max_len, data.shape[-1]), np.nan)
    for i, (start, end) in enumerate(intervals):
        if method == "start":
            aligned_data[i, : end - start + 1] = (
                data[start : end + 1]
                if end - start + 1 <= max_len
                else data[start : start + max_len]
            )
        elif method == "end":
            aligned_data[i, -(end - start + 1) :] = (
                data[start : end + 1]
                if end - start + 1 <= max_len
                else data[end - max_len + 1 : end + 1]
            )

    return aligned_data
