from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from motorlab import room


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


def _should_include_interval(
    interval_info: dict, include_trial: bool, include_homing: bool
) -> bool:
    """Check if interval should be included based on type and flags."""
    interval_type = interval_info["type"]
    if not include_homing and interval_type == "homing":
        return False
    if not include_trial and interval_type != "homing":
        return False
    return True


def _process_interval_frames(
    start: int,
    end: int,
    tiles: np.ndarray,
    include_sitting: bool,
    balance_intervals: bool,
) -> tuple[int, int]:
    """Process interval frames with sitting filter and balancing."""
    if not include_sitting:
        start, end = filter_sitting(start, end, tiles)
    if balance_intervals:
        start, end = balance_interval(start, end, tiles)
    return start, end


def load(
    data_dir: str | Path,
    session: str,
    experiment: str,
    include_trial: bool = True,
    include_homing: bool = True,
    include_sitting: bool = True,
    balance_intervals: bool = False,
    sampling_rate: int = 20,
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
    sampling_rate : int, optional
        Sampling rate in Hz. Default is 20.

    Returns
    -------
    dict
        Dictionary mapping session names to lists of LabeledInterval objects.
    """
    data_dir = Path(data_dir)
    experiment_dir = data_dir / experiment
    period = 1000 // sampling_rate

    intervals_dir = data_dir / experiment / session / "intervals"
    intervals = []
    tiles = room.load_location(
        experiment_dir,
        session,
        representation="tiles",
    )

    for interval_file in sorted(intervals_dir.iterdir()):
        with interval_file.open("r") as f:
            interval_info = yaml.safe_load(f)

        if not _should_include_interval(
            interval_info, include_trial, include_homing
        ):
            continue

        # Convert frame indices to sampling rate
        start = interval_info["first_frame_idx"] // period
        num_frames = interval_info["num_frames"] // period
        end = start + num_frames

        start, end = _process_interval_frames(
            start, end, tiles, include_sitting, balance_intervals
        )

        interval = LabeledInterval(
            start=start,
            end=end,
            cue_frame_idx=interval_info["cue_frame_idx"] // period,
            reward=interval_info["reward"],
            side=interval_info["side"],
            tier=interval_info["tier"],
            type=interval_info["type"],
        )
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
    sampling_rate: int = 20,
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
    sampling_rate : int, optional
        Sampling rate in Hz. Default is 20.

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
        sampling_rate=sampling_rate,
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
    sampling_rate: int = 20,
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
    sampling_rate : int, optional
        Sampling rate in Hz. Default is 20.

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
            sampling_rate=sampling_rate,
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
    sampling_rate: int = 20,
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
    sampling_rate : int, optional
        Sampling rate in Hz. Default is 20.

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
            sampling_rate=sampling_rate,
        )

    return intervals


def filter_sitting(start: int, end: int, tiles: np.ndarray) -> tuple[int, int]:
    """
    Filter interval to keep only frames where tile values are within the sitting range.

    Parameters
    ----------
    start : int
        Start frame index
    end : int
        End frame index
    tiles : np.ndarray
        1D array of tile indices

    Returns
    -------
    tuple[int, int]
        Filtered (start, end) frame indices
    """
    low, high = MIDDLE_RANGE
    segment = tiles[start:end]
    mask = (segment >= low) & (segment <= high)
    valid_indices = np.where(mask)[0]

    if len(valid_indices) == 0:
        return start, end

    start += valid_indices[0]
    end = start + len(valid_indices)
    return start, end


def balance_interval(
    start: int, end: int, tiles: np.ndarray
) -> tuple[int, int]:
    """
    Balance interval by removing overrepresented tile values at the edges.

    Trims frames from the beginning and end to ensure each trial has roughly
    the same number of frames per tile value.

    Parameters
    ----------
    start : int
        Start frame index
    end : int
        End frame index
    tiles : np.ndarray
        1D array of tile indices

    Returns
    -------
    tuple[int, int]
        Balanced (start, end) frame indices
    """
    mid_values = set(range(MIDDLE_RANGE[0], MIDDLE_RANGE[1] + 1))

    while start < end:
        segment = tiles[start:end]
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
        left_val, right_val = tiles[start], tiles[end - 1]
        left_count, right_count = counts[left_val], counts[right_val]

        if left_count > threshold and left_count >= right_count:
            start += 1
        elif right_count > threshold:
            end -= 1
        else:
            break

    return start, end
