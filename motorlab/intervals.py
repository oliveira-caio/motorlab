from pathlib import Path

import pandas as pd
import yaml


def load_as_df(
    session_dir: Path | str,
    sampling_rate: int,
) -> pd.DataFrame:
    session_dir = Path(session_dir)
    intervals_dir = session_dir / "intervals"

    with open(intervals_dir / "meta.yml", "r") as f:
        meta = yaml.safe_load(f)

    factor = meta["sampling_rate"] // sampling_rate
    intervals = pd.read_csv(intervals_dir / "data.csv")
    intervals["start"] = intervals["start"] // factor
    # i'm subtracting 1 because session old_gbyk/ken_20230618_denoised loads
    # a single row of nans for the poses at the end if i don't do this.
    intervals["end"] = intervals["end"] // factor - 1
    intervals["duration"] = intervals["end"] - intervals["start"]
    return intervals


def load(
    session_dir: Path | str,
    query: dict,
    sampling_rate: int,
) -> list[tuple[int, int]]:
    """Load interval data for a specific query and sampling rate."""
    intervals_df = load_as_df(session_dir, sampling_rate)

    mask = pd.Series(True, index=intervals_df.index)
    for col, val in query.items():
        mask &= intervals_df[col] == val
    filtered_df = intervals_df[mask]

    intervals_data = list(
        zip(filtered_df["start"].astype(int), filtered_df["end"].astype(int))
    )
    return intervals_data
