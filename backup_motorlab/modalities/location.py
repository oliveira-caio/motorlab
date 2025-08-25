from pathlib import Path

import numpy as np

from motorlab.modalities import poses


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
        Poses data as an array of shape (n_timestamps, n_signals)
    """
    session_dir = Path(session_dir)
    return poses.load_com(session_dir, sampling_rate=sampling_rate)


def old_load(
    experiment_dir: Path | str,
    session: str,
    representation: str = "com",
) -> np.ndarray:
    """
    Load location data for a session.

    Parameters
    ----------
    experiment_dir : Path or str
        Path to the experiment directory
    session : str
        Session name
    representation : str, optional
        Location representation ('com' or 'tiles'). Default is 'com'.

    Returns
    -------
    np.ndarray
        Location data as COM coordinates or tile numbers
    """
    experiment_dir = Path(experiment_dir)
    com_data = poses.load_com(experiment_dir, session)

    if representation == "com":
        return com_data
    elif representation == "tiles":
        # com_data[:, 0] and com_data[:, 1] are arrays, so compute_tiles returns array
        return compute_tiles(com_data[:, 0], com_data[:, 1])  # type: ignore[return-value]
    else:
        raise ValueError(
            f"Unknown representation: {representation}. Use 'com' or 'tiles'."
        )


def compute_tiles(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Compute the tile number for given x and y coordinates in the room.

    It works with single floats, but for type checking purposes, we explicitly
    use np.ndarray.

    Parameters
    ----------
    xs : float or np.ndarray
        X coordinates (meters).
    ys : float or np.ndarray
        Y coordinates (meters).

    Returns
    -------
    float or np.ndarray
        Tile number(s) corresponding to the input coordinates.
        Returns float for scalar inputs, np.ndarray for array inputs.
    """
    if isinstance(xs, np.ndarray) and isinstance(ys, np.ndarray):
        assert len(xs) == len(ys), "xs and ys arrays must have the same length"
    else:
        raise ValueError(
            "Both xs and ys must be the same type (both float or both array)"
        )

    tile_width = x_size / x_divisions
    tile_height = y_size / y_divisions

    col = xs // tile_width
    row = ys // tile_height

    # Clamp the indices to stay within bounds.
    col = np.clip(col, 0, x_divisions - 1)
    row = np.clip(row, 0, y_divisions - 1)

    # Tiles are numbered left-to-right, bottom-to-top.
    tile_number = row * x_divisions + col

    return tile_number
