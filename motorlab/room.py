from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Rectangle

from motorlab import poses


# room size
x_size = 2.595
y_size = 4.325

# tiles range from 0 to x_divisions * y_divisions - 1
x_divisions = 3
y_divisions = 5


def load_location(
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


def plot(save_path=None):
    """
    Plot the room layout with tile divisions and optionally save the figure.

    Parameters
    ----------
    save_path : str or Path, optional
        If provided, the plot will be saved to this path. Default is None.

    Returns
    -------
    None
    """
    tile_width = x_size / x_divisions
    tile_height = y_size / y_divisions

    fig, ax = plt.subplots(figsize=(4, 5))

    # Draw the outer rectangle (the room)
    ax.add_patch(
        Rectangle(
            (0, 0), x_size, y_size, edgecolor="black", facecolor="none", lw=4
        )
    )

    tile_number = 0
    for row in range(y_divisions):
        for col in range(x_divisions):
            # Bottom-left corner of the tile
            x0 = col * tile_width
            y0 = row * tile_height

            # Draw tile
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    tile_width,
                    tile_height,
                    edgecolor="gray",
                    facecolor="none",
                )
            )

            # Compute center for the text
            xc = x0 + tile_width / 2
            yc = y0 + tile_height / 2

            ax.text(
                xc,
                yc,
                f"Tile {(col, row)}",
                ha="center",
                va="center",
                fontsize=10,
            )
            tile_number += 1

    # Set ticks at division edges
    ax.set_xticks(np.linspace(0, x_size, x_divisions + 1))
    ax.set_yticks(np.linspace(0, y_size, y_divisions + 1))

    ax.set_xlim(0, x_size)
    ax.set_ylim(0, y_size)
    ax.set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Tile Layout")
    plt.show()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
