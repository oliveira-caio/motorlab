import matplotlib.pyplot as plt
import numpy as np


# room size
x_size = 2.595
y_size = 4.325

# tiles range from 0 to x_divisions * y_divisions - 1
x_divisions = 3
y_divisions = 5


def get_tile(xs, ys):
    tile_width = x_size / x_divisions
    tile_height = y_size / y_divisions

    col = xs // tile_width
    row = ys // tile_height

    # clamp the indices to stay within bounds. just in case the x and y values are out of bound or exact at the boundary of the room. it may occur due to numerical erros in the tracking, for example.
    col = np.clip(col, 0, x_divisions - 1)
    row = np.clip(row, 0, y_divisions - 1)

    # tiles are numbered left-to-right, bottom-to-top.
    tile_number = row * x_divisions + col
    return tile_number


def extract_tiles(com):
    tiles = get_tile(com[:, 0], com[:, 1])
    tiles = tiles[:, np.newaxis]
    return tiles


def plot(save_path=None):
    tile_width = x_size / x_divisions
    tile_height = y_size / y_divisions

    fig, ax = plt.subplots(figsize=(6, 10))

    # Draw the outer rectangle (the room)
    ax.add_patch(
        plt.Rectangle((0, 0), x_size, y_size, edgecolor="black", facecolor="none", lw=4)
    )

    tile_number = 0
    for row in range(y_divisions):
        for col in range(x_divisions):
            # Bottom-left corner of the tile
            x0 = col * tile_width
            y0 = row * tile_height

            # Draw tile
            ax.add_patch(
                plt.Rectangle(
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

            ax.text(xc, yc, f"Tile {(row, col)}", ha="center", va="center", fontsize=12)
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
        fig.savefig(save_path, bbox_inches="tight")
