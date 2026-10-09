#!/usr/bin/env python3
"""
Photo Mosaic Generator (OpenCV + NumPy)

Rebuilds a target image out of many small "tile" images. Each grid block of the
target is replaced by the tile whose average colour is closest (Euclidean
distance in BGR space) to the block's own average colour.

Quick start: edit the three paths in the CONFIG section below, then run:

    python photo_mosaic.py

...or override anything from the command line:

    python photo_mosaic.py --tiles-dir tiles --target photo.jpg --output out.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# CONFIG  -- set your file paths and defaults here (or use command-line flags)
# ----------------------------------------------------------------------------
TILES_DIR = "tiles"          # folder containing the small source images
TARGET_IMAGE = "target.jpg"  # the big picture you want to turn into a mosaic
OUTPUT_IMAGE = "mosaic.jpg"  # where the result will be saved
TILE_SIZE = 50               # each tile becomes TILE_SIZE x TILE_SIZE pixels
TARGET_COLS = None           # optional: number of tiles across (None = keep size)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CHUNK_SIZE = 4096            # blocks matched per batch (limits peak memory use)


# ----------------------------------------------------------------------------
# Step 1: load source images and resize them to a uniform tile size
# ----------------------------------------------------------------------------
def load_tiles(tiles_dir: Path, tile_size: int) -> np.ndarray:
    """Return all valid images in `tiles_dir` as an array (N, T, T, 3), uint8."""
    tiles = []
    for path in sorted(tiles_dir.iterdir()):
        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # always 3-channel BGR
        if img is None:  # corrupt or unreadable file
            print(f"  skipping unreadable file: {path.name}")
            continue

        # Centre-crop to a square first so tiles are not stretched/distorted.
        h, w = img.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        img = img[y0:y0 + side, x0:x0 + side]

        # INTER_AREA gives the best quality when shrinking images.
        tiles.append(cv2.resize(img, (tile_size, tile_size),
                                interpolation=cv2.INTER_AREA))

    if not tiles:
        sys.exit(f"No valid images found in '{tiles_dir}'.")
    return np.stack(tiles)


# ----------------------------------------------------------------------------
# Step 2: average colour of every tile
# ----------------------------------------------------------------------------
def average_colors(tiles: np.ndarray) -> np.ndarray:
    """Mean BGR colour of each tile -> float32 array of shape (N, 3)."""
    # (N, T, T, 3) -> average over the two spatial axes in one vectorised call.
    return tiles.mean(axis=(1, 2), dtype=np.float32)


# ----------------------------------------------------------------------------
# Step 3: prepare the target image
# ----------------------------------------------------------------------------
def prepare_target(path: Path, tile_size: int, cols: int | None) -> np.ndarray:
    """Load the target and make its width/height exact multiples of tile_size."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"Could not read target image '{path}'.")

    # Optionally rescale so the mosaic is exactly `cols` tiles wide
    # (keeps the aspect ratio). Handy for huge photos.
    if cols:
        new_w = cols * tile_size
        new_h = max(tile_size, round(img.shape[0] * new_w / img.shape[1]))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Centre-crop away the leftover pixels so dimensions divide evenly.
    h, w = img.shape[:2]
    new_h, new_w = (h // tile_size) * tile_size, (w // tile_size) * tile_size
    if new_h == 0 or new_w == 0:
        sys.exit("Target image is smaller than a single tile.")
    y0, x0 = (h - new_h) // 2, (w - new_w) // 2
    return img[y0:y0 + new_h, x0:x0 + new_w]


# ----------------------------------------------------------------------------
# Step 4: grid division and best-match search
# ----------------------------------------------------------------------------
def block_averages(target: np.ndarray, tile_size: int) -> np.ndarray:
    """Average colour of every grid block -> float32 array (rows, cols, 3)."""
    h, w = target.shape[:2]
    rows, cols = h // tile_size, w // tile_size
    # Reshape into (rows, T, cols, T, 3) so each block owns axes 1 and 3,
    # then average over those axes. No Python loops needed.
    blocks = target.reshape(rows, tile_size, cols, tile_size, 3)
    return blocks.mean(axis=(1, 3), dtype=np.float32)


def find_best_matches(block_colors: np.ndarray, tile_colors: np.ndarray) -> np.ndarray:
    """
    For each block colour, return the index of the closest tile colour.

    Uses the expansion  ||b - t||^2 = ||b||^2 - 2*b.t + ||t||^2.
    ||b||^2 is the same for every tile, so it cannot change the argmin and is
    dropped. What remains is a single matrix multiplication (fast BLAS) instead
    of a Python loop over every block/tile pair. Work is done in chunks so a
    big grid x big tile library never needs one enormous distance matrix.
    """
    flat = block_colors.reshape(-1, 3)                    # (B, 3)
    tile_sq = (tile_colors ** 2).sum(axis=1)              # (N,)
    best = np.empty(len(flat), dtype=np.int64)

    for start in range(0, len(flat), CHUNK_SIZE):
        chunk = flat[start:start + CHUNK_SIZE]            # (c, 3)
        # (c, N) pseudo-distance matrix: tile_sq - 2 * chunk @ tile_colors.T
        dist = tile_sq[None, :] - 2.0 * (chunk @ tile_colors.T)
        best[start:start + CHUNK_SIZE] = dist.argmin(axis=1)

    return best.reshape(block_colors.shape[:2])           # (rows, cols)


# ----------------------------------------------------------------------------
# Step 5: assemble the mosaic
# ----------------------------------------------------------------------------
def build_mosaic(tiles: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Place the chosen tile in every grid cell and return the full image."""
    rows, cols = indices.shape
    t = tiles.shape[1]
    # Fancy indexing fetches every chosen tile at once: (rows, cols, T, T, 3).
    chosen = tiles[indices]
    # Interleave the axes -> (rows, T, cols, T, 3), then flatten to an image.
    return chosen.transpose(0, 2, 1, 3, 4).reshape(rows * t, cols * t, 3)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a photo mosaic.")
    p.add_argument("--tiles-dir", default=TILES_DIR, help="folder of source images")
    p.add_argument("--target", default=TARGET_IMAGE, help="target image path")
    p.add_argument("--output", default=OUTPUT_IMAGE, help="output file path")
    p.add_argument("--tile-size", type=int, default=TILE_SIZE, help="tile edge in px")
    p.add_argument("--cols", type=int, default=TARGET_COLS,
                   help="resize target to this many tiles across (optional)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tiles_dir = Path(args.tiles_dir)
    if not tiles_dir.is_dir():
        sys.exit(f"Tiles directory '{tiles_dir}' does not exist.")

    print("Loading tile images...")
    tiles = load_tiles(tiles_dir, args.tile_size)
    tile_colors = average_colors(tiles)
    print(f"  {len(tiles)} tiles loaded ({args.tile_size}x{args.tile_size}px)")

    print("Preparing target image...")
    target = prepare_target(Path(args.target), args.tile_size, args.cols)
    block_colors = block_averages(target, args.tile_size)
    rows, cols = block_colors.shape[:2]
    print(f"  grid: {cols} x {rows} = {rows * cols} blocks")

    print("Matching blocks to tiles...")
    indices = find_best_matches(block_colors, tile_colors)

    print("Building mosaic...")
    mosaic = build_mosaic(tiles, indices)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), mosaic):
        sys.exit(f"Failed to write '{out}' (check the file extension).")
    print(f"Saved {mosaic.shape[1]}x{mosaic.shape[0]} mosaic to '{out}'")


if __name__ == "__main__":
    main()
