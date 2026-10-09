#!/usr/bin/env python3
"""
Photo Mosaic Generator (OpenCV + NumPy)  -- image, MP4 video, or live webcam

Each grid block of the input is replaced by the tile whose average colour is
closest to the block's average colour, BUT every tile is used as evenly as
possible. If you have at least as many tiles as blocks, no tile is repeated.
If you have fewer, each tile is used at most ceil(blocks / tiles) times.

Examples:
    python photo_mosaic.py --target photo.jpg
    python photo_mosaic.py --video input.mp4 --output out.mp4
    python photo_mosaic.py --webcam            # press 'q' to quit
    python photo_mosaic.py --webcam --output webcam.mp4
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
TILES_DIR = "tiles"
TARGET_IMAGE = "target.jpg"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Defaults depend on mode (video needs smaller output to stay fast)
IMAGE_TILE_SIZE = 50
VIDEO_TILE_SIZE = 16
VIDEO_COLS = 48     # tiles across for video/webcam when --cols is not given


# ----------------------------------------------------------------------------
# Tiles
# ----------------------------------------------------------------------------
def load_tiles(tiles_dir: Path, tile_size: int) -> np.ndarray:
    """Return all valid images in `tiles_dir` as an array (N, T, T, 3), uint8."""
    tiles = []
    for path in sorted(tiles_dir.iterdir()):
        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  skipping unreadable file: {path.name}")
            continue
        h, w = img.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        img = img[y0:y0 + side, x0:x0 + side]
        tiles.append(cv2.resize(img, (tile_size, tile_size),
                                interpolation=cv2.INTER_AREA))
    if not tiles:
        sys.exit(f"No valid images found in '{tiles_dir}'.")
    return np.stack(tiles)


def average_colors(tiles: np.ndarray) -> np.ndarray:
    return tiles.mean(axis=(1, 2), dtype=np.float32)          # (N, 3)


# ----------------------------------------------------------------------------
# Target frame -> grid of block colours
# ----------------------------------------------------------------------------
def prepare_target(img: np.ndarray, tile_size: int, cols: int | None) -> np.ndarray:
    """Rescale (optional) and crop so width/height are multiples of tile_size."""
    if cols:
        new_w = cols * tile_size
        new_h = max(tile_size, round(img.shape[0] * new_w / img.shape[1]))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    h, w = img.shape[:2]
    new_h, new_w = (h // tile_size) * tile_size, (w // tile_size) * tile_size
    if new_h == 0 or new_w == 0:
        sys.exit("Input is smaller than a single tile.")
    y0, x0 = (h - new_h) // 2, (w - new_w) // 2
    return img[y0:y0 + new_h, x0:x0 + new_w]


def block_averages(target: np.ndarray, tile_size: int) -> np.ndarray:
    h, w = target.shape[:2]
    rows, cols = h // tile_size, w // tile_size
    blocks = target.reshape(rows, tile_size, cols, tile_size, 3)
    return blocks.mean(axis=(1, 3), dtype=np.float32)          # (rows, cols, 3)


# ----------------------------------------------------------------------------
# Matching with maximum tile variety
# ----------------------------------------------------------------------------
def find_best_matches(block_colors: np.ndarray, tile_colors: np.ndarray) -> np.ndarray:
    """
    Closest-colour matching with a usage cap per tile.

    cap = ceil(blocks / tiles). A tile that reaches the cap is removed from the
    pool, so:
      * tiles >= blocks  -> cap is 1, no tile is ever repeated
      * tiles <  blocks  -> every tile is used almost equally (repeats only
                            because there is no other option)
    Blocks are visited in a fixed random order so no region of the picture
    gets all the best tiles (and so video frames stay stable).
    """
    flat = block_colors.reshape(-1, 3)
    n_blocks, n_tiles = len(flat), len(tile_colors)
    cap = -(-n_blocks // n_tiles)                    # ceil division

    tile_sq = (tile_colors ** 2).sum(axis=1)         # ||t||^2
    neg2_tiles = -2.0 * tile_colors                  # for the -2*b.t term
    penalty = np.zeros(n_tiles, dtype=np.float32)    # 0 = free, inf = used up
    usage = np.zeros(n_tiles, dtype=np.int32)
    best = np.empty(n_blocks, dtype=np.int64)

    for i in np.random.default_rng(0).permutation(n_blocks):
        # ||b-t||^2 without the constant ||b||^2 term (does not change argmin)
        score = tile_sq + neg2_tiles @ flat[i] + penalty
        t = int(score.argmin())
        best[i] = t
        usage[t] += 1
        if usage[t] >= cap:
            penalty[t] = np.inf

    return best.reshape(block_colors.shape[:2])


# ----------------------------------------------------------------------------
# Assemble
# ----------------------------------------------------------------------------
def build_mosaic(tiles: np.ndarray, indices: np.ndarray) -> np.ndarray:
    rows, cols = indices.shape
    t = tiles.shape[1]
    chosen = tiles[indices]                                    # (rows, cols, T, T, 3)
    return chosen.transpose(0, 2, 1, 3, 4).reshape(rows * t, cols * t, 3)


def make_mosaic(frame, tiles, tile_colors, tile_size, cols):
    """Full pipeline for a single frame/image."""
    target = prepare_target(frame, tile_size, cols)
    blocks = block_averages(target, tile_size)
    idx = find_best_matches(blocks, tile_colors)
    return build_mosaic(tiles, idx)


# ----------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------
def run_image(args, tiles, tile_colors):
    img = cv2.imread(args.target, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"Could not read target image '{args.target}'.")
    mosaic = make_mosaic(img, tiles, tile_colors, args.tile_size, args.cols)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), mosaic):
        sys.exit(f"Failed to write '{out}'.")
    print(f"Saved {mosaic.shape[1]}x{mosaic.shape[0]} mosaic to '{out}'")


def run_video(args, tiles, tile_colors):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Could not open video '{args.video}'.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer, n = None, 0
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            mosaic = make_mosaic(frame, tiles, tile_colors, args.tile_size, args.cols)
            if writer is None:
                h, w = mosaic.shape[:2]
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (w, h))
                if not writer.isOpened():
                    sys.exit(f"Could not open '{out}' for writing.")
            writer.write(mosaic)
            n += 1
            if n % 30 == 0:
                print(f"  frame {n}/{total if total > 0 else '?'}")
            if args.show:
                cv2.imshow("Mosaic (q to stop)", mosaic)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    print(f"Saved {n} frames to '{out}' (audio is not copied)")


def run_webcam(args, tiles, tile_colors):
    cap = cv2.VideoCapture(args.webcam)
    if not cap.isOpened():
        sys.exit(f"Could not open webcam {args.webcam}.")

    writer, last, fps_smooth = None, time.time(), 0.0
    print("Webcam running - press 'q' in the window to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)                         # mirror view
            mosaic = make_mosaic(frame, tiles, tile_colors, args.tile_size, args.cols)

            if args.output and writer is None:                 # record only if asked
                h, w = mosaic.shape[:2]
                writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                         20, (w, h))
            if writer:
                writer.write(mosaic)

            now = time.time()
            fps_smooth = 0.9 * fps_smooth + 0.1 / max(now - last, 1e-6)
            last = now
            cv2.putText(mosaic, f"{fps_smooth:.1f} fps", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Photo Mosaic (q to quit)", mosaic)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Photo mosaic for image / MP4 / webcam.")
    p.add_argument("--tiles-dir", default=TILES_DIR)
    p.add_argument("--target", default=TARGET_IMAGE, help="still image (default mode)")
    p.add_argument("--video", help="input video file (e.g. input.mp4)")
    p.add_argument("--webcam", nargs="?", const=0, type=int,
                   help="use webcam (optional camera index, default 0)")
    p.add_argument("--output", help="output file (default: mosaic.jpg / mosaic.mp4)")
    p.add_argument("--tile-size", type=int, help="tile edge in px")
    p.add_argument("--cols", type=int, help="tiles across the frame")
    p.add_argument("--show", action="store_true", help="preview window for video files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    is_video = args.video is not None
    is_webcam = args.webcam is not None
    live_or_video = is_video or is_webcam

    # Mode-dependent defaults
    if args.tile_size is None:
        args.tile_size = VIDEO_TILE_SIZE if live_or_video else IMAGE_TILE_SIZE
    if args.cols is None and live_or_video:
        args.cols = VIDEO_COLS
    if args.output is None and not is_webcam:
        args.output = "mosaic.mp4" if is_video else "mosaic.jpg"

    tiles_dir = Path(args.tiles_dir)
    if not tiles_dir.is_dir():
        sys.exit(f"Tiles directory '{tiles_dir}' does not exist.")

    print("Loading tile images...")
    tiles = load_tiles(tiles_dir, args.tile_size)
    tile_colors = average_colors(tiles)
    print(f"  {len(tiles)} tiles loaded ({args.tile_size}x{args.tile_size}px)")

    if is_webcam:
        run_webcam(args, tiles, tile_colors)
    elif is_video:
        run_video(args, tiles, tile_colors)
    else:
        run_image(args, tiles, tile_colors)


if __name__ == "__main__":
    main()
