#!/usr/bin/env python3
"""
Photo Mosaic Generator (OpenCV + NumPy)  -- image, MP4 video, or live webcam

Step 1 (automatic, runs once): every image in --tiles-dir is square-cropped,
resized, and saved into --cache-dir with the colours of its four quarters
(top-left, top-right, bottom-left, bottom-right) in the filename:

        TL_TR_BL_BR_index.png   e.g.  c8915a_c4905b_b27d4a_a96f3e_00012.png
        (each part is an RRGGBB hex colour)

Step 2: the filenames alone give the colours for matching. Tile pixels are
loaded lazily, only for tiles that are actually used.

Matching compares the 2x2 quarter colours of each block with those of each
tile, so edges and shapes stay sharp (not just the average colour). It is done
in LAB colour space (closer to how people see colour), and every tile is used
as evenly as possible. For video/webcam, a block keeps its
previous tile while that tile is still a close match, which removes flicker.

Examples:
    python photo_mosaic.py --target photo.jpg
    python photo_mosaic.py --video input.mp4 --output out.mp4
    python photo_mosaic.py --webcam
    python photo_mosaic.py --rebuild          # re-create the cache folder
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
TILES_DIR = "tiles"          # original images
CACHE_DIR = "tiles_ready"    # renamed/resized tiles are stored here
TARGET_IMAGE = "target.jpg"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

IMAGE_TILE_SIZE = 50
VIDEO_TILE_SIZE = 16
VIDEO_COLS = 48
VIDEO_STABILITY = 6.0        # LAB distance a kept tile may drift before changing


# ----------------------------------------------------------------------------
# Step 1: build the renamed tile folder
# ----------------------------------------------------------------------------
def prepare_cache(tiles_dir: Path, cache_dir: Path, tile_size: int) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in cache_dir.glob("*_*_*_*.png"):          # clear previous cache
        old.unlink()

    count = 0
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
        img = cv2.resize(img[y0:y0 + side, x0:x0 + side], (tile_size, tile_size),
                         interpolation=cv2.INTER_AREA)

        # Colour of each quarter (TL, TR, BL, BR) -> hex, stored in the filename.
        quads = img.reshape(2, tile_size // 2, 2, tile_size // 2, 3).mean(axis=(1, 3))
        name = "_".join(f"{int(round(r)):02x}{int(round(g)):02x}{int(round(b)):02x}"
                        for b, g, r in quads.reshape(4, 3))
        cv2.imwrite(str(cache_dir / f"{name}_{count:05d}.png"), img)
        count += 1

    if count == 0:
        sys.exit(f"No valid images found in '{tiles_dir}'.")
    print(f"  prepared {count} tiles in '{cache_dir}'")


# ----------------------------------------------------------------------------
# Step 2: read colours from filenames, load pixels only when needed
# ----------------------------------------------------------------------------
def list_cache(cache_dir: Path):
    """Return (paths, quarter colours BGR float32 (N,4,3)) using filenames only."""
    paths, quads = [], []
    for p in sorted(cache_dir.glob("*_*_*_*_*.png")):
        parts = p.stem.split("_")
        if len(parts) != 5:                               # not a tile from this version
            continue
        try:
            q = [(int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)) for h in parts[:4]]
        except ValueError:
            continue
        quads.append(q)                                   # stored as BGR
        paths.append(p)
    return paths, np.array(quads, dtype=np.float32).reshape(-1, 4, 3)


class TileStore:
    """Loads a tile from disk the first time it is requested, then keeps it."""

    def __init__(self, paths, tile_size):
        self.paths, self.size, self.cache = paths, tile_size, {}

    def get(self, i: int) -> np.ndarray:
        if i not in self.cache:
            img = cv2.imread(str(self.paths[i]), cv2.IMREAD_COLOR)
            if img.shape[0] != self.size:                # cache made at other size
                img = cv2.resize(img, (self.size, self.size),
                                 interpolation=cv2.INTER_AREA)
            self.cache[i] = img
        return self.cache[i]


# ----------------------------------------------------------------------------
# Target frame -> grid of block colours
# ----------------------------------------------------------------------------
def prepare_target(img: np.ndarray, tile_size: int, cols: int | None) -> np.ndarray:
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


def block_quadrants(target: np.ndarray, tile_size: int) -> np.ndarray:
    """Colour of each quarter (TL, TR, BL, BR) of every block -> (rows, cols, 4, 3)."""
    h, w = target.shape[:2]
    rows, cols, half = h // tile_size, w // tile_size, tile_size // 2
    blocks = target.reshape(rows, 2, half, cols, 2, half, 3)
    quads = blocks.mean(axis=(2, 5), dtype=np.float32)    # (rows, 2, cols, 2, 3)
    return quads.transpose(0, 2, 1, 3, 4).reshape(rows, cols, 4, 3)


# ----------------------------------------------------------------------------
# Matching: LAB colour, maximum tile variety, optional temporal stability
# ----------------------------------------------------------------------------
def to_lab(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR colours (0-255, any shape ending in 3) to LAB."""
    arr = np.asarray(bgr, dtype=np.float32).reshape(-1, 1, 3) / 255.0
    return cv2.cvtColor(arr, cv2.COLOR_BGR2LAB).reshape(np.shape(bgr))


def to_features(quads_bgr: np.ndarray) -> np.ndarray:
    """(..., 4, 3) BGR quarter colours -> (N, 12) LAB features.

    Divided by 2 so a distance equals the RMS colour difference per quarter,
    which keeps --stability on the same scale as plain colour distance.
    """
    return to_lab(quads_bgr).reshape(-1, 12) / 2.0


class Matcher:
    """
    Picks a tile for every block.

    * The 2x2 quarter colours of each block are compared with each tile's, in LAB space.
    * Each tile is used at most ceil(blocks / tiles) times.
    * If keep_thresh > 0, a block keeps the tile it had in the previous frame
      as long as the block's colour has moved less than keep_thresh (LAB
      distance) since that tile was chosen. Only the other blocks are matched
      again, so small changes and camera noise no longer make tiles flicker.
    """

    def __init__(self, tile_quads_bgr: np.ndarray, keep_thresh: float = 0.0):
        self.tile_feat = to_features(tile_quads_bgr)
        self.tile_sq = (self.tile_feat ** 2).sum(axis=1)
        self.neg2_tiles = -2.0 * self.tile_feat
        self.keep_thresh = keep_thresh
        self.prev = None                                  # previous frame's picks
        self.anchor = None                                # block colour when tile was chosen

    def match(self, block_quads_bgr: np.ndarray) -> np.ndarray:
        grid_shape = block_quads_bgr.shape[:2]
        flat = to_features(block_quads_bgr)
        n_blocks, n_tiles = len(flat), len(self.tile_feat)
        cap = -(-n_blocks // n_tiles)                     # ceil division

        best = np.full(n_blocks, -1, dtype=np.int64)
        usage = np.zeros(n_tiles, dtype=np.int32)
        keep = np.zeros(n_blocks, dtype=bool)

        # Keep last frame's tile where it is still a close match.
        if self.keep_thresh > 0 and self.prev is not None and self.prev.size == n_blocks:
            keep = np.linalg.norm(flat - self.anchor, axis=1) <= self.keep_thresh
            best[keep] = self.prev[keep]
            usage += np.bincount(best[keep], minlength=n_tiles).astype(np.int32)

        penalty = np.where(usage >= cap, np.inf, 0.0).astype(np.float32)

        # Fixed random order so no region of the picture gets all the best tiles.
        order = np.random.default_rng(0).permutation(n_blocks)
        for i in order[best[order] < 0]:
            t = int((self.tile_sq + self.neg2_tiles @ flat[i] + penalty).argmin())
            best[i] = t
            usage[t] += 1
            if usage[t] >= cap:
                penalty[t] = np.inf

        # Kept blocks retain their old anchor colour; re-matched ones get a new one.
        self.anchor = np.where(keep[:, None], self.anchor, flat) if keep.any() else flat
        self.prev = best
        return best.reshape(grid_shape)


# ----------------------------------------------------------------------------
# Assemble
# ----------------------------------------------------------------------------
def build_mosaic(store: TileStore, indices: np.ndarray) -> np.ndarray:
    rows, cols = indices.shape
    t = store.size
    out = np.empty((rows * t, cols * t, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            out[r * t:(r + 1) * t, c * t:(c + 1) * t] = store.get(int(indices[r, c]))
    return out


def make_mosaic(frame, store, matcher, tile_size, cols):
    target = prepare_target(frame, tile_size, cols)
    quads = block_quadrants(target, tile_size)
    return build_mosaic(store, matcher.match(quads))


# ----------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------
def run_image(args, store, matcher):
    img = cv2.imread(args.target, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"Could not read target image '{args.target}'.")
    mosaic = make_mosaic(img, store, matcher, args.tile_size, args.cols)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), mosaic):
        sys.exit(f"Failed to write '{out}'.")
    print(f"Saved {mosaic.shape[1]}x{mosaic.shape[0]} mosaic to '{out}'")


def run_video(args, store, matcher):
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
            mosaic = make_mosaic(frame, store, matcher, args.tile_size, args.cols)
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


def run_webcam(args, store, matcher):
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
            frame = cv2.flip(frame, 1)
            mosaic = make_mosaic(frame, store, matcher, args.tile_size, args.cols)

            if args.output and writer is None:
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
    p.add_argument("--tiles-dir", default=TILES_DIR, help="original tile images")
    p.add_argument("--cache-dir", default=CACHE_DIR, help="renamed tiles folder")
    p.add_argument("--rebuild", action="store_true",
                   help="re-create the cache folder (use after changing tiles or --tile-size)")
    p.add_argument("--target", default=TARGET_IMAGE, help="still image (default mode)")
    p.add_argument("--video", help="input video file (e.g. input.mp4)")
    p.add_argument("--webcam", nargs="?", const=0, type=int,
                   help="use webcam (optional camera index, default 0)")
    p.add_argument("--output", help="output file (default: mosaic.jpg / mosaic.mp4)")
    p.add_argument("--tile-size", type=int, help="tile edge in px")
    p.add_argument("--cols", type=int, help="tiles across the frame")
    p.add_argument("--stability", type=float,
                   help="video/webcam: keep a block's tile while it stays within this "
                        f"LAB distance (default {VIDEO_STABILITY}; 0 = off, higher = steadier)")
    p.add_argument("--show", action="store_true", help="preview window for video files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    is_video = args.video is not None
    is_webcam = args.webcam is not None
    live_or_video = is_video or is_webcam

    if args.tile_size is None:
        args.tile_size = VIDEO_TILE_SIZE if live_or_video else IMAGE_TILE_SIZE
    if args.cols is None and live_or_video:
        args.cols = VIDEO_COLS
    if args.stability is None:
        args.stability = VIDEO_STABILITY if live_or_video else 0.0
    if args.output is None and not is_webcam:
        args.output = "mosaic.mp4" if is_video else "mosaic.jpg"

    if args.tile_size % 2:
        sys.exit("--tile-size must be even (tiles are matched by their four quarters).")

    tiles_dir, cache_dir = Path(args.tiles_dir), Path(args.cache_dir)

    paths, tile_quads = list_cache(cache_dir) if cache_dir.is_dir() else ([], None)
    if args.rebuild or len(paths) == 0:
        if not tiles_dir.is_dir():
            sys.exit(f"Tiles directory '{tiles_dir}' does not exist.")
        print("Preparing tile folder...")
        prepare_cache(tiles_dir, cache_dir, args.tile_size)
        paths, tile_quads = list_cache(cache_dir)

    print(f"{len(paths)} tiles ready ({args.tile_size}x{args.tile_size}px)")
    store = TileStore(paths, args.tile_size)
    matcher = Matcher(tile_quads, args.stability)

    if is_webcam:
        run_webcam(args, store, matcher)
    elif is_video:
        run_video(args, store, matcher)
    else:
        run_image(args, store, matcher)


if __name__ == "__main__":
    main()
