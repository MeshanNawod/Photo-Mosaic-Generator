#!/usr/bin/env python3
"""
Photo Mosaic Generator (OpenCV + NumPy)  -- image, MP4 video, or live webcam

TILE FOLDER (built automatically, once)
    Every image in --tiles-dir is square-cropped, resized, and saved into
    --cache-dir with the colours of its four quarters in the filename:

        TL_TR_BL_BR_index.png   e.g.  c8915a_c4905b_b27d4a_a96f3e_00012.png
        (each part is an RRGGBB hex colour)

    Matching reads only the filenames; tile pixels are loaded lazily.

MATCHING
    * Each block is compared with each tile by its 2x2 quarter colours in LAB
      space, so edges and shapes stay sharp.
    * Still images: if there are enough tiles, the Hungarian algorithm finds the
      best possible one-tile-per-block assignment (no tile is repeated).
    * If tiles must repeat, they are used as evenly as possible and identical
      tiles are kept away from each other.
    * Video/webcam: fast greedy matching; a block keeps its tile until its
      colour really changes, so there is no flicker.
    * --recolor nudges each tile slightly toward its block's colour.
    * --adaptive (stills): flat areas such as sky use big tiles, detailed
      areas use small ones.

HIGH RESOLUTION
    --resolution 4k (or hd, 2k, 8k, or a width in pixels) sizes the grid so the
    output has that width. Tiles are cached at twice the tile size, so they stay
    sharp, and big adaptive tiles are never blown up from small ones.
    Use a .png output for the best quality.

Examples:
    python photo_mosaic.py --target photo.jpg
    python photo_mosaic.py --target photo.jpg --adaptive
    python photo_mosaic.py --target photo.jpg --resolution 4k --adaptive
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

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:                                      # optional dependency
    linear_sum_assignment = None

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
DEFAULT_RECOLOR = 0.15       # 0 = tiles untouched, 1 = tile forced to block colour
DEFAULT_ADAPTIVE = 6.0       # how flat a 2x2 group of blocks must be to merge
SPACING_RADIUS = 2           # identical tiles are kept this many cells apart
SPACING_PENALTY = 100.0      # score penalty for reusing a tile near itself
OPTIMAL_LIMIT = 60_000_000   # max blocks x tiles for the Hungarian algorithm
RESOLUTIONS = {"hd": 1920, "2k": 2560, "4k": 3840, "8k": 7680}   # output widths (px)
RESOLUTION_COLS = 96         # tiles across when only --resolution is given


# ----------------------------------------------------------------------------
# Step 1: build the renamed tile folder
# ----------------------------------------------------------------------------
def prepare_cache(tiles_dir: Path, cache_dir: Path, tile_size: int) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in cache_dir.glob("*_*_*_*.png"):          # clear previous cache
        old.unlink()

    count = small = 0
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

        # Keep tiles at up to 2x the tile size (even number) so they stay sharp,
        # including when they are used as big adaptive tiles.
        stored = min(side, 2 * tile_size) // 2 * 2
        if stored < 2:
            continue
        if side < tile_size:
            small += 1
        if side != stored:
            img = cv2.resize(img, (stored, stored), interpolation=cv2.INTER_AREA)

        # Colour of each quarter (TL, TR, BL, BR) -> hex, stored in the filename.
        quads = img.reshape(2, stored // 2, 2, stored // 2, 3).mean(axis=(1, 3))
        name = "_".join(f"{int(round(r)):02x}{int(round(g)):02x}{int(round(b)):02x}"
                        for b, g, r in quads.reshape(4, 3))
        cv2.imwrite(str(cache_dir / f"{name}_{count:05d}.png"), img)
        count += 1

    if count == 0:
        sys.exit(f"No valid images found in '{tiles_dir}'.")
    print(f"  prepared {count} tiles in '{cache_dir}'")
    if small:
        print(f"  note: {small} source images are smaller than the tile size and will look soft")


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

    def __init__(self, paths, tile_size, means):
        self.paths, self.size, self.means, self.cache = paths, tile_size, means, {}

    def get(self, i: int, size: int | None = None) -> np.ndarray:
        """Tile i at `size` px (default: the tile size), always resampled from the stored copy."""
        size = size or self.size
        key = (i, size)
        if key not in self.cache:
            img = cv2.imread(str(self.paths[i]), cv2.IMREAD_COLOR)
            if img.shape[0] != size:
                interp = cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_CUBIC
                img = cv2.resize(img, (size, size), interpolation=interp)
            self.cache[key] = img
        return self.cache[key]


# ----------------------------------------------------------------------------
# Target frame -> grid of block colours
# ----------------------------------------------------------------------------
def prepare_target(img: np.ndarray, tile_size: int, cols: int | None) -> np.ndarray:
    if cols:
        new_w = cols * tile_size
        new_h = max(tile_size, round(img.shape[0] * new_w / img.shape[1]))
        interp = cv2.INTER_AREA if new_w < img.shape[1] else cv2.INTER_LANCZOS4
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)

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
# Matching
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
    """Chooses a tile for every block (or item) of the target."""

    def __init__(self, tile_quads_bgr: np.ndarray, keep_thresh: float = 0.0):
        self.tile_feat = to_features(tile_quads_bgr)
        self.tile_sq = (self.tile_feat ** 2).sum(axis=1)
        self.neg2_tiles = -2.0 * self.tile_feat
        self.keep_thresh = keep_thresh
        self.prev = None                                  # previous frame's picks
        self.anchor = None                                # block colour when tile was chosen

    # -- grid of blocks (images and video frames) ---------------------------
    def match(self, block_quads_bgr: np.ndarray, optimal: bool = False) -> np.ndarray:
        grid = block_quads_bgr.shape[:2]
        flat = to_features(block_quads_bgr)
        n_items, n_tiles = len(flat), len(self.tile_feat)
        cap = -(-n_items // n_tiles)

        best = np.full(n_items, -1, dtype=np.int64)
        usage = np.zeros(n_tiles, dtype=np.int32)
        keep = np.zeros(n_items, dtype=bool)

        # Video: keep last frame's tile while the block colour has barely moved.
        if self.keep_thresh > 0 and self.prev is not None and self.prev.size == n_items:
            keep = np.linalg.norm(flat - self.anchor, axis=1) <= self.keep_thresh
            best[keep] = self.prev[keep]
            usage += np.bincount(best[keep], minlength=n_tiles).astype(np.int32)

        self._solve(flat, grid, best, usage, cap, optimal)

        # Kept blocks retain their old anchor colour; re-matched ones get a new one.
        self.anchor = np.where(keep[:, None], self.anchor, flat) if keep.any() else flat
        self.prev = best
        return best.reshape(grid)

    # -- arbitrary list of items (used by --adaptive) ------------------------
    def match_items(self, feats: np.ndarray, optimal: bool = True) -> np.ndarray:
        n_items, n_tiles = len(feats), len(self.tile_feat)
        best = np.full(n_items, -1, dtype=np.int64)
        usage = np.zeros(n_tiles, dtype=np.int32)
        self._solve(feats, None, best, usage, -(-n_items // n_tiles), optimal)
        return best

    # -- the actual assignment ------------------------------------------------
    def _solve(self, flat, grid, best, usage, cap, optimal):
        n_items, n_tiles = len(flat), len(self.tile_feat)

        # Best possible overall assignment when no tile has to repeat.
        if (optimal and cap == 1 and linear_sum_assignment is not None
                and not (best >= 0).any() and n_items * n_tiles <= OPTIMAL_LIMIT):
            # ||b||^2 is constant per row, so dropping it keeps the optimum.
            cost = flat @ self.neg2_tiles.T + self.tile_sq
            rows, cols = linear_sum_assignment(cost)
            best[rows] = cols
            return

        # Greedy: closest tile that is not used up; repeats are kept apart.
        penalty = np.where(usage >= cap, np.inf, 0.0).astype(np.float32)
        near = best.reshape(grid) if (grid is not None and cap > 1) else None
        order = np.random.default_rng(0).permutation(n_items)   # fixed order

        for i in order[best[order] < 0]:
            score = self.tile_sq + self.neg2_tiles @ flat[i] + penalty
            if near is not None:
                r, c = divmod(int(i), grid[1])
                around = near[max(r - SPACING_RADIUS, 0):r + SPACING_RADIUS + 1,
                              max(c - SPACING_RADIUS, 0):c + SPACING_RADIUS + 1]
                around = around[around >= 0]
                if around.size:
                    score[around] += SPACING_PENALTY
            t = int(score.argmin())
            best[i] = t
            usage[t] += 1
            if usage[t] >= cap:
                penalty[t] = np.inf


# ----------------------------------------------------------------------------
# Assemble
# ----------------------------------------------------------------------------
def build_mosaic(store: TileStore, indices: np.ndarray, block_means: np.ndarray,
                 recolor: float) -> np.ndarray:
    """Regular grid. Each tile is shifted `recolor` of the way to its block colour."""
    rows, cols = indices.shape
    t = store.size
    out = np.empty((rows * t, cols * t, 3), dtype=np.uint8)
    for r in range(rows):
        idx = indices[r]
        row = np.stack([store.get(int(i)) for i in idx])              # (cols, T, T, 3)
        if recolor > 0:
            shift = recolor * (block_means[r] - store.means[idx])     # (cols, 3)
            row = np.clip(row.astype(np.float32) + shift[:, None, None, :], 0, 255)
        out[r * t:(r + 1) * t] = row.astype(np.uint8).transpose(1, 0, 2, 3).reshape(t, cols * t, 3)
    return out


def adaptive_items(quads: np.ndarray, thresh: float):
    """
    Merge flat 2x2 groups of blocks into single big items (quadtree, one level).
    Returns positions (n,2), sizes (n,) in blocks (1 or 2), quarter colours (n,4,3).
    """
    rows, cols = quads.shape[:2]
    r2, c2 = rows // 2, cols // 2
    means = quads.mean(axis=2)                                        # (rows, cols, 3)
    lab_q, lab_m = to_lab(quads), to_lab(means)
    inner = np.linalg.norm(lab_q - lab_q.mean(axis=2, keepdims=True), axis=-1).max(axis=2)

    flat = np.zeros((r2, c2), dtype=bool)
    if r2 and c2:
        g = lab_m[:2 * r2, :2 * c2].reshape(r2, 2, c2, 2, 3)
        between = np.linalg.norm(g - g.mean(axis=(1, 3), keepdims=True), axis=-1).max(axis=(1, 3))
        inside = inner[:2 * r2, :2 * c2].reshape(r2, 2, c2, 2).max(axis=(1, 3))
        flat = (between <= thresh) & (inside <= thresh)

    covered = np.zeros((rows, cols), dtype=bool)
    pos, size, iq = [], [], []
    for gr, gc in zip(*np.nonzero(flat)):
        r, c = 2 * gr, 2 * gc
        covered[r:r + 2, c:c + 2] = True
        pos.append((r, c)); size.append(2)
        iq.append(means[r:r + 2, c:c + 2].reshape(4, 3))              # TL, TR, BL, BR
    for r, c in zip(*np.nonzero(~covered)):
        pos.append((r, c)); size.append(1); iq.append(quads[r, c])
    return np.array(pos), np.array(size), np.array(iq, dtype=np.float32)


def mosaic_adaptive(target, store, matcher, thresh, recolor):
    t = store.size
    quads = block_quadrants(target, t)
    rows, cols = quads.shape[:2]
    pos, size, iq = adaptive_items(quads, thresh)
    idx = matcher.match_items(to_features(iq), optimal=True)

    out = np.empty((rows * t, cols * t, 3), dtype=np.uint8)
    for (r, c), s, q, i in zip(pos, size, iq, idx):
        tile = store.get(int(i), s * t)                   # big tiles come from the 2x copy
        if recolor > 0:
            shift = recolor * (q.mean(axis=0) - store.means[i])
            tile = np.clip(tile.astype(np.float32) + shift, 0, 255).astype(np.uint8)
        out[r * t:(r + s) * t, c * t:(c + s) * t] = tile
    big = int((size == 2).sum())
    print(f"  adaptive: {big} big + {len(size) - big} small tiles "
          f"(instead of {rows * cols})")
    return out


def make_mosaic(frame, store, matcher, args, still=False):
    target = prepare_target(frame, args.tile_size, args.cols)
    if still and args.adaptive:
        return mosaic_adaptive(target, store, matcher, args.adaptive, args.recolor)
    quads = block_quadrants(target, args.tile_size)
    n_blocks = quads.shape[0] * quads.shape[1]
    if still:
        if n_blocks > len(store.paths):
            print(f"  note: {n_blocks} blocks but only {len(store.paths)} tiles - tiles will "
                  f"repeat. Add more tiles or use fewer --cols for no repeats.")
        elif n_blocks * len(store.paths) > 4_000_000 and linear_sum_assignment is not None:
            print("  finding the best tile for every block (large image, this can take a minute)...")
    idx = matcher.match(quads, optimal=still)
    return build_mosaic(store, idx, quads.mean(axis=2), args.recolor)


# ----------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------
def run_image(args, store, matcher):
    img = cv2.imread(args.target, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"Could not read target image '{args.target}'.")
    mosaic = make_mosaic(img, store, matcher, args, still=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ext = out.suffix.lower()
    params = ([cv2.IMWRITE_JPEG_QUALITY, 98] if ext in (".jpg", ".jpeg")
              else [cv2.IMWRITE_PNG_COMPRESSION, 3] if ext == ".png" else [])
    if not cv2.imwrite(str(out), mosaic, params):
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
            mosaic = make_mosaic(frame, store, matcher, args)
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
            mosaic = make_mosaic(frame, store, matcher, args)

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
def parse_resolution(text: str) -> int:
    key = text.lower()
    if key in RESOLUTIONS:
        return RESOLUTIONS[key]
    try:
        width = int(key)
    except ValueError:
        width = 0
    if width < 64:
        raise argparse.ArgumentTypeError("use hd, 2k, 4k, 8k or a width in pixels")
    return width


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
    p.add_argument("--tile-size", type=int, help="tile edge in px (must be even)")
    p.add_argument("--cols", type=int, help="tiles across the frame")
    p.add_argument("--resolution", type=parse_resolution, metavar="hd|2k|4k|8k|WIDTH",
                   help="output width, e.g. 4k = 3840 px. Sets --cols and --tile-size "
                        f"(default {RESOLUTION_COLS} tiles across) unless you give them")
    p.add_argument("--stability", type=float,
                   help="video/webcam: keep a block's tile while it stays within this "
                        f"LAB distance (default {VIDEO_STABILITY}; 0 = off, higher = steadier)")
    p.add_argument("--recolor", type=float, default=DEFAULT_RECOLOR,
                   help="shift tiles toward their block colour, 0-1 "
                        f"(default {DEFAULT_RECOLOR}; 0 = leave tiles untouched)")
    p.add_argument("--adaptive", nargs="?", const=DEFAULT_ADAPTIVE, type=float,
                   help="still images: big tiles in flat areas, small tiles in detail "
                        f"(optional flatness threshold, default {DEFAULT_ADAPTIVE}; higher = more big tiles)")
    p.add_argument("--show", action="store_true", help="preview window for video files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    is_video = args.video is not None
    is_webcam = args.webcam is not None
    live_or_video = is_video or is_webcam

    if args.resolution:                                   # size the grid for this width
        if args.tile_size is None and args.cols is None:
            args.cols = RESOLUTION_COLS
        if args.tile_size is None:
            args.tile_size = max(2, args.resolution // args.cols // 2 * 2)
        elif args.cols is None:
            args.cols = max(1, args.resolution // args.tile_size)
        print(f"Output width: {args.cols * args.tile_size}px "
              f"({args.cols} tiles of {args.tile_size}px across)")

    if args.tile_size is None:
        args.tile_size = VIDEO_TILE_SIZE if live_or_video else IMAGE_TILE_SIZE
    if args.cols is None and live_or_video:
        args.cols = VIDEO_COLS
    if args.stability is None:
        args.stability = VIDEO_STABILITY if live_or_video else 0.0
    if args.output is None and not is_webcam:
        args.output = ("mosaic.mp4" if is_video
                       else "mosaic.png" if args.resolution else "mosaic.jpg")

    if args.tile_size % 2:
        sys.exit("--tile-size must be even (tiles are matched by their four quarters).")
    if not 0 <= args.recolor <= 1:
        sys.exit("--recolor must be between 0 and 1.")
    if args.adaptive and live_or_video:
        print("Note: --adaptive only applies to still images; ignoring it.")
        args.adaptive = None
    if linear_sum_assignment is None and not live_or_video:
        print("Note: scipy not installed (pip install scipy) - using greedy matching "
              "instead of the optimal assignment.")

    # One cache folder per tile size, so switching sizes never needs a rebuild.
    tiles_dir = Path(args.tiles_dir)
    cache_dir = Path(args.cache_dir) / str(args.tile_size)

    paths, tile_quads = list_cache(cache_dir) if cache_dir.is_dir() else ([], None)
    if args.rebuild or len(paths) == 0:
        if not tiles_dir.is_dir():
            sys.exit(f"Tiles directory '{tiles_dir}' does not exist.")
        print("Preparing tile folder...")
        prepare_cache(tiles_dir, cache_dir, args.tile_size)
        paths, tile_quads = list_cache(cache_dir)

    print(f"{len(paths)} tiles ready ({args.tile_size}x{args.tile_size}px)")
    store = TileStore(paths, args.tile_size, tile_quads.mean(axis=1))
    matcher = Matcher(tile_quads, args.stability)

    if is_webcam:
        run_webcam(args, store, matcher)
    elif is_video:
        run_video(args, store, matcher)
    else:
        run_image(args, store, matcher)


if __name__ == "__main__":
    main()
