# Photo Mosaic Generator

A small Python script that rebuilds a target image out of a collection of small tile images, using OpenCV and NumPy.

## How it works

1. **Load tiles** – scans a folder, keeps valid image files (JPG, PNG, BMP, WebP, TIFF), centre-crops each to a square and resizes it to `TILE_SIZE × TILE_SIZE`. Unreadable files are skipped.
2. **Tile colours** – computes each tile's average BGR colour in one vectorised NumPy call.
3. **Target image** – optionally rescales it, then centre-crops it so width and height are exact multiples of the tile size.
4. **Matching** – splits the target into a grid, averages each block, and finds the tile with the smallest Euclidean colour distance. This is done as a single matrix multiplication per chunk (no per-block Python loops), so it stays fast even with thousands of tiles.
5. **Assembly** – swaps every block for its matched tile using NumPy indexing and reshaping.
6. **Save** – writes the result to your output path.

## Requirements

- Python 3.10+
- OpenCV and NumPy

```bash
pip install opencv-python numpy
```

## Setup

Project layout:

```
photo_mosaic.py
tiles/          <- put your small source images here
target.jpg      <- the image you want to turn into a mosaic
```

### Option A: edit the paths in the script

Open `photo_mosaic.py` and change the CONFIG section at the top:

```python
TILES_DIR = "tiles"          # folder of source images
TARGET_IMAGE = "target.jpg"  # main image
OUTPUT_IMAGE = "mosaic.jpg"  # where to save the result
TILE_SIZE = 50               # tile edge length in pixels
TARGET_COLS = None           # optional: tiles across (None = keep original size)
```

Then run:

```bash
python photo_mosaic.py
```

### Option B: command-line flags

```bash
python photo_mosaic.py --tiles-dir path/to/tiles --target path/to/photo.jpg --output path/to/result.jpg
```

| Flag | Meaning | Default |
|------|---------|---------|
| `--tiles-dir` | Folder of source images | `tiles` |
| `--target` | Target image path | `target.jpg` |
| `--output` | Output file path (extension sets format) | `mosaic.jpg` |
| `--tile-size` | Tile edge length in pixels | `50` |
| `--cols` | Resize target to this many tiles across | off |

## Tips

- **Output size** = (tiles across × tile size) by (tiles down × tile size). A 4000 px wide photo with 50 px tiles gives 80 columns and a 4000 px mosaic. Use `--cols` (e.g. `--cols 60`) or a smaller `--tile-size` to control it.
- **More tiles = better results.** A few hundred varied images is a reasonable minimum; a few thousand looks noticeably better.
- **Colour variety matters.** Tiles should cover a wide range of colours and brightness levels.
- **Repeats are allowed.** The same tile can appear many times, and with a small library you may see repeating patterns.
- **Memory.** Matching runs in chunks of 4096 blocks (`CHUNK_SIZE`); lower it if you run short of RAM.

## Troubleshooting

- `No valid images found` – check the tiles folder path and file extensions.
- `Failed to write` – use a supported output extension such as `.jpg` or `.png`.
- Mosaic looks washed out – add more tiles with a wider colour range.
