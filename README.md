# Photo Mosaic Generator

Rebuild a photo, an MP4 video, or your live webcam feed out of many small tile images.

Each block of the input is replaced by the tile that matches it best. Tiles are spread out so each one is used as few times as possible, and identical tiles are kept apart.

## Features

- **Three inputs:** still image, MP4 video, or live webcam
- **4K and beyond:** `--resolution 4k` (or `hd`, `2k`, `8k`, or any width) sizes the grid for that output width. Tiles are stored at twice the tile size, so they stay sharp, and big adaptive tiles are never blown up from small ones
- **Optimal matching (images):** the Hungarian algorithm finds the best possible one-tile-per-block assignment, so no block is left with a leftover tile
- **Detail-aware:** blocks and tiles are compared by the colours of their four quarters (2×2) in LAB colour space, so edges and shapes stay sharp
- **Maximum tile variety:** with enough tiles, none repeats. Otherwise tiles are reused as evenly as possible, and identical tiles are kept away from each other
- **Adaptive tile sizes (images):** big tiles in flat areas like sky, small tiles where there is detail
- **Light recolouring:** tiles are nudged toward their block's colour, so a small tile library still gives a faithful picture
- **Steady video:** in video and webcam mode a block keeps its tile until its colour really changes, so the mosaic doesn't flicker
- **Low memory:** tiles are renamed by their quarter colours, so matching reads only filenames and tile pixels are loaded only when used

## Requirements

- Python 3.10+
- OpenCV and NumPy
- SciPy (for the optimal matching; without it the script falls back to greedy matching)

```bash
pip install opencv-python numpy scipy
```

## Setup

Put your tile images (any size, any mix of jpg/png/bmp/webp/tif) in a `tiles/` folder:

```
photo_mosaic.py
tiles/
    img001.jpg
    img002.png
    ...
```

The first run creates a `tiles_ready/` folder automatically, with one sub-folder per tile size (for example `tiles_ready/40/`), so switching between sizes never needs a rebuild. Each tile is square-cropped, resized to up to twice the tile size, and saved with the colours of its four quarters (top-left, top-right, bottom-left, bottom-right) in the filename, as hex codes, for example `c8915a_c4905b_b27d4a_a96f3e_00012.png`.

## Usage

**Image**

```bash
python photo_mosaic.py --target photo.jpg --output mosaic.jpg
python photo_mosaic.py --target photo.jpg --adaptive      # big tiles in flat areas
```

**4K image**

```bash
python photo_mosaic.py --target photo.jpg --resolution 4k --adaptive
```

This writes `mosaic.png` at 3840 px wide by default (96 tiles of 40 px across). Use `--cols` or `--tile-size` to change the grid, for example `--resolution 4k --cols 128`. PNG is lossless, so it is the best format for 4K.

**Video (MP4)**

```bash
python photo_mosaic.py --video input.mp4 --output out.mp4
```

Add `--show` to preview while rendering (press `q` to stop).

**Webcam**

```bash
python photo_mosaic.py --webcam
```

Press `q` in the window to quit. Add `--output webcam.mp4` to record. Use `--webcam 1` to pick another camera.

## Options

| Option | Description | Default |
|---|---|---|
| `--target` | Input still image | `target.jpg` |
| `--video` | Input video file | - |
| `--webcam [index]` | Use webcam | camera `0` |
| `--output` | Output file | `mosaic.jpg` / `mosaic.mp4` (`mosaic.png` with `--resolution`) |
| `--tiles-dir` | Folder of original tile images | `tiles` |
| `--cache-dir` | Folder for renamed tiles | `tiles_ready` |
| `--tile-size` | Tile edge in pixels (must be even) | `50` (image), `16` (video/webcam) |
| `--cols` | Tiles across the frame | original size (image), `48` (video/webcam), `96` with `--resolution` |
| `--resolution` | Output width: `hd` (1920), `2k` (2560), `4k` (3840), `8k` (7680) or a width in pixels. Sets `--cols` and `--tile-size` unless you give them | off |
| `--recolor` | Shift tiles toward their block colour, `0` to `1`. `0` leaves tiles untouched | `0.15` |
| `--adaptive [thr]` | Images only: merge flat 2×2 groups of blocks into big tiles. Higher threshold = more big tiles | off (`6` if given with no value) |
| `--stability` | Video/webcam: keep a block's tile until its colour moves this far (LAB units). `0` turns it off, higher is steadier | `6` (video/webcam) |
| `--rebuild` | Re-create the renamed tile folder for the current tile size (use after adding or removing tiles) | off |
| `--show` | Preview window while rendering a video | off |

## How it works

1. **Prepare tiles:** crop to square, resize, save as `TL_TR_BL_BR_index.png` in `tiles_ready/`.
2. **Read colours:** the four quarter colours of every tile come from the filenames only.
3. **Split the input:** the frame is cropped to a multiple of the tile size and divided into a grid. With `--adaptive`, flat 2×2 groups of blocks are merged into one big block.
4. **Match:**
   - Images with enough tiles: the Hungarian algorithm picks the best overall one-tile-per-block assignment.
   - Otherwise (and for video): greedy matching where a tile that reaches its usage cap (`ceil(blocks / tiles)`) is removed from the pool, and identical tiles are penalised near each other. In video, blocks whose colour barely changed keep last frame's tile.
5. **Assemble:** chosen tiles are loaded on demand, recoloured slightly, and placed into the final image.

## Tips

- **No repeats:** you need at least as many tiles as blocks (`cols × rows`). For example, `--cols 30` on a 4:3 image needs about 30 × 23 = 690 tiles. `--adaptive` needs fewer because it uses fewer, bigger blocks.
- **More tiles with a wide range of colours** give better results.
- **Recolouring:** `--recolor 0.15` is subtle. Use `0` for pure, untouched photos as tiles, or `0.3` and up for a more faithful picture with few tiles.
- **Run `--rebuild`** whenever you add or remove tiles. Changing `--tile-size` is handled automatically (each size has its own cache folder). Old `tiles_ready/*.png` files from earlier versions are no longer used and can be deleted.
- **Sharp 4K:** use source photos at least as large as the tile size (ideally twice as large). Smaller ones are enlarged and look soft; the script tells you if that happens.
- **4K needs many tiles:** the default 4K grid is 96 × 54 = 5,184 blocks, so you want at least that many tiles for no repeats. With fewer, the script tells you and reuses tiles. `--adaptive` needs fewer.
- **Time and memory at 4K:** the optimal matching for about 5,000 blocks and 6,000 tiles took roughly 30 seconds in total and about 500 MB of memory in testing. It is used while `blocks × tiles` is at most 60 million; above that the script switches to the faster greedy matching automatically.
- **Flicker or sluggish changes?** Lower `--stability` if the mosaic reacts too slowly to movement; raise it if tiles still shimmer.
- **Smoother webcam:** lower `--cols` (for example `32`) for a higher frame rate.
- **Bigger output:** increase `--tile-size` and `--cols`, but expect slower processing.

## Limitations

- Output videos have no audio. You can add it back with ffmpeg:

  ```bash
  ffmpeg -i out.mp4 -i input.mp4 -map 0:v -map 1:a -c copy -shortest final.mp4
  ```

- `--adaptive` and the optimal matching apply to still images only. Video and webcam use the faster greedy matching.
- Big adaptive tiles are enlarged copies of normal tiles, so they look slightly softer.

## License

Add your license here (for example MIT).
