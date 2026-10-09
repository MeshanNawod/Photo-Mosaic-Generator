# Photo Mosaic Generator

Rebuild a photo, an MP4 video, or your live webcam feed out of many small tile images.

Each block of the input is replaced by the tile whose average colour is closest to the block's colour. Tiles are spread out so each one is used as few times as possible.

## Features

- **Three inputs:** still image, MP4 video, or live webcam
- **Maximum tile variety:** if you have at least as many tiles as blocks, no tile repeats; otherwise tiles are reused as evenly as possible
- **Low memory:** tiles are renamed by their average colour (`RRR_GGG_BBB_index.png`), so matching reads only filenames and tile pixels are loaded only when used
- **Fast:** NumPy vectorised block averaging, no per-pixel loops

## Requirements

- Python 3.10+
- OpenCV and NumPy

```bash
pip install opencv-python numpy
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

The first run creates a `tiles_ready/` folder automatically. Each tile is square-cropped, resized, and saved with its average colour in the filename, for example `200_145_090_00012.png`.

## Usage

**Image**

```bash
python photo_mosaic.py --target photo.jpg --output mosaic.jpg
```

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
| `--output` | Output file | `mosaic.jpg` / `mosaic.mp4` |
| `--tiles-dir` | Folder of original tile images | `tiles` |
| `--cache-dir` | Folder for renamed tiles | `tiles_ready` |
| `--tile-size` | Tile edge in pixels | `50` (image), `16` (video/webcam) |
| `--cols` | Tiles across the frame | original size (image), `48` (video/webcam) |
| `--rebuild` | Re-create the renamed tile folder | off |
| `--show` | Preview window while rendering a video | off |

## How it works

1. **Prepare tiles:** crop to square, resize, save as `RRR_GGG_BBB_index.png` in `tiles_ready/`.
2. **Read colours:** average colours come from the filenames only.
3. **Split the input:** the frame is cropped to a multiple of the tile size and divided into a grid.
4. **Match:** each block gets the closest-colour tile. A tile that reaches its usage cap (`ceil(blocks / tiles)`) is removed from the pool.
5. **Assemble:** chosen tiles are loaded on demand, kept in memory, and placed into the final image.

## Tips

- **No repeats:** you need at least `cols × rows` tiles. For example, `--cols 30` on a 4:3 image needs about 30 × 23 = 690 tiles.
- **More tiles with a wide range of colours** give better results.
- **Run `--rebuild`** whenever you add or remove tiles, or change `--tile-size`. A cache made at 16px looks blurry if reused at 50px.
- **Smoother webcam:** lower `--cols` (for example `32`) for a higher frame rate.
- **Bigger output:** increase `--tile-size` and `--cols`, but expect slower processing.

## Limitations

- Output videos have no audio. You can add it back with ffmpeg:

  ```bash
  ffmpeg -i out.mp4 -i input.mp4 -map 0:v -map 1:a -c copy -shortest final.mp4
  ```

- The usage cap can give a few blocks a slightly worse colour match in exchange for more variety.

## License

Add your license here (for example MIT).
