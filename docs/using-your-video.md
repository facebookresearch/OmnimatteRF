Using Your Video
---

This is a step-by-step guide to prepare a video for processing with our method.

Suppose you have a video file, `myvideo.mp4`, and the results will be stored in `/data/matting/wild/myvideo`. We will refer to this folder as `$ROOT`.

## 0. Use the CLI

The recommeded way to add your video is to [use the CLI](using-the-cli.md). After the initial setup, run

```
python ./ui/cli.py \
    add_data \
    wild/myvideo \
    --video /path/to/myvideo.mp4 \
    --v2i_step 2 \
    --v2i_limit 200 \
    --v2i_skip 1s
```

See [Video to Images](#1-video-to-images) for explanation of the `v2i` options. If you have image files already, it also works:

```
python ./ui/cli.py \
    add_data \
    wild/myvideo \
    --video /path/to/myvideo-image-folder
```

The command puts the images in `$ROOT/rgb_1x`. It then runs optical flow, Detectron segmentation, and homography estimation. These additional processings can be disabled with `--no_run_extra`.

You can add masks after obtaining them:

```
python ./ui/cli.py \
    add_data \
    wild/myvideo \
    --masks /path/to/first-obj,/path/to/second-obj  # comma-separate masks of multiple objects
```

Once the video has been imported to the data folder, you can proceed to prepare additional data (masks, camera poses, depths).

### Coarse Masks

There are many off-the-shelf tools that extract masks from videos. You can use your favorite and import the masks.

We also provide a simple GUI tool for converting Detectron 2 segmentation instance data to mask sequences. See the [instructions below](#5-coarse-masks).

### Pose Estimation

You can use COLMAP or RoDynRF to estimate poses. The latter is recommended for camera motions that are primarily rotations, for which COLMAP tend to fail. Our codebase includes a modified copy of RoDynRF code that does not optimize dynamic parts.

**Note:** you should prepare masks before running pose estimation so that the dynamic foreground can be masked out.

To use COLMAP, run:

```
python ./ui/cli.py \
    run_colmap \
    wild/myvideo
```

COLMAP artifacts are stored in `$ROOT/run_colmap`.

To use RoDynRF, run:

```
python ./ui/cli.py \
    run_rodynrf \
    wild/myvideo
```

The training artifacts are stored in `$ROOT/train_rodynrf`. When training for the first time, it also compute depths and optical flow and store them inside this folder. These data are reused for subsequent training sessions if you need to tune the hyperparameters of RoDynRF.

### Depth Estimation

```
python ./ui/cli.py \
    run_depth \
    wild/myvideo
```

## 1. Video to Images

> preprocess/videos_to_images.py

Obtain image files from a video file you captured or downloaded. We have tested our method with videos up to 200 frames, which is about 7 seconds if the video is 30fps. You can use a video editing tool to convert the desired clip to images, write a ffmpeg command, or use our script that calls `ffmpeg` for you:

```
python ./preprocess/video_to_images.py \
    input=/path/to/myvideo.mp4 \
    output=$ROOT/rgb_1x \
    step=2 \
    limit=200 \
    skip=2s
```

The option `step=2` drops every 2 frames, effectively treating 60fps videos as 30fps. `limit` is the number of frames to produce. `skip` specify the `-ss` flag of ffmpeg.

The script also downsamples the video's short edge to 1080 pixels if needed.

It is a convention to store the images in the `rgb_1x` folder.


## 2. COLMAP

> preprocess/run_colmap.py

```
python ./preprocess/run_colmap.py \
    images=$ROOT/rgb_1x \
    output=$ROOT/run_colmap

# if COLMAP succeeds...
mkdir $ROOT/colmap
cp $ROOT/run_colmap/poses_bounds.npy $ROOT/colmap
```

For instructions of running RoDynRF manually, check the `run_rodynrf` function in [ui/commands.py](../ui/commands.py).

## 3. Optical Flow

> preprocess/run_flow.py

First, obtain optical flow:

```
python preprocess/run_flow.py \
    input=$ROOT/rgb_1x \
    output=$ROOT/flow \
    model=/data/matting/pretrained/raft/raft-things.pth
```

Then, obtain confidence masks:

```
python third_party/omnimatte/datasets/confidence.py \
    $ROOT/flow \
    --rgb $ROOT/rgb_1x
```

## 4. Depth Estimation

> preprocess/run_depth.py

```
python preprocess/run_depth.py \
    input=$ROOT/rgb_1x \
    output=$ROOT/depth \
    model=/data/matting/pretrained/midas/dpt_beit_large_512.pt
```

## 5. Coarse Masks

> preprocess/run_segmentation.py

To produce coarse masks with our scripts, you'll need an environment with GUI. It does not need GPU compute capacity and can be virtually any laptop that runs Python.

First, run Detectron with this script (no GUI needed, requires GPU):

```
python preprocess/run_segmentation.py \
    input=$ROOT/rgb_1x \
    output=$ROOT/segmentation.zip
```

Then, you can use `tools/convert_masks.py` which is a GUI tool.

> If this is a different computer, it requires PyTorch (CPU) and these packages:
>
> ```
> opencv-python numpy pillow scikit-image hydra-core
> ```
>
> Also, you need to copy `rgb_1x` and `segmentation.zip` locally.

To use the script, run:

```
python tools/convert_masks.py ~/Downloads/myvideo
```

- Use numbers 0~9 and up down arrow keys to select a detected mask (shown in green)
- Use left mouse button to draw and _add_ masked region; right mouse button to _subtract_ mask
- Use left right arrow keys to save and move to next image
    - When moving to a new image, the program pre-selects a mask with maximum overlap (shown in blue)
    - You can keep pressing right if the masks are good, or make edits if needed

## Extra: convert to Omnimatte and Nerfies formats

```
python ./tools/matting_to_nerfies.py \
    $ROOT \
    /data/nerfies/wild/myvideo

python ./tools/matting_to_omnimatte.py \
    $ROOT \
    /data/omnimatte/wild/myvideo
```
