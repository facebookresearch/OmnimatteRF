Using the CLI
---

The CLI source file is `ui/cli.py`. The commands are defined in `ui/commands.py`.

All text below assumes working directory being root of source code.

## Data Management

The CLI has a data manager which assumes the following data organization:

- Two folders `$DATA` and `$OUTPUT` are provided.
- A video is stored in `$DATA/[format]/[dataset]/[video]`
    - Example: `$DATA/matting/movies/chicken`.
    - `[format]` is `matting` for our data, it also recognizes `omnimatte` and `nerfies`
- An experiment is stored in `$OUTPUT/train/[dataset]/[video]/[method]/[experiment]`.
    - Example: `$OUTPUT/train/movies/chicken/matting/20230830-103000`
    - The key of our method is `matting`

## Initial setup

Before the first use, configure the data manager:

- Copy `data_manager_example.json` to `data_manager.json`
- Set `data_root` and `output_root`

Also, the following files are needed to process new videos:

- `$DATA/pretrained/raft/raft-things.pth` (for optical flow)
- `$DATA/pretrained/midas/dpt_beit_large_512.pt` (for monocular depth)

To run RoDynRF, you'll need `midas_v21-f6b98070.pt`.

## Basic Usage

A CLI command looks like:

```
python ./ui/cli.py action --arg1 v1 --arg2 v2 -- extra_arg1=v1 extra_arg2=v2
```

The arguments before `--` are for the action, which uses Python argparse. The ones after `--`, if applicable, are passed to the underlying scripts (train, eval, preprocess, etc.) which usually use Hydra.

If you have downloaded our data, run these commands to check if you have placed them correctly:

```
python ./ui/cli.py list_data
python ./ui/cli.py print_data wild/walk
```

## Preprocess Data

See [Using Your Video](using-your-video.md) for instructions on processing new videos.

## Train

The basic command to start training is:

```
python ./ui/cli.py train_ours wild/walk
```

To train with depth supervision:

```
python ./ui/cli.py \
    train_ours \
    wild/bouldering \
    --use_depths \
    -- \
    fg_losses=[alpha_reg,brightness_reg,flow_recons,mask,recons,warped_alpha,bg_tv_reg,robust_depth_matching,bg_distortion] \
    fg_losses.robust_depth_matching.config.alpha=0.1 \
    fg_losses.bg_distortion.config.alpha=0.01 \
    data_sources.llff_camera.scene_scale=0.2
```

By default the experiment name is current date and time. Use `--name xxx` to append additional text. Use `--no_time_prefix` to exclude the date time prefix.

## Eval

The eval command expects `[dataset]/[video]/[method]/[experiment]`. You can use `python ./ui/cli.py list_experiments` to print out the experiments.

```
python ./ui/cli.py \
    eval_ours \
    movies/chicken/matting/20230830-103000
```

The output from a checkpoint `checkpoing_[step].pth` will be in the experiment folder under `eval/[step]`. By default, it runs eval with all saved checkpoints. You can specify a checkpoint to run (e.g. only the last) with `--step 15000`.

## Clean background retraining

If you find some shadows captured in both foreground and background layers, it may be possible to obtain a clean background by training the TensoRF model from scratch, using the mask from the jointly-trained foreground.

The eval script generates `fg_alpha` which is the combined alpha of foreground layers. You can train the background RF using:

```
python \
    ui/cli.py \
    train_ours \
    --config train_bg \
    --name retrain_bg \
    --mask \
    $OUTPUT/train/wild/walk/matting/20230830-103000/eval/15000/fg_alpha \
    wild/walk
```

The masks are thresholded at 0.5. If that doesn't work well (e.g. most shadows are captured with an alpha of 0.3), use the following script:

```
python \
    ui/cli.py \
    s1_mask \
    movies/chicken/matting/20230830-103000 \
    --threshold 0.2
```

It'll generate mask files in `s1_mask` and visualization in `s1_visualization`. You can check and rerun as necessary. Finally, replace `fg_alpha` with `s1_mask` in the above script.
