# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import hydra
import numpy as np
import torch
from detectron2 import model_zoo
from hydra.core.config_store import ConfigStore
from PIL import Image
from tqdm import tqdm

from utils.io_utils import mkdir, multi_glob_sorted


@dataclass
class RunSegmentationConfig:
    model: str
    device: str
    input: str
    output: str


ConfigStore.instance().store(name="run_segmentation_schema", node=RunSegmentationConfig)


@hydra.main(version_base="1.2", config_path="config", config_name="run_segmentation")
def main(cfg: RunSegmentationConfig):
    device = cfg.device
    in_root = Path(cfg.input)

    write_to_zip = cfg.output.endswith(".zip")
    if write_to_zip:
        zip_folder = Path(cfg.output).stem
        out_zip = ZipFile(cfg.output, "w", compression=ZIP_DEFLATED)
    else:
        out_root = mkdir(cfg.output)

    in_files = multi_glob_sorted(in_root, ["*.png", "*.jpg"])

    model = model_zoo.get(cfg.model, trained=True).eval().to(device)

    for file in tqdm(in_files):
        image = np.array(Image.open(file), dtype=np.float32)
        image = image[..., :3]  # RGBA to RGB
        image = torch.from_numpy(image.transpose(2, 0, 1))
        inputs = [
            {
                "image": image.to(device),
                "height": image.shape[1],
                "width": image.shape[2],
            }
        ]
        with torch.no_grad():
            instances = model(inputs)[0]["instances"].to("cpu")

        fields = instances.get_fields()
        result = {
            "pred_boxes": fields["pred_boxes"].tensor,
            **{k: fields[k] for k in ["scores", "pred_classes", "pred_masks"]}
        }

        if write_to_zip:
            with out_zip.open(f"{zip_folder}/{file.stem}.ckpt", mode="w") as f:
                torch.save(result, f)
        else:
            torch.save(result, out_root / f"{file.stem}.ckpt")

    if write_to_zip:
        out_zip.close()


if __name__ == "__main__":
    main()
