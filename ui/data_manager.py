# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import os
from os.path import isdir, isfile
from pathlib import Path
from typing import Dict, List

from ui.data_model import (DataAvailability, DataManagerConfig, Dataset,
                           Experiment)
from utils.io_utils import filter_dirs, multi_glob_sorted
from utils.json_utils import read_data_json

logger = logging.getLogger(__name__)


class DataManager:
    def __init__(
        self,
        config_file: str,
    ) -> None:
        self.config: DataManagerConfig = read_data_json(config_file, DataManagerConfig)
        self.local_data_root = Path(self.config.local.data_root)
        self.local_output_root = Path(self.config.local.output_root)

        self.config.local.training_folder = os.environ.get("MATTING_TRAINING_FOLDER", self.config.local.training_folder)

        # configure remote
        self.remote = None
        self.bucket = None

        # dataset_key -> Dataset
        self.datasets: Dict[str, Dataset] = {}
        self.load_local_datasets()

        # dataset_key -> Experiment[]
        self.experiments: Dict[str, List[Experiment]] = {}
        self.experiment_names: Dict[str, Experiment] = {}
        self.load_local_experiments()
        for key in self.datasets:
            if key not in self.experiments:
                self.experiments[key] = []

    def get_local_data_path(self, key: str):
        return self.local_data_root / "matting" / key

    def get_local_experiment_path(self, key: str, method: str, name: str):
        return self.local_output_root / self.config.local.training_folder / key / method / name

    def get_purged_experiment_path(self, key: str, method: str, name: str):
        return self.local_output_root / "purged" / key / method / name

    def get_failed_experiment_path(self, key: str, method: str, name: str):
        return self.local_output_root / "failed" / key / method / name

    def get_local_eval_path(self, exp: Experiment, eval_name: str):
        return self.get_local_experiment_path(f"{exp.category}/{exp.video}", exp.method, exp.name) / "eval" / eval_name

    def load_local_dataset(self, folder: Path):
        datasets = self.datasets
        key = f"{folder.parent.name}/{folder.name}"

        if not isdir(folder):
            return

        if key not in datasets:
            datasets[key] = Dataset(folder.parent.name, folder.name)

        # reload
        datasets[key].local = DataAvailability()

        ds = datasets[key]
        mask_folder = folder / "masks"
        self.set_data_availability(
            ds.local,
            [d.name for d in multi_glob_sorted(folder, "*")],
            [] if not isdir(mask_folder) else [d.name for d in sorted(filter_dirs(mask_folder.glob("*")))]
        )

        if isdir(self.local_data_root / "nerfies" / key):
            ds.local.other_formats.append("nerfies")

    def load_local_datasets(self):
        our_root = self.local_data_root / "matting"
        if not isdir(our_root):
            logger.warning(f"Local dataset root for our format ({our_root}) does not exist.")
            return

        count = 0
        for folder in filter_dirs(our_root.glob("*/*")):
            count += 1
            self.load_local_dataset(folder)

        logger.info(f"Loaded {count} local datasets")

    def load_local_experiment(self, folder: Path):
        datasets = self.datasets
        experiments = self.experiments

        ds_cat = folder.parent.parent.parent.name
        ds_name = folder.parent.parent.name
        ds_key = f"{ds_cat}/{ds_name}"
        method = folder.parent.name
        notes = ""

        notes_file = folder / "notes.txt"
        if isfile(notes_file):
            with open(notes_file, "r", encoding="utf-8") as f:
                notes = f.read().strip()

        if ds_key not in datasets:
            datasets[ds_key] = Dataset(ds_cat, ds_name)

        if ds_key not in experiments:
            experiments[ds_key] = []

        exp = Experiment(
            ds_cat,
            ds_name,
            method,
            folder.name,
            notes,
        )
        experiments[ds_key].append(exp)
        self.experiment_names[f"{exp.category}/{exp.video}/{exp.method}/{exp.name}"] = exp

        avail = exp.local
        avail.checkpoints = self.sorted_checkpoints(
            [f.name for f in folder.glob("checkpoints/checkpoint_*")] +
            [f.name for f in folder.glob("checkpoint_*")] +
            [f.name for f in folder.glob("checkpoint")]
        )
        avail.evals = self.sorted_evals([f.name for f in folder.glob("eval/*")])
        avail.other_artifacts = [f.name for f in multi_glob_sorted(folder, "*")]
        avail.other_artifacts = [f for f in avail.other_artifacts if f not in {"checkpoints", "eval"}]

    def load_local_experiments(self):
        exp_root = self.local_output_root / self.config.local.training_folder
        count = 0

        # out_root/train/[catetory]/[video]/[method]/[exp]
        for folder in sorted(filter_dirs(exp_root.glob("*/*/*/*"))):
            count += 1
            self.load_local_experiment(folder)
        logger.info(f"Loaded {count} local experiments")

    @staticmethod
    def sorted_checkpoints(checkpoints: List[str]):
        lst = [[int(Path(s).stem.split("_")[1]) if "_" in s else 0, s] for s in checkpoints]
        lst = sorted(lst, key=lambda t: t[0])
        return [t[1] for t in lst]

    @staticmethod
    def sorted_evals(evals: List[str]):
        lst = [[int(Path(s).stem.split("-")[0].split("_")[0]), s] for s in evals]
        lst = sorted(lst, key=lambda t: t[0])
        return [t[1] for t in lst]

    @staticmethod
    def set_data_availability(avail: DataAvailability, children: List[str], mask_children: List[str]):
        for name in children:
            if name == "rgb_1x":
                avail.images = True
            elif name == "flow":
                avail.flow = True
            elif name == "depth":
                avail.depth = True
            elif name == "homography":
                avail.homography = True
            elif name == "segmentation.zip":
                avail.segmentation = True
            elif name == "colmap":
                avail.poses.append("colmap")
            elif name == "rodynrf":
                avail.poses.append("rodynrf")
            elif name == "poses.json":
                avail.poses.append("blender")
            elif name == "masks":
                pass
            else:
                avail.other_artifacts.append(name)

        for child in mask_children:
            avail.masks.append(child)
