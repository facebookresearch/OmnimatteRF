# Copyright (c) Meta Platforms, Inc. and affiliates.

from ui.data_manager import DataManager
from pathlib import Path

dm_config_file = Path(__file__).parent.parent / "data_manager.json"


def create_data_manager() -> DataManager:
    if not dm_config_file.exists():
        raise ValueError(f"Config file {dm_config_file} does not exist. Create one by copying data_manager_example.json.")

    return DataManager(dm_config_file)
