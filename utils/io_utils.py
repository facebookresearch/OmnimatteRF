# Copyright (c) Meta Platforms, Inc. and affiliates.

import itertools
import os
from pathlib import Path
from typing import List, Union


def mkdir(path, exist_ok=True) -> Path:
    """Create the directory and returns a Path object."""
    path = Path(path)
    os.makedirs(path, exist_ok=exist_ok)
    return path


def multi_glob_sorted(path: Union[str, Path], appendices: Union[str, List[str]]) -> List[Path]:
    """List files in directory if its extension is in provided list.

    Returns: sorted Path objects
    """
    if not isinstance(appendices, list):
        appendices = [appendices]

    path = Path(path)
    return sorted(itertools.chain(*[path.glob(app) for app in appendices]))


def filter_dirs(paths: List[Path]) -> List[Path]:
    return [d for d in paths if os.path.isdir(d)]
