# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Any, Dict


def inject_dict(cfg: Dict[str, Any], injection: Dict[str, Any]):
    """Merge item from injection if the key exists in cfg and value is None.

    Note: this is different from dictionary merging.
    Returns: input cfg object
    """
    for k, v in injection.items():
        if k in cfg and cfg[k] is None:
            cfg[k] = v
    return cfg
