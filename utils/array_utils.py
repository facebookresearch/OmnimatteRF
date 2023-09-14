# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np


def log_stats(logger_func, name: str, array: np.ndarray):
    """Log the statistics of an array"""
    keys = ["min", "max", "mean", "median"]
    values = [func(array) for func in [np.min, np.max, np.mean, np.median]]

    percentiles = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    keys += [str(v) for v in percentiles]
    values += list(np.percentile(values, percentiles))

    length = max([len(k) for k in keys])
    message = [
        f"stats of {name}:",
        *(f"{key:>{length}} {value}" for key, value in zip(keys, values)),
    ]

    logger_func("\n".join(message))
