# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import os


def write_json(file, obj):
    text = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)

    os.makedirs(os.path.split(file)[0], exist_ok=True)
    with open(file, "w", encoding="utf-8") as f:
        f.write(text + "\n")


def read_json(file, default_factory):
    if not os.path.isfile(file):
        return default_factory()
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)


def write_data_json(file, obj, cls):
    """Serialize an object of a dataclass_json class"""
    os.makedirs(os.path.split(file)[0], exist_ok=True)
    text = cls.schema().dumps(
        obj, many=isinstance(obj, list), indent=2, ensure_ascii=False, sort_keys=True
    )
    with open(file, "w", encoding="utf-8") as f:
        f.write(text)


def read_data_json(file, cls):
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
    return cls.schema().loads(text)
