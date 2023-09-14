# Copyright (c) Meta Platforms, Inc. and affiliates.

from lib.loss import Loss
from lib.registry import create_registry, import_children

_, register_loss, build_loss = create_registry("Loss", Loss)
import_children(__file__, __name__)
