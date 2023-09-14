# Copyright (c) Meta Platforms, Inc. and affiliates.

import importlib
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")
TImpl = TypeVar("TImpl")


class Registry(Generic[T]):
    def __init__(self, name: str, base: Callable[..., T]) -> None:
        super().__init__()
        self.name = name
        self.constructors: Dict[str, Callable[..., T]] = {}

    def add(self, name: str, constructor: Callable[..., T]) -> None:
        logger.info(f"Register {self.name}: {name}")
        self.constructors[name] = constructor

    def register(self, name: str):
        def adder(cls: Type[TImpl]) -> Type[TImpl]:
            self.add(name, cls)
            return cls
        return adder

    def build(self, name: str, kwargs: Dict[str, Any]) -> T:
        return self.constructors[name](**kwargs)


def create_registry(name: str, base: Type[T]) -> Tuple[
    Registry[T],
    Callable[[str], Callable[[Type[TImpl]], Type[TImpl]]],
    Callable[[str, Dict[str, Any]], T]
]:
    registry = Registry(name, base)
    return registry, registry.register, registry.build


def import_children(path: str, module: str):
    folder = Path(path).parent
    for file in folder.glob("*.py"):
        if file.name == "__init__.py":
            continue
        name = module + "." + file.stem
        importlib.import_module(name)
