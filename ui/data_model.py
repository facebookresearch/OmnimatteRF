# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
from typing import List, Optional

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ExperimentAvailability:
    checkpoints: List[str] = field(default_factory=list)
    evals: List[str] = field(default_factory=list)
    other_artifacts: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        data = [
            f"{k}({', '.join(self.__dict__[k])})"
            for k in ["checkpoints", "evals", "other_artifacts"]
            if len(self.__dict__[k]) > 0
        ]
        return f"Experiment[{', '.join(data)}]"


@dataclass_json
@dataclass
class DataAvailability:
    images: bool = False
    poses: List[str] = field(default_factory=list)
    masks: List[str] = field(default_factory=list)
    flow: bool = False
    depth: bool = False
    homography: bool = False
    segmentation: bool = False
    other_artifacts: List[str] = field(default_factory=list)
    other_formats: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        data = [k for k in ["images", "flow", "depth", "homography", "segmentation"] if self.__dict__[k]]
        data += [
            f"{k}({', '.join(self.__dict__[k])})"
            for k in ["poses", "masks", "other_artifacts", "other_formats"]
            if len(self.__dict__[k]) > 0
        ]
        return f"Data[{', '.join(data)}]"


@dataclass_json
@dataclass
class Experiment:
    category: str
    video: str
    method: str
    name: str
    notes: str
    local: ExperimentAvailability = field(default_factory=ExperimentAvailability)
    remote: ExperimentAvailability = field(default_factory=ExperimentAvailability)


@dataclass_json
@dataclass
class Dataset:
    category: str
    video: str
    local: DataAvailability = field(default_factory=DataAvailability)
    remote: DataAvailability = field(default_factory=DataAvailability)


@dataclass_json
@dataclass
class RemoteConfig:
    endpoint: str
    bucket: str
    access_key: str
    secret_key: str


@dataclass_json
@dataclass
class LocalConfig:
    data_root: str
    output_root: str
    training_folder: str = field(default="train")


@dataclass_json
@dataclass
class DataManagerConfig:
    local: LocalConfig
    remote: Optional[RemoteConfig]
