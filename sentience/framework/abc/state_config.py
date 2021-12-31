from _typeshed import NoneType
from dataclasses import dataclass
from typing import Union, List
from sentience.framework.pt.model import PTModel
from sentience.framework.tf.model import TFModel
from sentience.framework.abc import Architecture, Dataset

@dataclass
class DatasetState:
    train_dataset: Dataset
    val_dataset: Union[Dataset, NoneType] = None
    test_dataset: Union[Dataset, NoneType] = None
    batch_size: int = 1
    shuffle: bool = False

@dataclass
class ModelState:
    model: Union[TFModel, PTModel]
    name: str
    architecture: Architecture

@dataclass
class TrainerState:
    model_state: ModelState
    epochs: int
    completed_epochs: int
    callbacks: List[object]
    dataset_state: Dataset
    checkpoint_path: str
