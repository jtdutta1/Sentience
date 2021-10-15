from dataclasses import dataclass

@dataclass
class Layer:
    name: str = "layer"

@dataclass
class Dense(Layer):
    input_size: tuple = (None, ..., None)
    output_size: tuple = (None, ..., None)
    activation : str = None

@dataclass
class Conv1D(Layer):
    input_size: tuple = (None, None)
    filters: int = 1
    filter_size: int = 3
    padding: str = "valid"
    dilution: int = 1
    channels_first: bool = False

@dataclass
class Conv2D(Layer):
    input_size: tuple = (None, None, None)
    filters: int = 1
    filter_size: int = 3
    padding: str = "valid"
    dilution: int = 1
    channels_first: bool = False

@dataclass
class RNN(Layer):
    input_size: tuple = None
    units: int = None
    return_sequences: bool = False
    return_state: bool = False
    bidirectional: bool = False

class Model:
    pass