from typing import List, Tuple, Union
from dataclasses import dataclass, field

@dataclass
class Layer:
    name: str = "layer"

@dataclass
class Activation(Layer):
    activation_name: Union[str, None] = None

@dataclass
class Tanh(Activation):
    activation_name: str = "tanh"

@dataclass
class ReLU(Activation):
    activation_name: str = "relu"

@dataclass
class Sigmoid(Activation):
    activation_name: str = "sigmoid"

@dataclass
class Dense(Layer):
    input_size: list = field(default_factory=list)
    output_size: list = field(default_factory=list)
    activation : Union[str, Activation] = None

@dataclass
class Conv1D(Layer):
    input_size: Tuple[int, int] = None
    filters: int = 1
    kernel_size: Union[int, Tuple[int]] = 3
    padding: str = "valid"
    dilation_rate: int = 1
    channels_first: bool = False

@dataclass
class Conv2D(Layer):
    input_size: Tuple[int, int, int] = None
    filters: int = 1
    kernel_size: Union[int, Tuple[int, int]] = 3
    padding: str = "valid"
    dilation_rate: int = 1
    channels_first: bool = False

@dataclass
class RNN(Layer):
    input_size: Tuple[int, ...] = None
    units: int = None
    return_sequences: bool = False
    return_state: bool = False
    bidirectional: bool = False
    time_major: bool = False

@dataclass
class LSTM(RNN):
    activation: Union[str, Activation] = Tanh
    recurrent_activation: Union[str, Activation] = Sigmoid

@dataclass
class GRU(RNN):
    activation: Union[str, Activation] = Tanh
    recurrent_activation: Union[str, Activation] = Sigmoid

