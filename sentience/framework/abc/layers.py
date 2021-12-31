from typing import Tuple, Union
from dataclasses import dataclass, field

@dataclass
class Layer:
    pass

@dataclass
class Activation(Layer):
    activation_name: Union[str, None] = None
    name = "Activation"

@dataclass
class Tanh(Activation):
    activation_name: str = "tanh"
    name = "Tanh"

@dataclass
class ReLU(Activation):
    activation_name: str = "relu"
    name = "ReLU"

@dataclass
class Sigmoid(Activation):
    activation_name: str = "sigmoid"
    name = "Sigmoid"

@dataclass
class Dense(Layer):
    input_size: list
    output_size: list
    activation : Union[str, Activation] = None
    name = "Dense"

@dataclass
class Conv1D(Layer):
    input_size: Tuple[int, int]
    filters: int
    kernel_size: Union[int, Tuple[int]]
    strides: Union[int, Tuple[int]] = 1
    padding: str = "valid"
    dilation_rate: int = 1
    channels_first: bool = False
    name = "Conv1D"

@dataclass
class Conv2D(Layer):
    input_size: Tuple[int, int, int]
    filters: int
    kernel_size: Union[int, Tuple[int, int]]
    strides: Union[int, Tuple[int, int]] = (1, 1)
    padding: str = "valid"
    dilation_rate: int = 1
    channels_first: bool = False
    name = "Conv2D"

@dataclass
class RNN(Layer):
    input_size: Tuple[int, ...]
    units: int
    return_sequences: bool = False
    return_state: bool = False
    bidirectional: bool = False
    time_major: bool = False
    name = "RNN"

@dataclass
class LSTM(RNN):
    activation: Union[str, Activation] = "tanh"
    recurrent_activation: Union[str, Activation] = "sigmoid"
    name = "LSTM"

@dataclass
class GRU(RNN):
    activation: Union[str, Activation] = Tanh
    recurrent_activation: Union[str, Activation] = Sigmoid
    name = "GRU"

