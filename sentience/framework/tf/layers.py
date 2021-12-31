from tensorflow.keras import layers, activations
from sentience.framework.abc.layers import *
from sentience.core.utils.errors import ActivationNotFoundError, LayerNotFoundError, LayerTypeMismatchError

def get_activation(obj:Union[str, Layer]):
    if isinstance(obj, Layer):
        try:
            assert type(obj) in [Tanh, Sigmoid, ReLU]
        except AssertionError:
            raise ActivationNotFoundError(type(obj).__name__)
        return _l2l_map[type(obj).__name__](obj)
    else:
        raise LayerTypeMismatchError(obj)

def tanh(obj:Layer):
    return activations.tanh

def relu(obj:Layer) -> layers.ReLU:
    return layers.ReLU()

def sigmoid(obj:Layer):
    return activations.sigmoid

def dense(obj:Layer) -> layers.Dense:
    units = obj.output_size[-1]
    activation = get_activation(obj.activation)
    return layers.Dense(units=units, activation=activation)

def conv1d(obj:Layer) -> layers.Conv1D:
    kwargs = {"filters": obj.filters,
              "kernel_size": obj.kernel_size,
              "strides": obj.strides,
              "padding": obj.padding,
              "data_format": "channels_first" if obj.channels_first else "channels_last",
              "dilation_rate": obj.dilation_rate}
    return layers.Conv1D(**kwargs)

def conv2d(obj:Layer) -> layers.Conv2D:
    kwargs = {"filters": obj.filters,
              "kernel_size": obj.kernel_size,
              "strides": obj.strides,
              "padding": obj.padding,
              "data_format": "channels_first" if obj.channels_first else "channels_last",
              "dilation_rate": obj.dilation_rate}
    return layers.Conv2D(**kwargs)

def rnn(obj:Layer) -> layers.RNN:
    kwargs = {"cell": obj.units,
              "return_sequences": obj.return_sequences,
              "return_state": obj.return_state,
              "time_major": obj.time_major}
    rnn_layer = layers.RNN(**kwargs)
    if obj.bidirectional:
        return layers.Bidirectional(rnn_layer)
    else:
        return rnn_layer

def lstm(obj:Layer) -> layers.LSTM:
    kwargs = {"cell": obj.units,
              "activation":get_activation(obj.activation),
              "recurrent_activation": get_activation(obj.recurrent_activation),
              "return_sequences": obj.return_sequences,
              "return_state": obj.return_state,
              "time_major": obj.time_major}
    rnn_layer = layers.LSTM(**kwargs)
    if obj.bidirectional:
        return layers.Bidirectional(rnn_layer)
    else:
        return rnn_layer

def gru(obj:Layer) -> layers.GRU:
    kwargs = {"cell": obj.units,
              "activation":get_activation(obj.activation),
              "recurrent_activation": get_activation(obj.recurrent_activation),
              "return_sequences": obj.return_sequences,
              "return_state": obj.return_state,
              "time_major": obj.time_major}
    rnn_layer = layers.GRU(**kwargs)
    if obj.bidirectional:
        return layers.Bidirectional(rnn_layer)
    else:
        return rnn_layer



_l2l_map = {"Tanh": tanh,
            "ReLU": relu,
            "Sigmoid": sigmoid,
            "Dense": dense,
            "Conv1D": conv1d,
            "Conv2D": conv2d,
            "RNN": rnn,
            "LSTM": lstm,
            "GRU": gru}

def get_layer(obj) -> layers.Layer:
    try:
        assert isinstance(obj, Layer)
    except AssertionError:
        raise LayerTypeMismatchError(obj)
    
    try:
        return _l2l_map[obj.name](obj)
    except KeyError:
        raise LayerNotFoundError(obj.name)