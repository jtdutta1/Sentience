"""
Module dealing with model creation in tensorflow
"""
import tensorflow as tf
from typing import List
from warnings import DeprecationWarning, warn
from tensorflow.python.framework.ops import Tensor
from sentience.framework.abc import layers

from sentience.framework.abc.architecture import Architecture
from sentience.framework.abc.layers import Layer
from sentience.framework.tf.layers import get_layer

# The following idea is made invalid in light of using 
# Sequential layer from Keras. It should be removed before 
# launch of current version.
class Block(tf.keras.layers.Layer):
    """Block class contains sequential connection of different layers. 
    This is analogous to a block cell in the architecture definition.
    """
    
    def __init__(self, 
                 list_of_layers: List[Layer], 
                 *args, 
                 **kwargs) -> None:
        super(Block, self).__init__(*args, **kwargs)
        warn("This class is deprecated and will be removed in future versions",
             DeprecationWarning)
        self.layers = list_of_layers
    
    def call(self, 
             inputs: Tensor) -> Tensor:
        x = inputs
        for i in self.layers:
            x = i(x)
        return x
        

def create_model(ARCH:Architecture) -> tf.keras.Model:
    """Creates a model from the given architecture.
    
    Keyword arguments:-
    ARCH -- Architecture object with the generated architecture. 
    """
    model_name = None
    input_shape = []
    model_layer_out = None
    for k, v in ARCH.ARCH:
        # Check for 3 default ks
        if k == "name":
            model_name = v
        elif k == "input_shape":
            input_shape = v
        elif k == "layers":
            try:
                assert(type(v) == list)
            except AssertionError:
                raise AssertionError("Can only accept layers defined in lists. Define a custom model creator for other types.")
            while True:
                try:
                    assert(len(input_shape) > 0)
                    break
                except AssertionError:
                    if "input_shape" in ARCH.ARCH:
                        warn("Always make sure the input_shape always comes before the layer definitions in the architecture.",
                                UserWarning)
                        input_shape = ARCH.ARCH["input_shape"]
                    else:
                        raise AssertionError("Input shapes needs to be defined in the architecture")
            
            # Define the Input layer to the model. 
            # We will construct the Keras model using functional API
            model_layer_out = tf.keras.layers.Input(shape=input_shape)
            _input_tensor = model_layer_out
            for layer in v:
                model_layer_out = get_layer(layer)(model_layer_out)
        else:
            raise KeyError
    _model = tf.keras.Model(inputs=_input_tensor,
                            outputs=model_layer_out,
                            name=model_name)
    return _model