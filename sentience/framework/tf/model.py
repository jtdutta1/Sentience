"""
Module dealing with model creation in tensorflow
"""
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

class Block(Layer):
    """Block class contains sequential connection of different layers. 
    This is analogous to a block cell in the architecture definition.
    """
    
    def __init__(self, block_arch):
        super(Block, self).__init__()