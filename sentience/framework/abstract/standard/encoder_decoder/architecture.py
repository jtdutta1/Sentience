from typing import List, Dict, Any
from sentience.framework.abstract.layers import Dense
from sentience.framework.abstract.randomization import Categorical, Discrete
from sentience.framework.abstract import ModelSchema, Architecture


class EncoderDecoderArch(Architecture):
    """A standard EncoderDecoder architecture generator. Inherits from sentience.framework.abstract.Architecture class.
    
    Keyword arguments:-
    schema -- ModelSchema. Holds the schema from which the architecture is to be generated. 
    input_shape -- list. Shape of the input features to the model.
    output_shape -- list. Shape of the output features from the model. If None the model assumes the shape of the input.
    """
    def __init__(self, 
                 schema:ModelSchema,
                 input_shape:List[int],
                 output_shape:List[int]=None) -> None:
        super().__init__(schema=schema)
        self.input_shape = input_shape
        if output_shape is None:
            self.output_shape = input_shape
        else:
            self.output_shape = output_shape
    
    def create_arch(self) -> Dict[str, Any]:
        _arch = dict()
        _arch["name"] = "EncoderDecoder"
        _arch["input_size"] = self.input_shape
        _arch["encoder_block"] = list()
        
        
        # Find the encoder layers and encoded dimensions. Excluding the encoding layer
        num_encoder_layers = self.SCHEMA["encoder"]["encoder_layers"].decide()
        encoded_dim = self.SCHEMA["encoder"]["encoded_dimension"].decide()
        
        # Create encoder block hidden layers        
        encoder_layers, output_shape = self.__get_block__(num_encoder_layers, self.input_shape, encoded_dim, self.SCHEMA["encoder"]["activation"])
        
        # Create encoding layer
        _arch["encoder_block"] = encoder_layers
        encoder_layer = Dense()
        encoder_layer.input_size = output_shape.copy()
        output_shape = output_shape.copy()
        output_shape[-1] = encoded_dim
        encoder_layer.output_size = output_shape
        encoder_layer.activation = self.SCHEMA["encoder"]["activation"].decide()
        _arch["encoded_layer"] = encoder_layer
        
        # Create decoder layers and decoded dimensions
        num_decoder_layers = self.SCHEMA["decoder"]["decoder_layers"].decide()
        decoded_dim = self.output_shape[-1]
        
        decoder_layers, output_shape = self.__get_block__(num_decoder_layers, output_shape, decoded_dim, self.SCHEMA["decoder"]["activation"])
        _arch["decoder_block"] = decoder_layers
        
        # Create decoding layer
        decoder_layer = Dense()
        decoder_layer.input_size = output_shape.copy()
        output_shape = output_shape.copy()
        output_shape[-1] = decoded_dim
        decoder_layer.output_size = output_shape
        decoder_layer.activation = self.SCHEMA["decoder"]["activation"].decide()
        _arch["decoded_layer"] = decoder_layer
        
        self.ARCH = _arch
    
    def __get_block__(self, 
                      num_layers:int, 
                      input_shape:List[int], 
                      output_dim:int, 
                      activation:Categorical=None):
        """Creates a block of Dense layers. This is done by calculating dimensions of the outputs for the layers
        in the block. The dimensions are bounded by the dimensions of the input and the output specified in the parameters.
        The last dimension represents the dimension of the encoder layer or the output layer, so the output dimension
        should be less than or more than that depending on whether the model is a dimensionality reducing/increasing model. 
        
        Keyword arguments:-
        num_layers -- int. Number of layers to generate in the block.
        input_shape -- list. Shape of the input feature.
        output_dim -- int. Number of features in the output. 
        activation -- sentience.framework.abstract.randomization.Categorical. Whether or not to provide an activation as 
                        defined in the schema.
        """
        # Calculate the units of each layer before that but in a increasing/decreasing 
        reduction = True if output_dim < input_shape[-1] else False
        low = output_dim + 1 if reduction else input_shape[-1] + 1
        high = input_shape[-1] - 1 if reduction else output_dim - 1
        unit_gen = Discrete(low, high)
        layer_dims = [unit_gen.decide() for i in range(num_layers)]
        layer_dims.sort(reverse=True if reduction else False)
        # print(layer_dims)
        
        # construct encoder
        encoder_layers = list()
        input_shape = input_shape
        for dim in layer_dims:
            output_shape = input_shape.copy()
            output_shape[-1] = dim
            layer = Dense()
            layer.input_size = input_shape
            layer.output_size = output_shape
            layer.activation = activation.decide() if activation is not None else None
            input_shape = output_shape.copy()
            encoder_layers.append(layer)
        return encoder_layers, input_shape