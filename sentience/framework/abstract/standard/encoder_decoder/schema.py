   
from typing import Dict
from sentience.framework.abstract.layers import ReLU, Tanh, Sigmoid
from sentience.framework.abstract import ModelSchema
from sentience.framework.abstract.randomization import Boolean, Discrete, Categorical

class EncoderDecoderSchema(ModelSchema):
    """
    Defines schema for an Encoder-Decoder model with Dense/Linear layers.
    
    Keyword arguments:-
    feature_size -- int. The number of input features. 
    max_encoded_dim -- int. Maximum number of features that can be used to encode the input.
    min_encoded_dim -- int. Minimum number of features that can be used to encode the input.
    max_encoder_layers -- int. Maximum number of layers in the encoder block, excluding the encoding layer. 
    min_encoder_layers -- int. Minimum number of layers in the encoder block, excluding the encoding layer. 
    max_decoder_layers -- int. Maximum number of layers in the decoder block, excluding the decoding layer. 
    min_decoder_layers -- int. Minimum number of layers in the decoder block, excluding the decoding layer. 
    """
    
    def __init__(self, 
                 feature_size:int,
                 max_encoded_dim:int=1000,
                 min_encoded_dim:int=1,
                 max_encoder_layers:int=5,
                 min_encoder_layers:int=1,
                 max_decoder_layers:int=5,
                 min_decoder_layers:int=1) -> None:
        super().__init__()
        # self.reduce_expand = reduce_expand
        self.FEATURE_SIZE = feature_size
        self.MAX_ENCODED_DIM = max_encoded_dim
        self.MIN_ENCODED_DIM = min_encoded_dim
        self.MAX_ENCODER_LAYERS = max_encoder_layers
        self.MIN_ENCODER_LAYERS = min_encoder_layers
        self.MAX_DECODER_LAYERS = max_decoder_layers
        self.MIN_DECODER_LAYERS = min_decoder_layers
    
    # def __layer_units
    
    def create_schema(self) -> Dict[str, Dict[str, Boolean]]:
        """
        Creates a new schema with the bounds specified by the class attributes. 
        """
        
        _schema = {"encoder": dict(),
                   "decoder": dict()}
        
        _schema["encoder"]["encoder_layers"] = Discrete(low=self.MIN_ENCODER_LAYERS,
                                                        high=self.MAX_ENCODER_LAYERS)
        _schema["encoder"]["encoded_dimension"] = Discrete(low=self.MIN_ENCODED_DIM,
                                               high=self.MAX_ENCODED_DIM)
        _schema["encoder"]["activation"] = Categorical([ReLU, Tanh, Sigmoid], optional=True)
        _schema["decoder"]["decoder_layers"] = Discrete(low=self.MIN_DECODER_LAYERS,
                                                        high=self.MAX_DECODER_LAYERS)
        _schema["decoder"]["activation"] = Categorical([ReLU, Tanh, Sigmoid], optional=True)
        return _schema
    
    def load_schema(self, schema: Dict):
        return schema
    
    def del_schema(self):
        return dict()
    
    SCHEMA = property(create_schema, load_schema, del_schema)