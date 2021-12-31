from typing import Dict
from abc import ABC, abstractmethod
from sentience.framework.abc.layers import *
from sentience.framework.abc.randomization import Boolean, Discrete

class ModelSchema(ABC):
    """
    Constructs a model schema based on defined\/random hyperparameters.
    """
    
    # def __init__(self):
    #     self.SCHEMA = property(self.create_schema, 
    #                            self.load_schema, 
    #                            self.del_schema)
    
    @abstractmethod
    def create_schema(self):
        """
        Create a model schema.
        """
    
    @abstractmethod
    def load_schema(self, schema: Dict):
        """
        Load from a predefined schema
        
        Keyword arguments:-
        schema -- A ModelSchema object with a predefined schema. 
        """
    
    @abstractmethod
    def del_schema(self):
        """
        Delete schema
        """

    SCHEMA = property(create_schema, load_schema, del_schema)
 