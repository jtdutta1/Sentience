from abc import ABC, abstractmethod
from typing import Any, Dict
from sentience.framework.abstract.model_schema import ModelSchema

class Architecture(ABC):
    """
    Model architecture base class. 
    
    Keyword arguments:-
    schema -- ModelSchema. Stores the model schema from which an architecture is constructed. 
    """
    def __init__(self, schema:ModelSchema):
        self.SCHEMA = schema.SCHEMA
        self.ARCH = dict()
    
    @abstractmethod
    def create_arch(self) -> Dict[str, Any]:
        """Create an architecture from the defined schema. Must be implemented for each 
        network search space.
        
        Returns:-
        Dict with the model architecture.
        """
        pass
    
    # ARCH = property(create_arch)