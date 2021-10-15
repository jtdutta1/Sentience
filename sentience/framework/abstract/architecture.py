from abc import ABC, abstractmethod
from sentience.framework.abstract.model import Model

class Architecture(ABC):
    """
    Architecture abstract class that holds the base architecture definition of the model.
    """
    
    @abstractmethod
    def __init__(self, arch_dict) -> None:
        super().__init__()
        pass
    
    @abstractmethod
    def create_model(self) -> Model:
        """
        Creates a base model from the architecture dictionary
        """
        pass
    