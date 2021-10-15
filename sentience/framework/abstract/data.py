from abc import ABC, abstractmethod

class Dataset(ABC):
    """
    Base class for defining a Dataset object.
    """
    
    @abstractmethod
    def data_gen(self, mapfn=None):
        """
        Generates batch wise data (features, labels)
        
        Keyword arguments:-
        mapfn -- A function to map the input sequence to.
        """
        pass
    