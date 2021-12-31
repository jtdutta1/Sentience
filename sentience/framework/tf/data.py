import os
from typing import AnyStr
from tensorflow import data
from sentience.framework.abc.data import Dataset

class TFDataset(Dataset):
    """
    A Dataset object that generates data during training and evaluation. This feature isn't completed
    yet.
    
    Keyword arguments:-
    path -- Path to a manifest file.
    """
    
    def __init__(self, 
                 path: AnyStr) -> None:
        super().__init__()
        
        # Raise incomplete error.
        raise ValueError(message="This class isn't completed yet. Please define your own dataset object.")
        
        try:
            assert os.path.exists(path)
        except AssertionError:
            raise FileNotFoundError(message="Manifest file not found at the path given.")
        self.path = path
    
    def data_gen(self, mapfn=None) -> data.Dataset:
        """
        
        """
        tfdt = data.TextLineDataset(self.path)
        