# Inspired from https://github.com/eliberis/uNAS/blob/master/schema_types.py

from numpy.random import default_rng
from typing import Any, List, Union


class Boolean:
    def __init__(self, optional:bool=False):
        self.optional = optional
    
    def __maybe_decide__(self) -> bool:
        self.rng = default_rng()
        if self.optional:
            return True if self.rng.random() > 0.5 else False
        else:
            return True
    
    def decide(self) -> bool:
        return self.__maybe_decide__()
    
class Categorical(Boolean):
    def __init__(self, categories: List[str], optional:bool=False):
        super().__init__(optional=optional)
        self.categories = categories
    
    def decide(self) -> Union[Any, None]:
        if self.__maybe_decide__():
            top = len(self.categories)
            rand_ind = int(self.rng.integers(0, top, size=1, endpoint=False))
            return self.categories[rand_ind]
        else:
            return None

class Discrete(Boolean):
    def __init__(self, 
                 low: int,
                 high: int,
                 optional:bool=False):
        super().__init__(optional=optional)
        self.low = low
        self.high = high
    
    def decide(self) -> Union[int, None]:
        if self.__maybe_decide__():
            return int(self.rng.integers(self.low, self.high, size=1, endpoint=True))
        else:
            return None

class Continuous(Boolean):
    def __init__(self, 
                 low: float,
                 high: float,
                 optional: bool = False):
        super().__init__(optional=optional)
        self.low = low
        self.high = high
    
    def decide(self) -> Union[float, None]:
        if self.__maybe_decide__():
            return (self.high - self.low) * self.rng.random() + self.low
        else:
            return None
    
        