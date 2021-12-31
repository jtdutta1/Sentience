class Error(Exception):
    
    def __init__(self):
        self.message = None
    
    def __repr__(self) -> str:
        return self.message
    
    def __str__(self) -> str:
        return self.message

class DimensionalityMismatchError(Error):
    def __init__(self, reduce_expand, *args) -> None:
        super().__init__(*args)
        if reduce_expand:
            self.message = "Expected dimensional reduction then expansion but got the reverse."
        else:
            self.message = "Expected dimensional expansion then reduction but got the reverse."

class ActivationNotFoundError(Error):
    def __init__(self, name:str, *args) -> None:
        super().__init__(*args)
        self.message = f"{name} activation not found."

class LayerTypeMismatchError(Error):
    def __init__(self, obj, *args) -> None:
        super().__init__(*args)
        self.message = f"Type of object is not Layer. Expected type Layer, given type {type(obj).__name__}."

class LayerNotFoundError(Error):
    def __init__(self, name:str, *args: object) -> None:
        super().__init__(*args)
        self.message = f"{name} layer not found."