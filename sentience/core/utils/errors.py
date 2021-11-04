class Error(Exception):
    
    def __init__(self):
        self.message = None
    
    def __repr__(self) -> str:
        return self.message
    
    def __str__(self) -> str:
        return self.message

class DimensionalityMismatchError(Error):
    def __init__(self, reduce_expand, *args: object) -> None:
        super().__init__(*args)
        if reduce_expand:
            self.message = "Expected dimensional reduction then expansion but got the reverse."
        else:
            self.message = "Expected dimensional expansion then reduction but got the reverse."

class ActivationNotFoundError(Error):
    def __init__(self, name:str):
        self.message = f"{name} activation not found"
        