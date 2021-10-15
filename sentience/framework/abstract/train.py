from abc import ABC, abstractmethod

class Trainer(ABC):
    """
    Base class for trainers. Define your own trainer classes using this as your base.
    """
    
    @abstractmethod
    def train(self, 
              state_path, 
              save_state_interval,
              resume=False,
              model_checkpoint_path=None):
        """
        Train and evaluate your model. Override for custom training mechanism.
        
        Keyword arguments:-
        state_path -- Path to save the state of the training cycle.
        save_state_interval -- Number of epochs after which the state needs to be saved.
        resume -- Resumes from the saved state at path state_path if True. checkpoint_path 
                    also needs to be supplied.
        model_checkpoint_path -- Path to the saved weights to load and continue training.
        """
        pass
    
    @abstractmethod
    def check_state_file_path(self, state_path):
        """
        Checks if a file exists at the defined path. If found a new file name is created by 
        appending an integer starting from 1 until a file of the same name doesn't exist.
        """
        pass
    
    @abstractmethod
    def create_state(self):
        """
        Create a state object during training.
        """
        pass
    
    @abstractmethod
    def save_state(self, save_path):
        """
        Save the state of the training cycle.
        
        Keyword arguments:-
        save_path -- Path to save the state to.
        """
        pass
    
    # @abstractmethod
    # def load_state(self, save_path):
    #     """
    #     Load a saved state to resume training from. 
        
    #     Keyword arguments:-
    #     save_path -- Path to load the state from.
    #     """
    #     pass