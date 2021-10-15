import os
import warnings
import tensorflow as tf
from sentience.framework.abstract.train import Trainer
from sentience.framework.abstract.state_config import state_config
from sentience.framework.abstract.data import Dataset
from tensorflow.keras import Model, callbacks, optimizers, losses

class TFTrainer(Trainer):
    """
    TensorFlow trainer state class. Trains a model and routinely saves the training state.
    """
    def __init__(self, tf_model: Model, 
                 tf_optimizer: optimizers.Optimizer, 
                 tf_loss: function,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 tf_callbacks: list, 
                 epochs: int=10,
                 batch_size: int=1) -> None:
        
        super().__init__()
        
        
        
        self.model = tf_model
        self.optimizer = tf_optimizer
        self.loss = tf_loss
        self.callbacks = callbacks.CallbackList(callbacks=tf_callbacks,
                                                add_history=False,
                                                add_progbar=True,
                                                model=self.model)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.epochs = epochs
        self.completed_epochs = 0
        self.batch_size = batch_size
    
    def train(self, state_path, save_state_interval, resume=False, model_checkpoint_path=None) -> None:
        """
        TBD
        """
        state_path = self.check_state_file_path(state_path)
        
        @tf.function
        def train_step(batch_inp, batch_labels):
            with tf.GradientTape() as grad:
                batch_out = self.model(batch_inp)
                loss_value = self.loss(batch_labels, batch_out)
            
            grad_out = grad.gradient(loss_value, self.model.trainable_variables())
            self.optimizer.apply_gradients(zip(grad_out, self.model.trainable_variables()))
            return loss_value
        
        for epoch in range(1, self.epochs + 1):
            print(f"Epoch {epoch}/{self.epochs}")
            