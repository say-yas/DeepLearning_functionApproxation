import torch.nn as nn
import torch
import numpy as np

class MLP_PINN(nn.Module):
    "Defines a standard fully-connected network in PyTorch"
    """
    Args:  
    num_input: int
        number of input features
    num_output: int
        number of output features
    num_hidden: int
        number of hidden units in the network
    num_layers: int
        number of hidden layers in the network 
    """
    
    def __init__(self, num_input, num_output, num_hidden, num_layers):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(num_input, num_hidden),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(num_hidden, num_hidden),
                            activation()]) for _ in range(num_layers-1)])
        self.fce = nn.Linear(num_hidden, num_output)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


class EarlyStopping_val_train:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True



class EarlyStopper:
    """Early stops the training if validation loss does not increase after a
    given patience.
    """
    def __init__(self, verbose=False, path='checkpoint.pt', patience=1):
        """Initialization.
        Args:
            verbose (bool, optional): Print additional information. Defaults to False.
            path (str, optional): Path where checkpoints should be saved. 
                Defaults to 'checkpoint.pt'.
            patience (int, optional): Number of epochs to wait for decreasing
                loss. If lossyracy does not increase, stop training early. 
                Defaults to 1.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.__early_stop = False
        self.val_loss_min = np.Inf
        self.path = path
                
    @property
    def early_stop(self):
        """True if early stopping criterion is reached.
        Returns:
            [bool]: True if early stopping criterion is reached.
        """
        return self.__early_stop
                
    def update(self, val_loss, model):
        """Call after one epoch of model training to update early stopper object.
        Args:
            val_loss (float): lossuracy on validation set
            model (nn.Module): torch model that is trained
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, val_loss)
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.__early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model, val_loss)
            self.counter = 0
            
    def save_checkpoint(self, model, val_loss):
        """Save model checkpoint.
        Args:
            model (nn.Module): Model of which parameters should be saved.
        """
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss        
        
    def load_checkpoint(self, model):
        """Load model from checkpoint.
        Args:
            model (nn.Module): Model that should be reset to parameters loaded
                from checkpoint.
        Returns:
            nn.Module: Model with parameters from checkpoint
        """
        model.load_state_dict(torch.load(self.path))
        return model