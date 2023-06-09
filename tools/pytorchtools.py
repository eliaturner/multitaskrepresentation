# import numpy as np
import torch
# from torch.utils.data import Dataset, TensorDataset
# from torch.utils.data import DataLoader
def get_device():
    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()
    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    return device
device = get_device()

# checkpoint_counter = 0
#
# def get_device():
#     # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
#     is_cuda = torch.cuda.is_available()
#     # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
#     if is_cuda:
#         device = torch.device("cuda:0")
#         print("GPU is available")
#     else:
#         device = torch.device("cpu")
#         print("GPU not available, CPU used")
#
#     return device
#
# device = get_device()


def np2torch(x, train=False):
    temp = torch.from_numpy(x).float()  # .to(pytorchtools.device)
    if train:
        temp = temp.to(device)

    return temp

def torch2np(x):
    return x.cpu().detach().numpy()
#

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, model, patience=2000, verbose=False, delta=1e-5, path='checkpoint.pt', minimal_loss=1e-5):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.model = model
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.opposite_counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss):

        score = -val_loss
        residue = 1

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                self.model.load_state_dict(torch.load(self.path + '/weights.pt'))
        else:
            self.best_score = score
            self.opposite_counter += 1
            #print('opposite counter', self.opposite_counter)
            if self.opposite_counter % residue == 0:
                global checkpoint_counter
                checkpoint_counter += 1
            self.save_checkpoint(val_loss)
            self.counter = 0
            if self.opposite_counter == 30:
                residue = 10

    def load_best(self):
        self.model.load_state_dict(torch.load(self.path + '/weights.pt'))

    def save_checkpoint(self, val_loss):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(self.model.state_dict(), f'{self.path}/weights.pt')
        torch.save(self.model.state_dict(), f'{self.path}/weights{checkpoint_counter}.pt')
        # if val_loss < self.minimal_loss:
        #     self.stop_training = self.early_stop = True
        #     self.counter = 0
        self.val_loss_min = val_loss
#
#
# def n2t(x, cpu=False):
#     if cpu:
#         return torch.from_numpy(x).to('cpu').float()
#     else:
#         return torch.from_numpy(x).to(device).float()
#
#
# def t2n(x):
#     return x.detach().cpu().numpy()
#
#

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader

checkpoint_counter = 0




#
class EarlyStoppingOLD:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=150, verbose=False, delta=1e-7, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.opposite_counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.stop_training = False

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                self.load_best(model)
        else:
            self.best_score = score
            self.opposite_counter += 1
            #print('opposite counter', self.opposite_counter)
            if self.opposite_counter % 1 == 0:
                global checkpoint_counter
                checkpoint_counter += 1
            self.save_checkpoint(val_loss, model)
            self.counter = 0


    def load_best(self, model):
        model.load_state_dict(torch.load(self.path + '.pt'))


    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path + '.pt')
        torch.save(model.state_dict(), self.path + str(checkpoint_counter) + '.pt')
        if val_loss < 5e-5:
            self.stop_training = self.early_stop = True
        self.val_loss_min = val_loss


def get_loaders(generator):
    device = get_device()
    x_train, y_train = generator.generate_train_data()
    x_train = torch.from_numpy(x_train).to(device)
    y_train = torch.from_numpy(y_train).to(device)
    train_loader = DataLoader(dataset=TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
    return train_loader
