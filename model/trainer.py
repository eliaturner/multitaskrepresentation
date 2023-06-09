import torch
from tools import pytorchtools, training_utils
from torch.optim import Adam, lr_scheduler, SGD
from tools.utils import load_pickle, dump_pickle
import matplotlib.pyplot as plt
import numpy as np

def custom_loss(target, predict, gpu=True):
    temp = torch.isnan(target)
    temp = torch.logical_not(temp)
    # mask = temp.clone().detach().type(torch.FloatTensor).requires_grad_(True)
    if gpu and torch.cuda.is_available():
        mask = torch.tensor(temp, dtype=torch.float64, requires_grad=True)
    else:
        mask = temp.clone().detach().type(torch.FloatTensor).requires_grad_(True)

    if predict.ndim == 2:
        predict = predict[:,:,None]

    # loss_tensor = (mask * (target - predict)).pow(2).mean(dim=-1)
    # # Account for different number of masked values per trial
    # loss_by_trial = loss_tensor.sum(dim=-1) / mask[:, :, 0].sum(dim=-1)
    # return loss_by_trial.mean()


    masked_diff = (predict - target) * mask
    masked_diff[masked_diff != masked_diff] = 0
    # masked_diff = torch.nan_to_num((predict - target) * mask)
    out = torch.sum(masked_diff ** 2.0) / torch.sum(mask)
    return out

def custom_loss2(y, output):
    # Assuming output and y have the same shape
    mask = ~torch.isnan(y)
    diff = (output - y) ** 2
    masked_diff = torch.where(mask, diff, torch.zeros_like(diff))
    loss = masked_diff.sum() / mask.sum()
    return loss


def train_epoch(train_loader, train_step):
    losses = []
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(pytorchtools.device).float()
        y_batch = y_batch.to(pytorchtools.device).float()
        loss = train_step(x_batch, y_batch)
        losses.append(loss)


    return np.average(losses)


class PyTorchTrainer:
    def __init__(self, model_dir, train_loader, epochs, minimal_loss, initial_lr):
        # self.model = model
        self.model_dir = model_dir
        self.train_loader = train_loader
        self.epochs = epochs
        self.minimal_loss = minimal_loss
        # if 'gru' in model_dir:
        #     self.minimal_loss = minimal_loss
        # else:
        #     self.minimal_loss = 8e-5
        self.train_on = True
        self.initial_lr = initial_lr

    def log_loss(self, epoch, curr_loss):
        if epoch % 5 == 0:
            print('Epoch: {}/{}.............'.format(epoch, self.epochs), end=' ')
            print("Loss: {:e}, {:.5f}".format(curr_loss, curr_loss), end=' ')
            print()

    def check_epoch(self, early_stopping, epoch, lr):
        if (-early_stopping.best_score > 3e-2 and epoch > 2000) or (-early_stopping.best_score > 2e-2 and epoch > 2000):
            print("EPOCH - Early stopping, epoch {}, score={:e}".format(epoch,
                                                                early_stopping.best_score))
            return True
        if early_stopping.early_stop:
            print("Early stopping, epoch {}, score={:e}".format(epoch,
                                                                early_stopping.best_score))
            return True

        if lr < 1e-6:
            print("LR - Early stopping, epoch {}, score={:e}".format(epoch,
                                                                early_stopping.best_score))
            return True

        if -early_stopping.best_score < self.minimal_loss:
            print('Minimal loss achieved! finish training')
            self.train_on = False
            return True

        return False

    def check_training(self, loss, lr):
        if lr <= 0.0001 and loss >= 1e-2:
            print('Train Again')
            print(lr, loss, 1e-2)
            return True

        return not self.train_on

    def grid_search(self, model):
        pass

    def train(self, model):
        torch.save(model.state_dict(), self.model_dir + '/initial_weights.pt')
        cooldown = 20
        early_stopping = pytorchtools.EarlyStopping(model, path=self.model_dir)
        epoch = 0
        optimizer = Adam(model.parameters(), lr=self.initial_lr)
        # optimizer = SGD(model.parameters(), lr=self.initial_lr)
        weight_history = []
        train_step = training_utils.make_train_step(model, custom_loss, optimizer, weight_history)
        average_losses = []
        early_stopping = pytorchtools.EarlyStopping(model, path=self.model_dir)

        for epoch in range(1, 3000):
            # start = time.time()
            curr_loss = train_epoch(self.train_loader, train_step)

            average_losses.append(curr_loss)
            self.log_loss(epoch, curr_loss)
            early_stopping(curr_loss)

            if self.check_epoch(early_stopping, epoch, optimizer.param_groups[0]['lr']):
                model.load_state_dict(torch.load(self.model_dir + '/weights.pt'))
                self.train_on = False
                if min(average_losses) > 0.001:
                    self.train_on = True
                return

        print('STEP1 - reload best', -early_stopping.best_score)
        early_stopping.load_best()
        train_step = training_utils.make_train_step(model, custom_loss, optimizer)
        optimizer = Adam(model.parameters(), lr=self.initial_lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=150, verbose=True, cooldown=cooldown, factor=0.8, threshold=1e-7)
        while True:
        # for epoch in range(1, 2*self.epochs + 1):
        # for epoch in range(1, 5):
            epoch += 1
            curr_loss = train_epoch(self.train_loader, train_step)
            average_losses.append(curr_loss)
            scheduler.step(curr_loss)
            self.log_loss(epoch, curr_loss)
            early_stopping(curr_loss)
            if epoch > 1 and scheduler.cooldown_counter == cooldown:
                print('reload best')
                early_stopping.load_best()
                epoch = 0
            elif epoch > 1000 and min(average_losses[-1000:]) > 3e-2:
                early_stopping.load_best()

            if self.check_epoch(early_stopping, epoch, optimizer.param_groups[0]['lr']):
                    break

        model.load_state_dict(torch.load(self.model_dir + '/weights.pt'))
        self.train_on = False
        if min(average_losses) > 0.001:
            print('Curr vs. min avg loss', curr_loss, min(average_losses))
            self.train_on = True