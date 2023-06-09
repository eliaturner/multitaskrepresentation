import os
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import pt_modules
from model.rnn_models import RNNModel
from model.trainer import PyTorchTrainer

from tools import training_utils, pytorchtools
from tools.pytorchtools import torch2np, np2torch
from tools.utils import load_pickle, dump_pickle
import matplotlib.pyplot as plt
import shutil
import os
import glob
#from ray import tune
import pandas as pd
import time

def maxnorm_loss(input, target):
    return torch.nn.MSELoss()(input, target)
    return 0.01*(torch.max(torch.abs(input-target)))**2 + torch.nn.MSELoss()(input, target)


def initialization1(weights, units):
    return nn.Parameter(torch.normal(0, 1 / (units ** 0.25), weights.shape))

def initialization2(weights, units):
    return nn.Parameter(weights/np.sqrt(units))


def smaller_output_weights(weights_dict):
    units = weights_dict['fc.weight'].shape[1]
    for k, weights in weights_dict.items():
        if 'fc' in k:
            weights_dict[k] = nn.Parameter(weights/np.sqrt(units))

    return weights_dict


def lowrank_weights(weights_dict):
    units = weights_dict['fc.weight'].shape[1]
    for k, weights in weights_dict.items():
        if 'fc' in k:
            weights_dict[k] = nn.Parameter(weights/np.sqrt(units))

    return weights_dict


def smaller_input_weights(weights_dict):
    units = weights_dict['fc.weight'].shape[1]
    for k, weights in weights_dict.items():
        if 'ih' in k:
            weights_dict[k] = nn.Parameter(weights/np.sqrt(units))

    return weights_dict

def larger_output_weights(weights_dict):
    units = weights_dict['fc.weight'].shape[1]
    for k, weights in weights_dict.items():
        if 'fc' in k:
            weights_dict[k] = nn.Parameter(weights * (units ** 0.25))

    return weights_dict


class PTModelArchitecture(RNNModel):
    def create_model(
            self,
            initial_state=None,
            return_output=True,
            return_sequences=True):
        if initial_state is None and self.initial_state is not None:
            initial_state = self.initial_state.to(pytorchtools.device)
        model = self.rnn_func(self.inputs, self.outputs, self.units, initial_state, nonlinearity=self.activation, recurrent_bias=self.recurrent_bias, readout_bias=self.readout_bias)
        return model

    def get_weights(self):
        weights = torch.load(self.model_dir + '/weights.pt', map_location='cpu')
        # weights = {key.replace('rnn.rnn.', 'rnn.'): val for key, val in weights.items()}
        return weights

    def rewrite_weights(self, weights):
        torch.save(weights, self.model_dir + '/weights.pt')

    def load_weights(self, weights=None):
        if weights is None:
            weights = self.get_weights()

        self.weights = weights

    def initial_weights(self):
        weights = torch.load(self.model_dir + '/initial_weights.pt', map_location='cpu')
        weights = {key.replace('rnn.rnn.', 'rnn.'): val for key, val in weights.items()}

        return weights

    def assign_weights(self, model, weights=None):
        if weights is None:
            # if os.path.isfile(self.model_dir + '/weights.pt'):
            #     weights = torch.load(self.model_dir + '/weights.pt', map_location='cpu')
            #     weights = {key.replace('rnn.rnn.', 'rnn.'): val for key, val in weights.items()}
            # else:
            #     print(self.model_dir)
            #     exit()
                # weights_tf = load_pickle(self.model_dir + '/weights.pkl')
                # weights = convert_tensorflow_to_pytorch_weights(weights_tf)
            weights = self.weights
        # elif type(weights) == dict:
        #     for k in weights.keys():
        #         if 'rnn' in k:
        #             weights[k] = weights[k][self.units:2*self.units]
        #     self.weights = weights
        #
        #     return
        if 'Wih_context.weight' in weights.keys():
            weights['Wih.weight'] = torch.cat([weights['Wih_context.weight'], weights['Wih_data.weight']], dim=1)
            del weights['Wih_context.weight']
            del weights['Wih_data.weight']
        model.load_state_dict(weights)

    def run_system_from_input(self, inputs):
        return self.predict(inputs)

    def get_instance_from_dir(self):
        return int(self.model_dir.split('i')[-1])
        print()

    def get_model(self):
        model = self.create_model(initial_state=None).cpu()
        self.assign_weights(model, self.weights)
        return model

    def run_system_from_inits(
            self,
            init_states,
            steps=None,
            input_value=0
    ):
        initial_states = np2torch(init_states)
        batch_size = init_states.shape[0]
        if type(input_value) == np.float64 or type(input_value) == int:
            x = input_value * np.ones((batch_size, steps, self.inputs))
        else:
            x = input_value

        #            x = np.concatenate([val*np.ones((batch_size, steps, 1)) for val in input_value], axis=-1)
        return self.predict(x, initial_states)

    def predict(self, x, initial_states=None, weights=None):
        steps = x.shape[1]
        model = self.create_model(initial_state=initial_states).cpu()
        self.assign_weights(model, weights)
        pred = model.forward_states(np2torch(x))
        output, state = torch2np(pred[0]), torch2np(pred[1])
        predictions = {'output': output.squeeze(),
                       'state': state[:,::self.num_layers]}
        if self.num_layers > 1:
            for i in range(1, self.num_layers):
                predictions[f'state{i}'] = state[:,i::self.num_layers]

        return predictions

    def train(self, x_train, y_train, x_val, y_val, params, weights=None, shuffle=True):
        epochs, batch_size, _, minimal_loss = params.epochs, params.batch_size, params.loss, params.minimal_loss
        x_train = np2torch(x_train, train=True)
        y_train = np2torch(y_train, train=True)
        train_loader = DataLoader(dataset=TensorDataset(x_train, y_train), batch_size=batch_size,
                                  shuffle=shuffle)
        trainer = PyTorchTrainer(self.model_dir, train_loader, epochs, minimal_loss, params.initial_lr)
        shutil.rmtree(self.model_dir)
        os.makedirs(self.model_dir)
        pytorchtools.checkpoint_counter = 0
        model = self.create_model().to(pytorchtools.device)
        if weights:
            self.assign_weights(model, weights)
        else:
            instance = self.get_instance_from_dir()
            self.weights_init_func(model, instance)

        ## TO REMOVE!!!!!
        # model.state_dict()['m'][:] *= 2
        # model.state_dict()['Who.weight'][:] /= 10
        # if 'rnn.weight_hh_l0' in model.state_dict().keys():
        #     model.state_dict()['rnn.weight_hh_l0'][:] = 0

        if self.freeze_weights is not None:
            for name, param in model.named_parameters():
                if name in self.freeze_weights:
                    param.requires_grad = False

        # # Data_weights = model.state_dict()['Wih_data.weight']
        # # TASKS = ['x', 'x_rev', 'x2', 'x2_rev', 'x2_rot', 'x4', 'x4_rot', 'x4_rev', 'tan', 'tanh']
        # for i in range(20):
        #     print()
        #     model = self.create_model().to(pytorchtools.device)
        #     W = model.state_dict()
        #     torch.save(W, f'initial_weights/N{self.units}/i{i}_vanilla.pt')
        # exit()
        trainer.train(model)
        return trainer.train_on


class Rank1Architechture(PTModelArchitecture):

    @property
    def rnn_func(self):
        return pt_modules.Rank1RNNV2

class Rank2Architechture(PTModelArchitecture):
    @property
    def rnn_func(self):
        return pt_modules.Rank2RNNV2

class Rank3Architechture(PTModelArchitecture):

    @property
    def rnn_func(self):
        return pt_modules.Rank3RNNV2

class Rank50Architechture(PTModelArchitecture):
    @property
    def rnn_func(self):
        return pt_modules.Rank50RNN

class FullRankArchitechture(PTModelArchitecture):
    @property
    def rnn_func(self):
        return pt_modules.FullRankRNN

class VanillaArchitecture(PTModelArchitecture):
    @property
    def rnn_type(self):
        return 'vanilla'

    @property
    def rnn_func(self):
        return pt_modules.Vanilla


class LSTMArchitecture(PTModelArchitecture):
    @property
    def rnn_type(self):
        return 'lstm'

    @property
    def rnn_func(self):
        return pt_modules.LSTM


class GRUArchitecture(PTModelArchitecture):
    @property
    def rnn_type(self):
        return 'gru'

    @property
    def rnn_func(self):
        return pt_modules.GRU


class GRUBiArchitecture(PTModelArchitecture):
    @property
    def rnn_type(self):
        return 'gru_bi'

    @property
    def rnn_func(self):
        return pt_modules.BiRNN


class GRUMultiArchitecture(PTModelArchitecture):
    @property
    def rnn_type(self):
        return 'gru_multi'

    @property
    def rnn_func(self):
        return pt_modules.MultiLayerRNN
