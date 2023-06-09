from abc import abstractmethod

import torch
import torch.nn as nn
import torch.autograd.profiler as profiler


# class LowRank(nn.RNN):

class RankRNNBase(nn.Module):
    def __init__(self, input_size,
                 output_size,
                 hidden_size,
                 initial_state,
                 recurrent_bias=False,
                 readout_bias=False,
                 num_layers=1,
                 nonlinearity='tanh'):
        super(RankRNNBase, self).__init__()
        self.initial_state = initial_state
        if self.initial_state is not None:
            self.initial_state = self.initial_state.unsqueeze(0)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.nonlinearity = nonlinearity
        # self.Wih_context = nn.Linear(input_size - 1, hidden_size, recurrent_bias=False)
        self.Wih = nn.Linear(input_size, hidden_size, bias=recurrent_bias)
        self.Who = nn.Linear(hidden_size, output_size, bias=readout_bias)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = 'cpu'

    def forward(self, x, return_dynamics=False):
        alpha = 0.2
        seq_len = x.shape[1]
        batch_size = x.shape[0]
        if return_dynamics:
            trajectories = torch.zeros(batch_size, seq_len, self.hidden_size, device=self.device)

        if self.nonlinearity == 'tanh':
            nonlinearity_func = torch.tanh
        else:
            nonlinearity_func = torch.relu

        # simulation loop
        if self.initial_state is not None:
            h = self.initial_state
        else:
            h = torch.zeros(self.hidden_size, device=self.device)
        # u_data = self.Wih_data(x[:,:,-1:])
        # u_context = self.Wih_context(x[:,:,:-1])
        u = self.Wih(x)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.device)
        # with profiler.profile(use_cuda=True, record_shapes=True) as prof:
        prod = self.get_rec_matrix()
        for i in range(seq_len):
            r = u[:,i] + h.matmul(prod)
            if self.training:
                r = r + torch.randn_like(r) * 0.01
            r = nonlinearity_func(r)
            h = (1 - alpha)*h + alpha*r
            output[:, i, :] = self.Who(h)
            if return_dynamics:
                trajectories[:, i, :] = h

        if return_dynamics:
            return output, trajectories
        return output

    def forward_states(self, x):
        return self.forward(x, return_dynamics=True)

    def get_rec_matrix(self):
        pass

class LowRankRNN(RankRNNBase):
    def __init__(self, input_size,
                 output_size,
                 hidden_size,
                 initial_state,
                 bias=False,
                 num_layers=1):
        super(LowRankRNN, self).__init__(input_size,
                 output_size,
                 hidden_size,
                 initial_state,
                 bias=False,
                 num_layers=1)
        self.m = nn.Parameter(torch.Tensor(hidden_size, type(self).rank))
        self.n = nn.Parameter(torch.Tensor(hidden_size, type(self).rank))
        with torch.no_grad():
            self.m.normal_(std = 1/((self.hidden_size) ** 0.5))
            self.n.normal_(std = 1/((self.hidden_size) ** 0.5))

        # self.Wih_data.weight.requires_grad = False
        # self.Wih_context.weight.requires_grad = False

    def get_rec_matrix(self):
        return self.n.matmul(self.m.t())


class LowRankRNNV2(RankRNNBase):
    def __init__(self, input_size,
                 output_size,
                 hidden_size,
                 initial_state,
                 recurrent_bias=False,
                 readout_bias=False,
                 num_layers=1,
                 nonlinearity='tanh'):
        super(LowRankRNNV2, self).__init__(input_size,
                 output_size,
                 hidden_size,
                 initial_state,
                 recurrent_bias=recurrent_bias,
                 readout_bias=readout_bias,
                 num_layers=1)
        self.m = nn.Parameter(torch.Tensor(hidden_size, type(self).rank))
        self.n = nn.Parameter(torch.Tensor(hidden_size, type(self).rank))
        with torch.no_grad():
            k = 1/((self.hidden_size) ** 0.5)
            self.m.uniform_(-k, k)
            self.n.uniform_(-k, k)
            # self.Who.weight.uniform_(-k, k)
            # self.Wih.weight.uniform_(-k, k)
            if recurrent_bias:
                self.Wih.bias.fill_(0)

        # self.Wih_data.weight.requires_grad = False
        # self.Wih_context.weight.requires_grad = False

    def get_rec_matrix(self):
        return self.n.matmul(self.m.t())

    def forward(self, x, return_dynamics=False):
        seq_len = x.shape[1]
        batch_size = x.shape[0]
        if return_dynamics:
            trajectories = torch.zeros(batch_size, seq_len, self.hidden_size, device=self.device)

        # simulation loop
        if self.initial_state is not None:
            h = self.initial_state
        else:
            h = torch.zeros(self.hidden_size, device=self.device)

        u = self.Wih(x)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.device)
        prod = self.get_rec_matrix()
        for i in range(seq_len):
            r = u[:,i] + h.matmul(prod)
            if self.training:
                r = r + torch.randn_like(r) * 0.00001
            h = torch.tanh(r)
            output[:, i, :] = self.Who(h)
            if return_dynamics:
                trajectories[:, i, :] = h

        if return_dynamics:
            return output, trajectories
        return output


class FullRankRNN(RankRNNBase):
    rnn_type = 'fullrank'

    def __init__(self, input_size,
                 output_size,
                 hidden_size,
                 initial_state,
                 bias=False,
                 num_layers=1):
        super(FullRankRNN, self).__init__(input_size,
                 output_size,
                 hidden_size,
                 initial_state,
                 bias=False,
                 num_layers=1)

        self.W_rec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        with torch.no_grad():
            # self.W_rec.normal_(std=1/hidden_size)
            self.W_rec.fill_(0)


    def get_rec_matrix(self):
        return self.W_rec


# class Rank1RNN(LowRankRNN):
#     rank = 1
#     rnn_type = 'rank1'
#
# class Rank2RNN(LowRankRNN):
#     rank = 2
#     rnn_type = 'rank2'
#
# class Rank3RNN(LowRankRNN):
#     rank = 3
#     rnn_type = 'rank3'


class Rank1RNNV2(LowRankRNNV2):
    rank = 1
    rnn_type = 'rank1'

class Rank2RNNV2(LowRankRNNV2):
    rank = 2
    rnn_type = 'rank2'

class Rank3RNNV2(LowRankRNNV2):
    rank = 3
    rnn_type = 'rank3'

class Rank50RNN(LowRankRNN):
    rank = 50
    rnn_type = 'rank50'

class RNN(nn.Module):
    def __init__(self, input_size,
                 output_size,
                 hidden_size,
                 initial_state,
                 recurrent_bias=False,
                 readout_bias=False,
                 num_layers=1,
                 nonlinearity='tanh'):
        super(RNN, self).__init__()
        self.initial_state = initial_state
        if self.initial_state is not None:
            self.initial_state = self.initial_state.unsqueeze(0)
        self.rnn = self.rnn_class(input_size, hidden_size, batch_first=True, bias=recurrent_bias, num_layers=num_layers)
        # self.rnn.weight_ih_l0.requires_grad = False
        # self.rnn.bias_ih_l0.requires_grad = False
        self.fc = nn.Linear(hidden_size, output_size, bias=readout_bias)


    @property
    @abstractmethod
    def rnn_class(self):
        pass

    def forward(self, x):
        # if self.return_states:
        #     return self.forward_states(x)
        # with profiler.profile(use_cuda=True, record_shapes=True) as prof:

        rnn_out, _ = self.rnn(x.float(), self.initial_state)
        readout = self.fc(rnn_out)
        # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10)); exit()
        return readout

    @abstractmethod
    def forward_states(self, x):
        pass


class Vanilla(RNN):
    rnn_type = 'vanilla'

    @property
    def rnn_class(self):
        return nn.RNN

    def forward_states(self, x):
        rnn_out, _ = self.rnn(x.float(), self.initial_state)
        readout = self.fc(rnn_out)
        return readout, rnn_out


class GRU(RNN):
    rnn_type = 'gru'

    @property
    def rnn_class(self):
        return nn.GRU

    def forward_states(self, x):
        rnn_out, _ = self.rnn(x.float(), self.initial_state)
            # rnn_out, _ = self.rnn[0](x.float(), self.initial_state[0])
            # for i in range(1, len(self.hidden_size)):
            #     rnn_out, _ = self.rnn[0](rnn_out, self.initial_state[i])
        readout = self.fc(rnn_out)
        return readout, rnn_out




class LSTM(RNN):
    rnn_type = 'lstm'

    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__(*args, **kwargs)
        if self.initial_state is not None:
            self.initial_state = torch.split(self.initial_state, len(self.initial_state[-1])//2, dim=-1)

    def forward_states(self, x):
        # rnn_out, hidden = self.rnn(x.float(), self.initial_state)
        # readout_origin = self.fc(rnn_out)

        hidden = self.initial_state
        hs = []
        cs = []
        readouts = []
        steps = x.shape[1]

        for step in range(steps):
            temp, hidden = self.rnn(x[:, step:step + 1], hidden)
            cs.append(hidden[1].detach())
            hs.append(hidden[0].detach())
            readouts.append(self.fc(temp).detach())

        #cs = np.concatenate(cs)
        cs = torch.cat(cs).transpose(0, 1)
        hs = torch.cat(hs).transpose(0, 1)
        readouts = torch.cat(readouts, 1)
        hidden = torch.cat((hs, cs), dim=-1)

        return readouts, hidden


class BiRNN(nn.Module):
    def __init__(self, input_size,
                 output_size,
                 hidden_size,
                 initial_state):
        super(BiRNN, self).__init__()
        self.initial_state = initial_state
        self.rnn_left = nn.GRU(input_size, hidden_size, batch_first=True, bias=True)
        self.rnn_right = nn.GRU(input_size, hidden_size, batch_first=True, bias=True)
        self.fc = nn.Linear(2*hidden_size, output_size, bias=True)

    def forward(self, x):
        # if self.return_states:
        #     return self.forward_states(x)
        rnn_out_left, _ = self.rnn_left(x.float())
        rnn_out_right, _ = self.rnn_right(x.float())
        readout = self.fc(torch.cat([rnn_out_left, rnn_out_right], axis=-1))
        return readout

    def forward_states(self, x):
        rnn_out_left, _ = self.rnn_left(x.float())
        rnn_out_right, _ = self.rnn_right(x.float())
        readout = self.fc(torch.cat([rnn_out_left, rnn_out_right], axis=-1))
        return readout, rnn_out_left, rnn_out_right


class MultiLayerRNN(nn.Module):
    def __init__(self, input_size,
                 output_size,
                 hidden_size,
                 initial_state,
                 num_layers):
        super(MultiLayerRNN, self).__init__()
        self.initial_state = initial_state
        self.rnns = [nn.GRU(input_size, hidden_size[0], batch_first=True, bias=True, num_layers=num_layers[0])]
        for i in range(1, len(hidden_size)):
            self.rnns.append(nn.GRU(hidden_size[i-1], hidden_size[i], batch_first=True, bias=True, num_layers=num_layers[i]))

        self.fc = nn.Linear(hidden_size[-1], output_size, bias=True)

    def forward(self, x):
        rnn_out, _ = self.rnns[0](x.float(), self.initial_state[0])
        for i in range(1, len(self.hidden_size)):
            rnn_out, _ = self.rnn[0](rnn_out, self.initial_state[i])

        readout = self.fc(rnn_out)
        return readout

    def forward_states(self, x):
        hs = []
        rnn_out, _ = self.rnns[0](x.float(), self.initial_state[0])
        hs.append(rnn_out)
        for i in range(1, len(self.hidden_size)):
            rnn_out, _ = self.rnn[i](rnn_out, self.initial_state[i])
            hs.append(rnn_out)

        readout = self.fc(rnn_out)
        return readout


