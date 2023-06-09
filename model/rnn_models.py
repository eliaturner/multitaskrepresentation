from abc import abstractmethod, ABC

PULSE = 10


class RNNModel(ABC):
    def __init__(self, units=70, inputs=1, outputs=1, activation='tanh', recurrent_bias=False, readout_bias=False, initial_state=None, num_layers=1, weight_init_func=None, freeze_weights=None):
        self.activation = activation
        self.inputs = inputs
        self.outputs = outputs
        self.units = units
        self.model_dir = None
        self.weights = None
        self.recurrent_bias = recurrent_bias
        self.readout_bias = readout_bias
        self.initial_state = initial_state
        self.num_layers = num_layers
        self.weights_init_func = weight_init_func
        self.freeze_weights = freeze_weights

    @abstractmethod
    def create_model(
            self,
            initial_state=None,
            return_output=True,
            return_sequences=True):
        pass

    def set_model_dir(self, model_dir):
        self.model_dir = model_dir

    @property
    @abstractmethod
    def rnn_func(self):
        pass

    @abstractmethod
    def load_weights(self):
        pass

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def assign_weights(self, model, weights=None):
        pass

    @abstractmethod
    def train(self, x_train, y_train, x_val, y_val, params, weight_init_func, weights=None, shuffle=True):
        pass

    @property
    def name(self):
        name = self.rnn_func.rnn_type + '_' + str(self.units)
        if self.num_layers > 1:
            name += f'_{self.num_layers}'

        if self.activation != 'tanh':
            name += f'_{self.activation}'

        return name

