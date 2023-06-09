from analysis.ff import FFPCA
from data.custom_data_generator import *
from model.model_wrapper import ModelWrapper
from model.pt_models import VanillaArchitecture

def get_tasks(n_bits):
    return [FlipFlopGenerator(n_bits=n_bits), OrthogonalFlipFlopGenerator(n_bits=n_bits),
            ParallelFlipFlopGenerator(n_bits=n_bits)]


kwargs = {'architecture_func': VanillaArchitecture,
          'units': 100,
          'instance_range': range(800, 810),
          'recurrent_bias': False,
          'readout_bias': True,
          'freeze_params': None,
          'weight_init_func': None}

bits_range = [2, 3]

n_networks = len(kwargs['instance_range'])

for n_bits in bits_range:
    v = get_tasks(n_bits)
    for i, train_data in enumerate(v):
        kwargs['train_data'] = train_data
        full_wrapper = ModelWrapper(**kwargs)
        full_wrapper.analyze([FFPCA]);
