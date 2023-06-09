from model.pt_models import Rank2Architechture
from data.custom_data_generator import ParallelFlipFlopGenerator
from analysis.basinofattraction import BasinOfAttraction
from model.model_wrapper import ModelWrapper
kwargs = {'architecture_func': Rank2Architechture,
          'units': 100,
          'train_data': ParallelFlipFlopGenerator(n_bits=2),
          'instance_range': [100],
          'recurrent_bias': True,
          'readout_bias': True,
          'freeze_params': None,
          'weight_init_func': None}

full_wrapper = ModelWrapper(**kwargs)
full_wrapper.get_analysis_checkpoints([BasinOfAttraction], checkpoints=range(0, 50, 10))
