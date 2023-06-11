from analysis.tasks_pca import TasksPCA
from data.functions import X2, X2Rotate
from data.data_config import *
from data.data_generator import FamilyOfTasksGenerator
from model.model_wrapper import ModelWrapper
from model.pt_models import *

tasks = [X2(), X2Rotate()]

args_lists = [single_single_args, single_single_tonic_args, single_multi_args, single_multi_tonic_args, multi_single_args, multi_multi_args]
units = 100
inst = 5000
for args in args_lists:
    args['steps'] = 200
    kwargs = {'architecture_func': VanillaArchitecture,
              'units': units,
              'train_data': FamilyOfTasksGenerator(tasks, **args),
              'instance_range': [5000],
              'recurrent_bias': True,
              'readout_bias': True,
              'freeze_params': None,
              'weight_init_func': None}
    full_wrapper = ModelWrapper(**kwargs)
    full_wrapper.analyze([TasksPCA])