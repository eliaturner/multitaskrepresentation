from analysis.kappaplane import KappaPlane
from model.model_wrapper import ModelWrapper
from model.pt_models import Rank2Architechture
from data.data_generator import FamilyOfTasksGenerator
from data.functions import *

TASKS = [X4Rotate(), X2Rotate(), XReverse(), X(), X2(), X4()]

kwargs = {'architecture_func':Rank2Architechture,
          'units': 100,
          'train_data':FamilyOfTasksGenerator(TASKS),
          'instance_range': range(2000, 2020),
}

full_wrapper = ModelWrapper(**kwargs)
full_wrapper.analyze([KappaPlane])
