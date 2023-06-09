from abc import ABC, abstractmethod
from pathlib import Path
from enum import IntEnum

import numpy as np
import seaborn as sns
from data.functions import Function1D, Function2D
from tools.math_utils import random_noisy_states
from tools.utils import dump_pickle, load_pickle
sns.set()

DATA_PATH = 'data'
MODEL_PATH = 'models'


class Index(IntEnum):
    READY = 1,
    READY_SET = 2,
    SET = 3,
    SET_GO = 4,
    GO = 5,
    END = 6


# TODO: Edit
class Analyzer(ABC):
    results_path = 'analysis_results'

    def __init__(self, model, data_params, model_name, instance, outputs, states, checkpoint=None):
        self.model = model
        self.X, self.Y = data_params.get_data()
        self.data_params = data_params
        self.model_name = model_name
        self.instance = instance
        self.checkpoint = checkpoint
        self.data_params.get_data
        # self.X = self.data_params.x
        # self.Y = self.data_params.y
        self.outputs, self.states = outputs, states
        if 'vanilla' in model_name:
            self.rank_analyzer = VanillaAnalyzer(model)

    def save_plot(self, plt, desc='', format='png'):
        # save analysis-major
        path_analysis = f'analysis_results/{self.name}/'
        analysis_model_name = f'{desc}_{self.model_name}_i{self.instance}'
        if self.checkpoint is not None:
            path_analysis += 'valid_checkpoints/'
            analysis_model_name += f'_chkpt{self.checkpoint}'

        Path(path_analysis).mkdir(parents=True, exist_ok=True)
        plt.savefig(path_analysis + analysis_model_name + '.' + format, bbox_inches='tight', dpi=1000)
        plt.close()
        return
        # save model major
        path_model = f'analysis_results/models/{self.model_name}/'
        if self.checkpoint is None:
            analysis_full_name = f'i{self.instance}_{self.name}_{desc}.png'
        else:
            path_model += 'checkpoints/'
            analysis_full_name = f'i{self.instance}_chkpt{self.checkpoint}_{self.name}_{desc}.png'

        Path(path_model).mkdir(parents=True, exist_ok=True)
        plt.savefig(path_model + analysis_full_name, bbox_inches='tight')
        plt.close()

    def save_file(self, file, desc):
        if self.checkpoint is None:
            path = f'models/{self.model_name}/i{self.instance}/{desc}.pkl'
        else:
            path = f'models/{self.model_name}/i{self.instance}/valid_checkpoints/chkpt{self.checkpoint}_{desc}.pkl'
            Path(f'models/{self.model_name}/i{self.instance}/valid_checkpoints/').mkdir(parents=True, exist_ok=True)

        dump_pickle(path, file)

    def load_file(self, desc):
        path = f'models/{self.model_name}/i{self.instance}/{desc}.pkl'
        return load_pickle(path)

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def run(self):
        pass


from analysis.task_family_aux import extract_lists_from_trials, points_by_task_1d, points_by_task_2d

class VanillaAnalyzer:
    def __init__(self, model):

        self.W_rec = model.weights['rnn.weight_hh_l0'].numpy()
        u, s, v = np.linalg.svd(self.W_rec)
        self.s = s
        self.Who = model.weights['fc.weight'].squeeze().numpy()
        self.Wih = model.weights['rnn.weight_ih_l0'].squeeze().numpy()
        if 'rnn.bias_ih_l0' in model.weights:
            self.bh = model.weights['rnn.bias_ih_l0'].squeeze().numpy() + model.weights['rnn.bias_hh_l0'].squeeze().numpy()
        else:
            self.bh = np.zeros(model.units)

    def get_approx_fp(self):
        I = np.eye(self.s.shape[0])
        M1 = (I - (1 - np.diag(np.tanh(self.bh)**2)) @ self.W_rec)
        x = np.linalg.inv(M1) @ np.tanh(self.bh)
        return x






class RankAnalyzer:
    def __init__(self, model):
        self.m = model.weights['m'].squeeze().numpy()
        self.n = model.weights['n'].squeeze().numpy()
        W_rec = self.m @ self.n.transpose()
        self.W_rec = W_rec
        u, s, v = np.linalg.svd(W_rec)
        rank = model.rnn_func.rank
        self.s = s
        self.m = s[:rank] * u[:, :rank]
        self.n = v[:rank, :].transpose()
        self.W_rec = W_rec
        self.Who = model.weights['Who.weight'].squeeze().numpy()
        self.Wih = model.weights['Wih.weight'].squeeze().numpy()
        if 'Wih.bias' in model.weights:
            self.bh = model.weights['Wih.bias'].squeeze().numpy()
        else:
            self.bh = np.zeros(model.units)



    def state_to_kappa(self, s):
        return self.n.transpose() @ s

    def max_kappa(self, num):
        return np.dot(self.n[:, num], np.sign(self.n[:, num]))

    def kappa_to_state(self, kappa):
        return np.tanh(self.m @ kappa + self.bh)

    def next_kappa(self, kappa):
        return self.state_to_kappa(self.kappa_to_state(kappa))

    def generate_grid(self, U, V):
        next_states = []
        Z = np.zeros(U.shape)
        for row in range(Z.shape[0]):
            for col in range(Z.shape[1]):
                kappa = np.array([U[row, col], V[row, col]])
                next_state = self.kappa_to_state(kappa)
                next_states.append(next_state)
                Z[row, col] = np.linalg.norm(kappa - self.state_to_kappa(next_state))

        return Z, np.vstack(next_states)

    def get_kappas_range(self, dim, n_points=100):
        max_k = self.max_kappa(dim) - 2
        # if dim == 0:
        #     max_k = max_k - 5
        # else:
        #     max_k = max_k - 4
        return np.linspace(-max_k, max_k, n_points)

    def kappa_UVZ(self, n_points = 100):
        kappas1_range = self.get_kappas_range(0, n_points)
        kappas2_range = self.get_kappas_range(1, n_points)
        U, V = np.meshgrid(kappas1_range, kappas2_range)
        Z, next_states = self.generate_grid(U, V)
        return U, V, Z, next_states

    def project_outputs_to_kappa(self):
        return [self.n.transpose() @ v for v in self.Who]
        from tools.math_utils import solve_optimization
        return [solve_optimization(self.m, v)[0] for v in self.Who]

    def project_inputs_to_kappa(self):
        return [self.n.transpose() @ np.tanh(v + self.bh) for v in self.Wih.transpose()]
        from tools.math_utils import solve_optimization
        return [solve_optimization(self.m, v)[0] for v in self.Wih.transpose()]





class TaskFamilyAnalyzer(Analyzer):
    def __init__(self, model, data_params, model_name, inst, outputs, states, checkpoints=None):
        super().__init__(model, data_params, model_name, inst, outputs, states, checkpoints)
        self.n_tasks = len(self.data_params.tasks)
        self.TASKS = np.copy(self.data_params.tasks)
        X = self.extract_input()
        self.points_dict, self.values_dict, times_dict, speeds = {}, {}, {}, {}
        self.all_points_dict = {}
        for i, task in enumerate(self.TASKS):
            D = extract_lists_from_trials(X, self.states[i::self.n_tasks], task, self.data_params.max_delay)
            self.points_dict[task] = D['points']
            self.values_dict[task] = D['output']
            self.all_points_dict[task] = D['all_points']
            self.xn = D['xn']
            self.xnm1 = D['xnm1']
            speeds[task] = D['speed']

        self.all_points = np.vstack(list(self.points_dict.values()))
        is1D = [issubclass(type(task), Function1D) for task in self.TASKS]
        is2D = [issubclass(type(task), Function2D) for task in self.TASKS]
        self.task_dimensionality = 1 if np.all(is1D) else 2 if np.all(is2D) else None
        # return
        if self.task_dimensionality == 1:
            self.task_points_dict, self.points_dpca, self.task_points_var_dict = points_by_task_1d(self.points_dict, self.TASKS, self.model.units, self.data_params.n_values,
                                                              self.data_params.validation_range, self.xn)

        else:
            # self.task_points_dict, self.task_points_var_dict = points_by_task_2d(self.points_dict, self.TASKS,
            #                                                   self.data_params.validation_range,
            #                                                   self.xn, self.xnm1)
            # self.points_dpca = task_points_dict_to_dPCA_2D(self.task_points_dict, self.TASKS,  self.data_params.validation_range)
            self.task_points_dict, self.points_dpca, self.task_points_var_dict = points_by_task_2d(self.points_dict, self.TASKS, self.model.units, self.data_params.n_values,
                                                              self.data_params.validation_range, self.xn, self.xnm1)


    def extract_input(self):
        if self.data_params.input_type == 'multi':
            X = self.X[::len(self.data_params.tasks), :, 0]
        else:
            X = self.X[::len(self.data_params.tasks), :, -1]

        return X


    def return_color(self, an=None, cmap='jet'):
        if an is None:
            an = self.data_params.validation_range
        from matplotlib import cm, colors
        norm = colors.Normalize(vmin=self.data_params.vmin, vmax=self.data_params.vmax, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        if self.task_dimensionality == 2:
            return [mapper.to_rgba(v) for v in an for v1 in an]

        return [mapper.to_rgba(v) for v in an]

    def return_output_color(self, task, an=None, cmap='rocket'):
        if an is None:
            an = self.data_params.validation_range
        an = [task.value(xn=v) for v in an]
        from matplotlib import cm, colors
        norm = colors.Normalize(vmin=self.data_params.vmin, vmax=self.data_params.vmax, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        if self.task_dimensionality == 2:
            return [mapper.to_rgba(v) for v in an for v1 in an]

        return [mapper.to_rgba(v) for v in an]
    def return_color_dict(self, cmap='rocket'):
        an = self.data_params.validation_range
        from matplotlib import cm, colors
        norm = colors.Normalize(vmin=self.data_params.vmin, vmax=self.data_params.vmax, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        return {v:mapper.to_rgba(v) for v in an}

    def return_points(self, task):
        return np.vstack(list(self.task_points_dict[task].values()))

class LowRankAnalyzer(Analyzer):
    def __init__(self, model, data_params, model_name, inst, outputs, states, checkpoints=None):
        super().__init__(model, data_params, model_name, inst, outputs, states, checkpoints)
        self.rank_analyzer = RankAnalyzer(model)

    def state_to_kappa(self, s):
        return self.rank_analyzer.state_to_kappa(s)


class TaskFamilyLowRankAnalyzer(TaskFamilyAnalyzer):

    def __init__(self, model, data_params, model_name, inst, outputs, states, checkpoints=None):
        super().__init__(model, data_params, model_name, inst, outputs, states, checkpoints)
        self.rank_analyzer = RankAnalyzer(model)

    def state_to_kappa(self, s):
        return self.rank_analyzer.state_to_kappa(s)

    def get_kappas(self):
        KAPPAS = {}
        for task in self.TASKS:
            an, xn = list(zip(*self.task_points_dict[task].items()))
            KAPPAS[task] = np.vstack([self.state_to_kappa(s) for s in xn])

        all_kappas = [e for l in (list(KAPPAS.values())) for e in l]
        all_kappas = [k[0] for k in all_kappas] + [k[1] for k in all_kappas]

        return KAPPAS, min(all_kappas), max(all_kappas)



