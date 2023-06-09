import matplotlib.pyplot as plt
# plt.rcParams['axes.facecolor'] = 'white'
import numpy as np
import seaborn as sns

from analysis.analyzer import Analyzer

sns.set()
# sns.set_palette("viridis")
plt.rcParams['axes.facecolor'] = 'white'

class LinearSeparability(Analyzer):
    name = 'kappa'

    def run(self):
        n_points = 10
        X = np.zeros((2*n_points, 60, 2))
        values_range = np.linspace(self.data_params.vmin, self.data_params.vmax, n_points)
        for i, value in enumerate(values_range):
            X[i, :5, 0] = value
            X[i+n_points, :5, 1] = value

        states = self.model.predict(X)['state'][:, -40:]
        taskA_states = states[:n_points].reshape(-1, self.states.shape[-1])
        taskB_states = states[n_points:].reshape(-1, self.states.shape[-1])
        from tools.math_utils import svm_if_separable
        separability = svm_if_separable(taskA_states, taskB_states)
        self.save_file(separability, 'separability')
        # print(separability)
        return 1 if separability > 0 else 0