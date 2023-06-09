from analysis.analyzer import Analyzer
import numpy as np


class VarianceBetweenVarianceWithin(Analyzer):
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
        # Calculate the variance within taskA_states
        variance_within_A = np.var(taskA_states)

        # Calculate the variance within taskB_states
        variance_within_B = np.var(taskB_states)

        # Concatenate taskA_states and taskB_states into a single array
        combined_states = np.concatenate((taskA_states, taskB_states))

        # Calculate the variance of the combined data
        variance_combined = np.var(combined_states)

        # Calculate the variance between the two sets
        variance_between = variance_combined - (variance_within_A + variance_within_B) / 2
        f_ratio = variance_between / (variance_within_A + variance_within_B)
        self.save_file(f_ratio, 'f_ratio')