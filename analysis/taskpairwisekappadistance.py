
import numpy as np
from numpy.linalg import norm
from scipy.interpolate import CubicSpline

from analysis.analyzer import TaskFamilyLowRankAnalyzer


def interpolate_2d_curve(curve, steps=20):
    # Compute the distance along the curve (assumed to be approximately ordered)
    # This is our new 'x' value for the interpolation
    t = np.zeros(curve.shape[0])
    t[1:] = np.sqrt((np.diff(curve[:, 0]) ** 2) + (np.diff(curve[:, 1]) ** 2)).cumsum()

    # Create interpolation functions for x and y separately
    fx = CubicSpline(t, curve[:, 0])
    fy = CubicSpline(t, curve[:, 1])

    # Create new 't' values and compute the interpolated x and y values
    t_new = np.linspace(t.min(), t.max(), steps)
    x_new = fx(t_new)
    y_new = fy(t_new)

    new_curve = np.hstack([x_new[:, None], y_new[:, None]])

    return new_curve


def curve_distance(array1, array2):
    return min(norm(array1 - array2), norm(array1 - array2[::-1]), norm(array1 + array2), norm(array1 + array2[::-1]))


class TasksPairwiseKappaDistance(TaskFamilyLowRankAnalyzer):
    name = 'kappa'

    def run(self):
        kappas, _, _ = self.get_kappas()
        curves = [interpolate_2d_curve(kappas[task]) for task in self.TASKS]
        distance_list = []
        for task1 in range(self.n_tasks):
            for task2 in range(task1 + 1, self.n_tasks):
                distance = curve_distance(curves[task1], curves[task2])
                distance_list.append(distance)

        self.save_file(np.array(distance_list), 'kappa_distance_array')
