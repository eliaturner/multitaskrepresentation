import itertools

from matplotlib.path import Path
from matplotlib.patches import PathPatch
from analysis.analyzer import TaskFamilyLowRankAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg
import seaborn as sns
import sklearn.metrics
import contourpy
import torch
from tools.math_utils import solve_optimization
from matplotlib.gridspec import GridSpec

from analysis.analyzer import Analyzer, TaskFamilyAnalyzer, TaskFamilyLowRankAnalyzer, LowRankAnalyzer
# plt.rcParams['axes.facecolor'] = 'white'
from sklearn.decomposition import PCA
from analysis.task_family_aux import pc_dimension_coding_scores, plot_manifolds, diff_activity_single_neurons, task_points_dict_to_dPCA_1D, dpca_process_1d, dpca_process_2d, plot_manifolds_without_grid, map_points_to_values
from sklearn.manifold import MDS
from tools.math_utils import calc_normalized_q_value
from matplotlib import cm, colors
from scipy.optimize import minimize
from numpy.linalg import norm
from descartes import PolygonPatch
import numpy as np
sns.set()
# sns.set_palette("viridis")
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
plt.rcParams['axes.facecolor'] = 'white'


def points_to_attractor(n_points, contours, final_kappas):
    matrix = np.empty((n_points, n_points))

    for index in range(n_points**2):
        for i, contour in enumerate(contours):
            point = Point(*final_kappas[index])
            if contour.contains(point):
                matrix[index // n_points, index % n_points] = i
                break

    return matrix

def get_contours(U, V, Z):
    gen = contourpy.contour_generator(U, V, Z)
    contours = [Polygon(contour) for contour in (gen.lines(0.2))]
    return contours

def plot_contours(contours, matrix, eigvals):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    axes = [ax, ax]
    axes[0].set_xlabel(r'$\kappa_1$')
    axes[0].set_ylabel(r'$\kappa_2$')
    axes[1].set_xlabel(r'$\kappa_1$')
    axes[1].set_ylabel(r'$\kappa_2$')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    cmap = plt.cm.get_cmap('Set3', len(contours))
    # contours = [contours[i] for i in np.argsort([contour.centroid.x for contour in contours])]
    # contours = contours[np.argsort([contour.centroid for contour in contours])]
    for i, contour in enumerate(contours):
        color = cmap(i)
        if len(contours) == 9:
            # print(contour.centroid.x, contour.centroid.y)
            if (np.abs(contour.centroid.x) < 2 or np.abs(contour.centroid.y) < 2):
                color = 'lightgrey'
        patch = PathPatch(Path(contour.exterior.coords), color=color, alpha=0.5)
        axes[1].add_patch(patch)

    newax = fig.add_axes([0.8, 0.8, 0.2, 0.2], anchor='NE', zorder=1)
    newax2 = fig.add_axes([0.0, 0.8, 0.15, 0.15], anchor='NE', zorder=1)
    newax.imshow(matrix, cmap=cmap, interpolation='nearest', origin='lower', vmin=0, vmax=len(contours))
    newax.set_xlabel(r'$\kappa_1$')
    newax.set_ylabel(r'$\kappa_2$')
    newax.set_xticks([])
    newax.set_yticks([])
    newax2.scatter(eigvals.real, eigvals.imag, c='black', s=10)
    newax2.spines['right'].set_visible(False)
    newax2.spines['top'].set_visible(False)
    newax2.spines['left'].set_visible(True)
    newax2.spines['bottom'].set_visible(True)
    newax2.set_xlabel(r'$\Re(\lambda)$')
    newax2.set_xlim([-2, 2])
    newax2.set_ylim([-2, 2])
    newax2.set_xticks([])
    newax2.set_yticks([])
    # newax2.set_xticklabels([-1, 1])
    # newax2.set_yticklabels([-1, 1])
    newax2.set_ylabel(r'$\Im(\lambda)$')
    theta = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    newax2.plot(x, y, color='grey')
    # p = axes[0].imshow(matrix, cmap=cmap, interpolation='nearest', origin='lower', vmin=0, vmax=len(contours))
    return fig, axes



class BasinOfAttraction(LowRankAnalyzer):
    name = 'kappa'

    def run(self):
        n_points = 121
        U, V, Z, initial_states = self.rank_analyzer.kappa_UVZ(n_points)
        # p = plt.imshow(Z)
        # plt.colorbar(p)
        # plt.show()
        # return
        contours = get_contours(U, V, Z)
        TRAJECTORIES = self.model.run_system_from_inits(initial_states, 50)['state']
        final_kappas = [self.state_to_kappa(s) for s in TRAJECTORIES[:, -1]]
        matrix = points_to_attractor(n_points, contours, final_kappas)
        eigvals = np.linalg.eigvals(self.rank_analyzer.W_rec)
        eigvals = eigvals[np.argsort(np.abs(eigvals))]
        _, axes = plot_contours(contours, matrix, eigvals[-2:])
        max_k1 = self.rank_analyzer.max_kappa(0) - 2
        max_k2 = self.rank_analyzer.max_kappa(1) - 2
        axes[1].set_xlim(-max_k1, max_k1)
        axes[1].set_ylim(-max_k2, max_k2)

        inputs_kappa = self.rank_analyzer.project_inputs_to_kappa()
        # Plot the origin
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for task_num, kappas in enumerate(inputs_kappa):
            kappas = np.array(kappas)
            axes[1].annotate('', xy=kappas, xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=colors[task_num]))

        X = self.data_params.get_minimal_input()
        pred = self.model.predict(X)
        states = pred['state'][:,10:]
        # q_vals = calc_normalized_q_value(states)
        for index, state in enumerate(states):
            trajectory = np.vstack([self.state_to_kappa(s) for s in state])
            axes[1].plot(trajectory[:, 0], trajectory[:, 1], '->', color='black', zorder=1)

        self.save_plot(plt, 'basin', format='pdf')

