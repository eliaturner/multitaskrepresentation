import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm

from analysis.analyzer import TaskFamilyLowRankAnalyzer

sns.set()
plt.rcParams['axes.facecolor'] = 'white'


class KappaPlane(TaskFamilyLowRankAnalyzer):
    name = 'kappa_plane'

    def run(self):
        KAPPAS, _, _ = self.get_kappas()
        U, V, Z, _ = self.rank_analyzer.kappa_UVZ(n_points=61)
        levels = [0.1, 0.25, 0.5, 1]
        fig, axes = plt.subplots(1, 1, figsize=(4, 4))
        ax = axes
        ax.contour(U, V, Z, levels=levels, filled=False, linewidths=1,
                   antialiased=False, cmap='binary_r')
        ax.axvline(0, color='lightgrey', zorder=0, linewidth=0.5)
        ax.axhline(0, color='lightgrey', zorder=0, linewidth=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r'$\kappa_1$')
        ax.set_ylabel(r'$\kappa_2$')
        # ax.set_xlim(left=0)
        colors = self.return_color(cmap=cm.viridis)
        markers = ['o', 'v', 'x', 's', 'p', '.']
        for task_num, task in enumerate(self.TASKS):
            kappas = KAPPAS[task]
            if np.sum(kappas[:, 0]) < 0:
                kappas = -kappas

            ax.scatter(kappas[:, 0], kappas[:, 1], color=colors, marker=markers[task_num])

        self.save_plot(plt)
