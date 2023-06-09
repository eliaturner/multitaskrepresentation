import matplotlib.pyplot as plt
import seaborn as sns

from analysis.analyzer import TaskFamilyAnalyzer

plt.rcParams['axes.facecolor'] = 'white'
from sklearn.decomposition import PCA

sns.set()
plt.rcParams['axes.facecolor'] = 'white'


class TasksPCA(TaskFamilyAnalyzer):
    name = 'kappa'

    def run(self):
        pca = PCA(2)
        pca.fit(self.all_points)
        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_subplot(projection=None)
        # ax.set_aspect('equal')
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xticks([])
        ax.set_yticks([])
        markers = ['o', 'v', 'x', 's', 'p', '.']
        for task_num, task in enumerate(self.TASKS):
            xn = self.return_points(task)
            colors = self.return_color(cmap='viridis')
            xn_2d = pca.transform(xn)
            ax.scatter(*xn_2d.transpose(), color=colors, marker=markers[task_num])

        self.save_plot(plt, 'pca', format='pdf')
