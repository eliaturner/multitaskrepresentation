import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.manifold import MDS
from sklearn.preprocessing import PolynomialFeatures
from collections import OrderedDict
from data.functions import Function1D, Function2D

from tools.math_utils import calc_normalized_q_value

plt.rcParams['axes.facecolor'] = 'white'


from sklearn.decomposition import PCA
import matplotlib.cm as cm

sns.set()
plt.rcParams['axes.facecolor'] = 'white'
from collections import defaultdict
from sklearn.model_selection import train_test_split
from scipy.linalg import orthogonal_procrustes
from scipy.linalg import schur


def pairwise_distance(x1, x2):
    matrix = np.zeros((len(x1), len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            matrix[i, j] = np.linalg.norm(x1[i] - x2[j])

    return matrix

def tasks_distance(task1, task2):
    values = np.linspace(task1.vmin, task1.vmax, 10)
    max_dist = np.linalg.norm(task1.vmax - task1.vmin)
    try:
        task1_values = [task1.value(xn=v) for v in values]
        task2_values = [task2.value(xn=v) for v in values]
    except KeyError:
        task1_values = [task1.value(xn=v, xnm1=vm1) for v in values for vm1 in values]
        task2_values = [task2.value(xn=v, xnm1=vm1) for v in values for vm1 in values]

    m1 = pairwise_distance(task1_values, task1_values)/max_dist
    m2 = pairwise_distance(task2_values, task2_values)/max_dist
    output_dist = np.sum([np.linalg.norm(task1_values[i] - task2_values[i])/max_dist for i in range(len(task1_values))])
    output_dist = output_dist/len(task1_values)
    relational_dist = np.linalg.norm(m1 - m2)
    print(task1.name, task2.name, output_dist, relational_dist)
    return output_dist, relational_dist



def procrustes_to_orthogonal(data1, data2):
    mtx1 = np.array(data1, dtype=np.double, copy=True)
    mtx2 = np.array(data2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2_before = mtx2
    mtx2 = np.dot(mtx2, R.T) * s

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    return mtx1, mtx2, disparity, R, mtx2_before


def points_by_task_1d(points_dict, TASKS, units, n_steps, values_range, xn):
    task_points_dict = OrderedDict()
    task_points_var_dict = {}
    for task in TASKS:
        task_points_dict[task] = {}
        task_points_var_dict[task] = {}

    points_dpca = np.zeros((units, n_steps, len(TASKS)))
    for x1_idx, x1 in enumerate(values_range):
        idx = np.array(xn == x1).flatten()
        for task_idx, task in enumerate(TASKS):
            task_points_dict[task][x1] = np.mean(points_dict[task][idx], axis=0)
            task_points_var_dict[task][x1] = np.std(points_dict[task][idx], axis=0)
            # print(task.name, np.mean(task_points_var_dict[task][x1]))
            points_dpca[:, x1_idx, task_idx] = task_points_dict[task][x1]

    points_dpca = points_dpca - np.mean(points_dpca.reshape((units, -1)), 1)[:, None, None]
    return task_points_dict, points_dpca, task_points_var_dict


def points_by_task_2d(points_dict, TASKS, units, n_steps, values_range, xn, xnm1):
    task_points_dict = {}
    task_points_var_dict = {}
    for task in TASKS:
        task_points_dict[task] = {}
        task_points_var_dict[task] = {}

    points_dpca = np.zeros((units, n_steps, n_steps, len(TASKS)))
    for x1_idx, x1 in enumerate(values_range):
        for x2_idx, x2 in enumerate(values_range):
            idx = np.logical_and(xn == x1, xnm1 == x2).flatten()
            for task_idx, task in enumerate(TASKS):
                task_points_dict[task][(x1, x2)] = np.mean(points_dict[task][idx], axis=0)
                task_points_var_dict[task][(x1, x2)] = np.std(points_dict[task][idx], axis=0)
                points_dpca[:, x1_idx, x2_idx, task_idx] = task_points_dict[task][(x1, x2)]


    return task_points_dict, points_dpca, task_points_var_dict


def task_points_dict_to_dPCA_2D(task_points_dict, TASKS, values_range):

    n_steps = len(values_range)
    units = list(task_points_dict[TASKS[0]].values())[0].shape[-1]

    points_dpca = np.zeros((units, n_steps, n_steps, len(TASKS)))
    for x1_idx, x1 in enumerate(values_range):
        for x2_idx, x2 in enumerate(values_range):
            for task_idx, task in enumerate(TASKS):
                points_dpca[:, x1_idx, x2_idx, task_idx] = task_points_dict[task][(x1, x2)]

    points_dpca = points_dpca - np.mean(points_dpca.reshape((units, -1)), 1)[:, None, None, None]
    return points_dpca

def task_points_dict_to_dPCA_1D(task_points_dict, TASKS, values_range):
    n_steps = len(values_range)
    units = list(task_points_dict[TASKS[0]].values())[0].shape[-1]
    points_dpca = np.zeros((units, n_steps, len(TASKS)))
    for x1_idx, x1 in enumerate(values_range):
        for task_idx, task in enumerate(TASKS):
            points_dpca[:, x1_idx, task_idx] = task_points_dict[task][x1]

    points_dpca = points_dpca - np.mean(points_dpca.reshape((units, -1)), 1)[:, None, None]
    return points_dpca



def dpca_process_1d(points_dpca, TASKS, values_range, mse=0):
    # points_dpca[:, np.arange(10)] = points_dpca[:, np.random.permutation(10)]
    from dPCA import dPCA

    dpca = dPCA.dPCA(labels='at', regularizer=None, n_components=2)
    dpca.protect = ['t']
    # points_dpca[:,:,:,1] = points_dpca[:,:,:,0]
    Z = dpca.fit_transform(points_dpca)
    task_dpca = np.mean(Z['t'][0], axis=0)
    task_dpca = task_dpca - min(task_dpca)
    if task_dpca[0] > task_dpca[-1]:
        task_dpca = - task_dpca
        task_dpca = task_dpca - min(task_dpca)

    # return task_dpca #- min(task_dpca)

    task_colors = cm.jet(np.linspace(0.0, 1.0, len(values_range)))
    fig, axes = plt.subplots(1, 2, figsize=(7, 4))
    for x1_idx, x1 in enumerate(values_range):
        axes[0].plot(Z['t'][0, x1_idx], Z['t'][1, x1_idx],
                        color='lightgrey', zorder=0)
        for task_num in range(len(TASKS)):
            axes[0].scatter(Z['t'][0, x1_idx, task_num], Z['t'][1, x1_idx, task_num],
                            color=TASKS[task_num].color, s=100)


    scores = []
    for task_idx in range(len(TASKS)):
        for x1_idx, x1 in enumerate(values_range):


        # for x_idx, _ in enumerate(values_range):
            axes[1].scatter(Z['a'][0, x1_idx, task_idx], Z['a'][0, x1_idx, task_idx], color=task_colors[x1_idx], s=100)
            # axes[1].scatter(Z['a'][0, x1_idx, task_idx], Z['a'][1, x1_idx, task_idx], color=task_colors[x1_idx], s=100)


        # X, Y = np.arange(len(values_range)).reshape(-1, 1), Z['a'][0, :, task_idx]
        # poly = PolynomialFeatures(2)
        # X = poly.fit_transform(X)
        # reg = LinearRegression()
        # reg.fit(X, Y)
        # scores.append(reg.score(X, Y))

    # print('reg score', np.mean(scores))

    # axes[0, 1].set_title('1st xn component')
    # axes[0, 1].set_xticklabels([])
    # axes[1, 1].set_title('2nd xn component')
    # axes[1,1].set_title('1st xnm1 component')
    # plt.suptitle('{:.5f}'.format(mse))
    axes[0].set_xlabel(r'$dPC_1(\phi)$')
    axes[0].set_ylabel(r'$dPC_2(\phi)$')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].set_xlabel(r'$dPC_1(a_n)$')
    axes[1].set_ylabel(r'$dPC_2(a_{n})$')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title('{:0.2e}'.format(dpca.explained_variance_ratio_['a'][0]))
    axes[0].set_title('{:0.2e}'.format(dpca.explained_variance_ratio_['t'][0]))
    return np.mean(Z['t'][0], axis=0)


def dpca_process_2d(points_dpca, TASKS, values_range, mse=0):
    from dPCA import dPCA

    dpca = dPCA.dPCA(labels='abt', regularizer=None, n_components=2)
    dpca.protect = ['t']
    # points_dpca[:,:,:,1] = points_dpca[:,:,:,0]
    Z = dpca.fit_transform(points_dpca)

    task_dpca = np.mean(np.mean(Z['t'][0], axis=0), axis=0)
    task_dpca = task_dpca - min(task_dpca)
    if task_dpca[0] > task_dpca[-1]:
        task_dpca = - task_dpca
        task_dpca = task_dpca - min(task_dpca)

    return task_dpca #- min(task_dpca)
    # task_colors = cm.jet(np.linspace(0, 1, len(TASKS)))
    # markers = [task.marker for task in TASKS]
    fig, axes = plt.subplots(1, 2, figsize=(7, 4))
    for x1_idx, x1 in enumerate(values_range):
        for x2_idx, x2 in enumerate(values_range):
            axes[0].plot(Z['t'][0, x1_idx, x2_idx], Z['t'][1, x1_idx, x2_idx],
                         color='lightgrey', zorder=0)
            for task_num in range(len(TASKS)):
                axes[0].scatter(Z['t'][0, x1_idx, x2_idx, task_num], Z['t'][1, x1_idx, x2_idx, task_num], color=TASKS[task_num].color, s=100)

    # axes[0, 0].set_title('1st task component')
    # axes[1, 0].set_title('2nd task component')
    # axes[0, 0].set_xticklabels([])
    task_colors = cm.Blues(np.linspace(0.2, 0.8, len(values_range)))


    for task_idx in range(len(TASKS)):
        for x1_idx, x1 in enumerate(values_range):
            for x2_idx, x2 in enumerate(values_range):
        # for x_idx, _ in enumerate(values_range):
                axes[1].scatter(Z['a'][0, x1_idx, x2_idx, task_idx], Z['b'][0, x1_idx, x2_idx, task_idx], facecolor=task_colors[x1_idx], edgecolors=task_colors[x2_idx], linewidth=3, s=100)
            # axes[1, 1].scatter(values_range, Z['a'][1, :, x_idx, task_idx], color=task_colors[task_idx])
            # axes[1,1].scatter(values_range, Z['b'][0, x_idx, :, task_idx], color=TASKS[task_idx].color)

    # axes[1].set_title(r'$a_n, a_{n-1}$')
    # axes[0].set_title('task 2D-dPCA')
    axes[0].set_xlabel(r'$dPC_1(\phi)$')
    axes[0].set_ylabel(r'$dPC_2(\phi)$')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].set_xlabel(r'$dPC_1(a_n)$')
    axes[1].set_ylabel(r'$dPC_1(a_{n-1})$')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    # axes[0, 1].set_xticklabels([])
    # axes[1, 1].set_title('2nd xn component')
    # axes[1,1].set_title('1st xnm1 component')
    # plt.suptitle('{:.5f}'.format(mse))


def custom_loss(target, predict):
    mask = np.logical_not(np.isnan(target))
    masked_diff = np.nan_to_num((predict - target) * mask)
    out = np.sum(masked_diff ** 2.0) / np.sum(mask)
    return out


def newman_metric(A, CC):
    m = np.sum(A) / 2
    k = [np.sum(A[i]) for i in range(len(A))]
    q = 0
    for i in range(len(A)):
        for j in range(len(A)):
            if i != j:
                q += (A[i, j] - ((k[i] + k[j]) / 2 * m)) * (CC[i] == CC[j])

    q = q / (2 * m)
    print(q)


def task_dict_to_two_lists(task_points_dict, TASKS):
    task_lists = []
    for i, task in enumerate(TASKS):
        task_lists.append(list(task_points_dict[TASKS[i]].values()))

    return np.vstack(task_lists[::2]), np.vstack(task_lists[1::2])


def angle(x, y):
    rad = np.arctan2(y, x)
    degrees = np.int(rad * 180 / np.pi)
    if degrees < 0:
        degrees = 360 + degrees
    return degrees % 360


def value_gradient_1D(task1D, xs):
    dx = xs[1] - xs[0]
    fx = lambda x: task1D.value(xn=x)
    value_1D = np.array([fx(x) for x in xs])
    dydx = np.gradient([fx(x) for x in xs], dx)
    gradient_1D = np.abs(dydx)
    return value_1D, gradient_1D


def value_gradient_interval(task1D, xs):
    dx = xs[1] - xs[0]
    fx = lambda x: task1D.value(interval=x, xn=0, xnm1=0)
    value_1D = np.array([fx(x) for x in xs])
    dydx = np.gradient([fx(x) for x in xs], dx)
    gradient_1D = np.abs(dydx)
    return value_1D, gradient_1D

def value_gradient_2D(task2D, xs):
    dx = xs[1] - xs[0]
    value_2D = np.vstack([np.array([task2D.value(xn=xn, xnm1=xnm1) for xnm1 in xs]) for xn in xs]).flatten()
    grad_xn = []
    for xn in xs:
        fx = lambda x: task2D.value(xn=xn, xnm1=x)
        dydx = np.gradient([fx(x) for x in xs], dx)
        grad_xn.append(dydx)

    gradient_2D = np.abs(np.vstack(grad_xn))
    return value_2D, gradient_2D

def value_gradient(task, xs):
    if isinstance(task, Function1D):
        return value_gradient_1D(task, xs)
    if isinstance(task, Function2D):
        return value_gradient_2D(task, xs)
    return value_gradient_interval(task, xs)
    raise ValueError


def value_gradient_1d(task1D, xs):
    value_1D, gradient_1D = value_gradient_1D(task1D, xs)
    mapp = {}
    for i, x in enumerate(xs):
        mapp[(x)] = gradient_1D[i]

    return None, mapp, 0.01, 2.5

def value_gradient_2d(task2D, xs):
    value_2D, gradient_2D = value_gradient_2D(task2D, xs)
    grad_xn = gradient_2D**2 + (gradient_2D.transpose())**2

    mapp = {}
    for i, x in enumerate(xs):
        for j, x2 in enumerate(xs):
            mapp[(x, x2)] = grad_xn[i, j]
    return None, mapp, 0.44, 0.69



def map_points_to_values(xn, xnm1, task):
    xs = np.sort(np.unique(xn))

    if 'x' in task.name:
        mapp_values, mapp, vmin, vmax = value_gradient_1d(task, xs)
        values = np.array([mapp[x] for x in xn])
        # values = mapp_values
        return values, vmin, vmax
    else:
        mapp_values, mapp , vmin, vmax= value_gradient_2d(task, xs)
        values = np.array([mapp[(x, x1)] for x, x1 in zip(xn, xnm1)])
        return values, vmin, vmax




def plot_manifolds_without_grid(points_dict, TASKS, xn, values_dict, text_list, vmin, vmax):
    dim = 2
    projection = '3d' if dim >= 3 else None
    pca = PCA(dim)
    pca.fit(np.vstack(list(points_dict.values())))
    fig = plt.figure()
    ax0 = fig.add_subplot(1, 1, 1, projection=projection)
    ax1 = ax0
    for task_num, task in enumerate(TASKS):
        lowd = pca.transform(points_dict[task])
        colors = cm.viridis((values_dict[task] - vmin) / (vmax - vmin))

        if 'rev' in task.name or 'rot' in task.name:
            ax1.scatter(*np.hsplit(lowd, dim), edgecolors=task.color, marker=task.marker, s=500, alpha=1,
                        facecolors='none', linewidth=3)
            s = 300

        else:
            p = ax1.scatter(*np.hsplit(lowd, dim), marker=task.marker, s=400, alpha=1, edgecolors='none',
                            facecolors=task.color)
            s = 150
            # if 'rot' in task.name:
            #     s += 50


    ax = ax1
    # Get rid of the spines
    ax.set_facecolor("white")
    ax.patch.set_facecolor('white')
    if dim == 3:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.set_zticks([])

    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_title(str(pca.explained_variance_ratio_))
    # plt.colorbar(p)
    # plt.figtext(0.5, 0.8, text_list[0])
    # plt.figtext(0.5, 0.7, text_list[1])
    return fig


def plot_manifolds(points_dict, TASKS, xn, xnm1, values_dict, text_list, vmin, vmax):
    dim = 2
    theta = 0
    projection = '3d' if dim - theta >= 3 else None
    pca = PCA(dim)
    pca.fit(np.vstack(list(points_dict.values())))
    fig = plt.figure()
    ax0 = fig.add_subplot(1, 1, 1, projection=projection)
    # plt.title(pca.explained_variance_ratio_)
    # ax1 = fig.add_subplot(1, 2, 2, projection=projection)
    ax1 = ax0
    # TASKS = TASKS[-2:]
    # out_weights = self.model.weights['fc.weight'].detach().numpy()
    for task_num, task in enumerate(TASKS):
        lowd = pca.transform(points_dict[task])
        # if task == TASKS[-1]:
        #     lowd = -lowd
        if theta == 2:
            angles = [angle(s[0], s[1]) for s in lowd]
            print(min(angles), max(angles))
            if dim == 4:
                ax0.scatter(angles, lowd[:, 2], lowd[:, 3], c=xn, cmap=cm.Blues, marker=task.marker, s=200)
                ax1.scatter(angles, lowd[:, 2], lowd[:, 3], c=values_dict[task], cmap=cm.binary, marker=task.marker,
                            s=200)
            else:
                ax0.scatter(angles, lowd[:, 2], c=xn, cmap=cm.Blues, marker=task.marker, s=200)
                ax1.scatter(angles, lowd[:, 2], c=values_dict[task], cmap=cm.binary, marker=task.marker, s=200)

        elif theta == 1:
            # ax0.scatter(*np.hsplit(lowd[:,1:], dim-1), c=xn, cmap=cm.Blues, marker=task.marker, s=200)
            ax1.scatter(*np.hsplit(lowd[:, 1:], dim - 1), c=values_dict[task], cmap=cm.binary, marker=task.marker,
                        s=200, alpha=0.5)
        else:
            color_values, vminn, vmaxx = map_points_to_values(xn, xnm1, task)
            # colors = cm.jet((xn - vmin) / (vmax - vmin))
            print(task.name, np.min(color_values), np.max(color_values))
            if np.max(color_values) - np.min(color_values) > 0.01:
                colors = cm.binary((color_values - vminn)/(vmaxx - vminn))
            else:
                colors = cm.binary(0.5*np.ones(len(color_values)))

            colors = cm.jet((values_dict[task] - vmin) / (vmax - vmin))
            # ax0.scatter(*np.hsplit(lowd, dim), c=xn, cmap=cm.Blues, marker=task.marker, s=100)
            # if 'rot' in task.name and 'cosine' in task.name:
            #     ax1.scatter(*np.hsplit(lowd, dim), edgecolors=colors, marker=task.marker, s=500, alpha=0.1,
            #                     facecolors='none')

            if 'rev' in task.name or 'rot' in task.name:
                # p = ax1.scatter(*np.hsplit(lowd, dim),  c=values_dict[task], cmap=cm.viridis, marker=task.marker, s=100, alpha=0.5, facecolors='none')
                ax1.scatter(*np.hsplit(lowd, dim), edgecolors=colors, marker=task.marker, s=500, alpha=1,
                            facecolors='none')
                # p = ax1.scatter(*np.hsplit(lowd, dim),  c=values_dict[task], cmap=cm.viridis, marker=task.marker, s=100, alpha=0.5, facecolors='none')
                s = 300
                # if 'rot' in task.name:
                #     s += 50
                # ax1.scatter(*np.hsplit(lowd, dim), edgecolors=task.color, marker=task.marker, s=s, alpha=0.1,
                #                 facecolors='none')
            else:
                p = ax1.scatter(*np.hsplit(lowd, dim), marker=task.marker, s=400, alpha=1, edgecolors='none',
                                facecolors=colors, cmap=cm.viridis)
                s = 150
                if 'rot' in task.name:
                    s += 50
                # p = ax1.scatter(*np.hsplit(lowd, dim), marker=task.marker, s=s, alpha=0.1, edgecolors='none',
                #                 facecolors=task.color)

    ax = ax1
    # Get rid of the spines
    ax.set_facecolor("white")
    ax.patch.set_facecolor('white')
    if dim == 3:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.set_zticks([])

    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_title(str(pca.explained_variance_ratio_))
    # plt.colorbar(p)
    # plt.figtext(0.5, 0.8, text_list[0])
    # plt.figtext(0.5, 0.7, text_list[1])
    # plt.show()
    return fig


def extract_lists_from_trials(inputs, states, task, delay):
    D = defaultdict(list)
    # delay = 40
    for i in range(len(inputs)):
        nonzero = np.argwhere(inputs[i]).flatten()[::5]
        if nonzero.size == 1:
            nonzero = np.append(nonzero, inputs.shape[1])
        candidates = nonzero[:-1:] + 5#nonzero[1::] - 1
        candidates = nonzero[1::] - 1
        D['points'].extend([s for s in states[i, candidates]])
        temp = [states[i, nonzero[j] + 15:nonzero[j+1]:1] for j in range(len(nonzero) - 1)]
        D['all_points'].extend(temp)
        # D['intervals'].extend([None] + [nonzero[j+1] - nonzero[j] for j in range(len(nonzero) - 1)])
        D['time'].extend(list(candidates))
        D['speed'].extend([np.log10(calc_normalized_q_value(states[i, c - 1:c + 1])[0, 0]) for c in candidates])
        xn_seq = [inputs[i, c] for c in nonzero[:-1]]
        D['xn'].extend(xn_seq)
        xnm1_seq = [xn_seq[0]] + xn_seq[:-1]
        D['xnm1'].extend(xnm1_seq)
        if 'int' in task.name:
            D['output'].extend(task.sequence_to_values(nonzero))
        else:
            D['output'].extend(values_to_task([inputs[i, c] for c in nonzero[:-1]], task))

    for key in D.keys():
        if key == 'all_points':
            continue
        if type(D[key][0]) == np.ndarray and D[key][0].ndim > 1:
            D[key] = np.vstack(D[key])
        else:
            D[key] = np.stack(D[key])
    # points, values, times, input_values, speeds = np.vstack(points), np.stack(values), np.stack(times), np.stack(input_values), np.stack(speeds)
    idx = np.argwhere(np.logical_and(D['time'] > delay + 5, D['time'] < states.shape[1] - delay)).flatten()
    for key in D.keys():
        if type(D[key]) == list:
            D[key] = [D[key][ii] for ii in idx]
        else:
            D[key] = D[key][idx]

    return D


def state_input_mapping(inputs, states, task):
    points = []
    values = []
    for i in range(len(inputs)):
        nonzero = np.argwhere(inputs[i]).flatten()[::5]
        for input_idx in nonzero:
            value = inputs[i, input_idx]
            j = input_idx + 15
            while j < len(inputs[i]) and inputs[i, j] == 0:
                points.append(states[i, j])
                values.append(value)
                j += 1

    points, values = np.vstack(points), np.stack(values)
    return points, values


def four_tasks_stuff(TASKS, points_dict, task_points_dict):
    assert len(TASKS) == 4
    pca = PCA(3)
    pca.fit(np.vstack(list(points_dict.values())))
    points_list = []
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for (x1, x2) in task_points_dict[TASKS[0]].keys():
        points = [task_points_dict[TASKS[i]][(x1, x2)] for i in range(4)]
        points_lowd = pca.transform(np.vstack(points))
        curve = np.vstack(
            [points_lowd[2], points_lowd[0], points_lowd[1], points_lowd[2], points_lowd[3], points_lowd[0],
             points_lowd[1], points_lowd[3]])
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], '-o')
        # ax.scatter(curve[:, 0], curve[:, 1], curve[:, 2])

    plt.show();


def diff_activity_single_neurons(points_dict, TASKS, xn, task_points_dict):
    A, B = points_dict[TASKS[0]], points_dict[TASKS[1]]
    diff = np.abs(np.mean(A, axis=0) - np.mean(B, axis=0))  # /(np.var(A, axis=0) + np.var(B, axis=0))
    idx = np.argsort(diff)[-4:]

    fig, axes = plt.subplots(2, 2)
    for i, ax in enumerate(axes.flatten()):
        ## The fiest option is to color the single neuron activity in both manifolds and to look at the gradients
        if False:
            x1 = [k[0] for k in task_points_dict[TASKS[0]].keys()]
            x2 = [a + 5 for a in x1]
            x = x1 + x2
            y = [k[1] for k in task_points_dict[TASKS[0]].keys()]
            y = y + y
            vals1 = [s[i] for s in task_points_dict[TASKS[0]].values()]
            vals2 = [s[i] for s in task_points_dict[TASKS[1]].values()]
            vals = vals1 + vals2
            ax.scatter(x, y, c=vals, cmap=cm.viridis, s=100)
            ax.axis('off')

        ## The second option is to substract one from another and see whether the offset is uniform across memories.
        if True:
            x1 = [k[0] for k in task_points_dict[TASKS[0]].keys()]
            x = x1
            y = [k[1] for k in task_points_dict[TASKS[0]].keys()]
            vals1 = [s[i] for s in task_points_dict[TASKS[0]].values()]
            vals2 = [s[i] for s in task_points_dict[TASKS[1]].values()]
            vals = [abs(vals1[j] - vals2[j]) for j in range(len(vals1))]
            ax.scatter(x, y, c=vals, cmap=cm.viridis, s=100)
            ax.axis('off')
        # ax.hist(A[:,idx[i]], alpha=0.4)
        # ax.hist(B[:,idx[i]], alpha=0.4)

    plt.suptitle(diff[idx[-1]])


def pc_dimension_coding_scores(points, var_dict):
    dim = 5
    pca = PCA(dim)
    points = pca.fit_transform(points)
    colors = ['blue', 'orange', 'green', 'red']
    plt.rcParams['axes.facecolor'] = 'white'
    plt.clf()
    # plt.scatter([], [], facecolors='black', s=200, label='task A')
    # plt.scatter([], [], edgecolors='black', facecolors='none', s=200, label='task B')
    # plt.legend()
    # plt.savefig('legend.pdf')
    # exit()
    for color, key in zip(colors, var_dict.keys()):
        scores = []
        scorespoly = []
        scorespoly3 = []
        for c in range(dim):
            # reg = SVR(max_iter=200, kernel='poly')
            # print(f'n dim {c}')
            # 1D regression
            scores.append(fit_regression(points[:, :c + 1], var_dict[key], deg=1))
            scorespoly.append(fit_regression(points[:, :c + 1], var_dict[key], deg=2))
            # scorespoly3.append(fit_regression(points[:,:c+1], var_dict[key], deg=3))

        if key == 'xnm1':
            key = r'$p_{n-1}$'
        elif key == 'xn':
            key = r'$p_{n}$'

        plt.plot(scores, label=key, alpha=1, linewidth=2, color=color)
        plt.plot(scorespoly, alpha=1, linewidth=2, linestyle='--', color=color)
        # plt.plot(scorespoly3, alpha=1, linewidth=1, linestyle='-.', color=color)

    plt.xticks([0, 1, 2, 3, 4], [1, 2, 3, 4, 5], fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('PCs', fontsize=20)
    plt.ylabel('regression score', fontsize=20)


def get_distances(array):
    n_tasks = len(array)
    distances = np.zeros((n_tasks, n_tasks))
    for i1 in range(n_tasks):
        for i2 in range(i1, n_tasks):
            distances[i1, i2] = distances[i2, i1] = np.linalg.norm(array[i1] - array[i2])

    return distances #/ np.max(distances)

def get_distances_reflection(array):
    n_tasks = len(array)
    distances = np.zeros((n_tasks, n_tasks))
    for i1 in range(n_tasks):
        for i2 in range(i1, n_tasks):
            distances[i1, i2] = distances[i2, i1] = np.linalg.norm(array[i1] - (4 - array[i2]))

    return distances #/ np.max(distances)

def get_distances_rotation(array):
    n_tasks = len(array)
    distances = np.zeros((n_tasks, n_tasks))
    for i1 in range(n_tasks):
        for i2 in range(i1, n_tasks):
            distances[i1, i2] = distances[i2, i1] = np.linalg.norm(array[i1] - array[i2][::-1])

    return distances# / np.max(distances)

def get_distances_rotation_reflection(array):
    n_tasks = len(array)
    distances = np.zeros((n_tasks, n_tasks))
    for i1 in range(n_tasks):
        for i2 in range(i1, n_tasks):
            distances[i1, i2] = distances[i2, i1] = np.linalg.norm(array[i1] - (4 - array[i2][::-1]))

    return distances #/ np.max(distances)

def fit_regression(X, y, deg):
    if deg > 1:
        poly = PolynomialFeatures(deg)
        X = poly.fit_transform(X)

    reg = LinearRegression()
    scores = []
    for _ in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        reg.fit(X_train, y_train)
        scores.append(reg.score(X_test, y_test))

    return np.mean(scores)


def values_to_task(values, task):
    return [values[0]] + [task.value(xn=values[i + 1], xnm1=values[i]) for i in range(len(values) - 1)]






    return ' '.join(stress_list)
