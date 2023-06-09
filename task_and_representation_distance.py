import matplotlib.pyplot as plt

from analysis.taskpairwisekappadistance import TasksPairwiseKappaDistance
from data.data_config import *
from model.model_wrapper import ModelWrapper
from model.pt_models import Rank2Architechture

values_range = np.linspace(1, 3, 10)


def pairwise_distance(array, pairs):
    return [np.linalg.norm(np.array(array[pair[0]]) - np.array(array[pair[1]])) for pair in pairs]


def f_and_fprime(TASKS):
    f_values = []

    for task in TASKS:
        f_values.append([task.value(xn=i) for i in values_range])

    fprime_values = []
    dx = 0.01
    for y in f_values:
        dydx = np.abs(np.gradient(y, dx))
        dydx = dydx / np.max(dydx)
        fprime_values.append(dydx)

    return f_values, fprime_values


def plot_tasks_vertical(f_values):
    fig, axes = plt.subplots(6, 1, figsize=(0.75, 6))
    for i in range(len(TASKS)):
        axes[i].plot(values_range, f_values[i], color='black', label=r'$\phi_1$')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    plt.savefig('tasksvertical.pdf', bbox_inches='tight')


def plot_pairs_fprime(fprime_values, pairs):
    fig, axes = plt.subplots(1, 15, figsize=(8, 0.6))
    for idx, i in enumerate(indices):
        task0, task1 = pairs[i]
        axes[idx].plot(values_range, fprime_values[task0], color='black', label=r'$|\phi_1\'|$')
        axes[idx].plot(values_range, fprime_values[task1], color='black', linestyle='--', label=r'$|\phi_2\'|$')
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])

    plt.savefig('pairwisefunctionsderivative.pdf', bbox_inches='tight', dpi=1000)


def plot_pairs_f(f_values, pairs):
    fig, axes = plt.subplots(1, 15, figsize=(8, 0.6))

    for idx, i in enumerate(indices):
        task0, task1 = pairs[i]
        axes[idx].plot(values_range, f_values[task0], color='black', label=r'$|\phi_1\'|$')
        axes[idx].plot(values_range, f_values[task1], color='black', linestyle='--', label=r'$|\phi_2\'|$')
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])

    plt.savefig('figures/pairwisefunctions.pdf', bbox_inches='tight', dpi=1000)


def plot_pairwise_task_distance(representation_distance, f_distance, fprime_distance):
    fig, axes = plt.subplots(2, 1, figsize=(8, 4))
    ax = axes[0]
    ax.violinplot(representation_distance)
    values_distance = [v / max(f_distance) for v in f_distance]
    derivative_distance = [v / max(fprime_distance) for v in fprime_distance]
    axes[1].plot(values_distance, '-', label=r'$\Vert{\phi_1-\phi_2}\Vert_2$', color='lightgreen')
    axes[1].plot(derivative_distance, '-', label=r'$\Vert{|\phi_1\'|-|\phi_2\'|}\Vert_2$', color='darkgreen')
    ax.set_ylabel(r'$d(R_1, R_2)$')
    axes[1].set_ylabel(r'$d(\phi_1,\phi_2)$')
    ax.set_xticks([])
    axes[1].set_xticks([])
    ax.set_yticks([0, 2, 4])
    axes[1].set_yticks([0, 1])
    plt.savefig('figures/attractor_similarity.pdf', bbox_inches='tight', dpi=1000)
    plt.close()


TASKS = [X4Rotate(), X2Rotate(), XReverse(), X(), X2(), X4()]

from itertools import combinations

pairs = list(combinations(np.arange(len(TASKS)), 2))

kwargs = {'architecture_func': Rank2Architechture,
          'units': 100,
          'train_data': FamilyOfTasksGenerator(TASKS),
          'instance_range': range(2000, 2025),
          }

wrapper = ModelWrapper(**kwargs)

wrapper.analyze([TasksPairwiseKappaDistance])
representation_distance = np.vstack(list(wrapper.get_file('kappa_distance_array').values()))
means = np.mean(representation_distance, axis=0)
indices = np.argsort(means)
representation_distance = representation_distance[:, indices]
pairs = [pairs[i] for i in indices]

f_values, fprime_values = f_and_fprime(TASKS)
f_distance = pairwise_distance(f_values, pairs)
fprime_distance = pairwise_distance(fprime_values, pairs)
plot_pairwise_task_distance(representation_distance, f_distance, fprime_distance)
