import matplotlib.pyplot as plt

from analysis.spectrum_analysis import SpectrumAnalysis
from data.custom_data_generator import *
from model.model_wrapper import ModelWrapper
from model.pt_models import VanillaArchitecture


def get_tasks(n_bits):
    return [FlipFlopGenerator(n_bits=n_bits), OrthogonalFlipFlopGenerator(n_bits=n_bits),
            ParallelFlipFlopGenerator(n_bits=n_bits)]


def plot_spectrum():
    fig, axes = plt.subplots(2, 1, figsize=(3, 5))
    colors = ['lightblue', 'royalblue', 'darkblue']

    soft_bits_to_points = {0: defaultdict(list), 1: defaultdict(list), 2: defaultdict(list)}
    hard_bits_to_points = {0: defaultdict(list), 1: defaultdict(list), 2: defaultdict(list)}

    for n_bits in bits_range:
        v = get_tasks(n_bits)
        for i, train_data in enumerate(v):
            kwargs['train_data'] = train_data
            full_wrapper = ModelWrapper(**kwargs)
            soft_average = list(full_wrapper.get_file('soft_average').values())
            hard_average = list(full_wrapper.get_file('hard_average').values())
            hard_average = 100 * np.array(hard_average)
            soft_bits_to_points[i][n_bits].append(soft_average)
            hard_bits_to_points[i][n_bits].append(hard_average)

    for i in range(3):
        x_locations = [k + i * 0.2 for k in soft_bits_to_points[i].keys()]
        soft_means = []
        soft_stds = []
        hard_means = []
        hard_stds = []

        for n_bits, soft_points in soft_bits_to_points[i].items():
            soft_means.append(np.mean(soft_points))
            soft_stds.append(np.std(soft_points))

        for n_bits, hard_points in hard_bits_to_points[i].items():
            hard_means.append(np.mean(hard_points))
            hard_stds.append(np.std(hard_points))

        axes[0].errorbar(x_locations, soft_means, yerr=soft_stds, fmt='o', color=colors[i])
        axes[1].errorbar(x_locations, hard_means, yerr=hard_stds, fmt='o', color=colors[i])

    # Set x-axis labels, legends, and titles
    xticks = [k + 2.2 for k in range(len(bits_range))]
    x_labels = [str(k + 2) for k in range(len(bits_range))]
    axes[0].set_xticks([])
    axes[0].set_yticks([0, 0.1])
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    # axes[0].set_xticklabels(x_labels)
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels(x_labels)

    axes[1].set_xlabel('n. bits')
    axes[0].set_ylabel(r'$\sum w(|\lambda|)|\lambda|$ ')
    axes[1].set_ylabel(r'n. $|\lambda| > 1$')

    plt.tight_layout()
    plt.savefig('figures/spectrum_ff.pdf', dpi=1000)


if __name__ == '__main__':
    kwargs = {'architecture_func': VanillaArchitecture,
              'units': 100,
              'instance_range': range(800, 810),
              'recurrent_bias': False,
              'readout_bias': True,
              'freeze_params': None,
              'weight_init_func': None}

    bits_range = range(2, 7)

    n_networks = len(kwargs['instance_range'])

    for n_bits in bits_range:
        v = get_tasks(n_bits)
        for i, train_data in enumerate(v):
            kwargs['train_data'] = train_data
            full_wrapper = ModelWrapper(**kwargs)
            full_wrapper.analyze([SpectrumAnalysis])

    plot_spectrum()
