import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from analysis.analyzer import TaskFamilyLowRankAnalyzer

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist, squareform


def parameterize_curve(points):
    # Compute the Euclidean distance between each point and the next point
    diffs = np.diff(points, axis=0)
    distances = np.sqrt((diffs ** 2).sum(axis=1))

    # Compute the cumulative sum of distances to get the length of the path up to each point
    cumulative_lengths = np.concatenate([[0], np.cumsum(distances)])

    # Normalize the cumulative lengths so they go from 0 to 1
    normalized_lengths = cumulative_lengths / cumulative_lengths[-1]

    # Use interpolation to create a function that can take a 2D point and return its normalized length
    length_to_point = interp1d(normalized_lengths, points, axis=0)

    return normalized_lengths, length_to_point


def point_to_length(new_point, points, normalized_lengths):
    # Compute the Euclidean distance from the new point to each original point
    distances = np.sqrt(((points - new_point) ** 2).sum(axis=1))

    # Find the index of the closest original point
    closest_index = np.argmin(distances)

    # Return the normalized length of the closest original point
    return normalized_lengths[closest_index]




# Assuming "points" is your numpy array of size [n_points, 2]

# Example usage
# curve = np.array([[0, 0], [1, 1], [2, 0], [3, 1], [4, 0]])  # Example curve
# point = np.array([0.5, 2.5])  # Example point
#
# parameterized_curve = parameterize_curve(curve)
# interpolated_point = interpolate_point(curve, parameterized_curve, point)
#
# print("Interpolated point:", interpolated_point)


def shared_histogram(arrays, num_bins):
    # Step 1: Determine the range of values
    min_value = np.min([np.min(arr) for arr in arrays])
    max_value = np.max([np.max(arr) for arr in arrays])

    # Step 2: Calculate the bin edges
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)

    # Step 3: Initialize count tracker and histogram
    count_tracker = np.zeros(num_bins)
    hist = np.zeros(num_bins)

    # Step 4: Process arrays
    for arr in arrays:
        curr_hist = np.histogram(arr, bins=bin_edges)[0]
        # Step 4a: Calculate the histogram counts for the current array
        hist += curr_hist

        # Step 4b: Increment count for corresponding bins
        count_tracker += (curr_hist > 0)

    # Step 5: Normalize histogram
    shared_hist = np.zeros(num_bins)
    non_zero_bins = count_tracker.nonzero()
    shared_hist[non_zero_bins] = np.divide(hist[non_zero_bins], count_tracker[non_zero_bins])

    return shared_hist, bin_edges


def linear_fit(A, B):
    coefficients = np.polyfit(A, B, 1)

    # Extract the optimal w and b
    w = coefficients[0]
    b = coefficients[1]

    # Calculate the error
    error = np.linalg.norm((w * A + b) - B)
    return error


class KappaLine(TaskFamilyLowRankAnalyzer):
    name = 'kappa'

    def run(self):
        KAPPAS, _, _ = self.get_kappas()
        colors = self.return_color(cmap=cm.viridis)
        markers = ['o', 'v', 'x', 's', 'p', '.']
        all_kappas = []
        for task_num, task in enumerate(self.TASKS):
            kappas = KAPPAS[task]
            if np.sum(kappas[:,0]) < 0:
                kappas = -kappas
            all_kappas.append(kappas)

        all_kappas = np.vstack(all_kappas)
        all_kappas = all_kappas[np.argsort(all_kappas[:, 1])]
        normalized_lengths, length_to_point = parameterize_curve(all_kappas)
        from sklearn.decomposition import PCA
        pca = PCA(1)
        pca.fit(all_kappas)

        fig, axes = plt.subplots(2, 2, figsize=(7, 5))

        # axes[1, 0].hist(normalized_lengths, bins=9, color='black')
        # axes[1, 1].hist(pca.transform(all_kappas).flatten(), bins=9, color='black')
        axes[0, 0].set_xticks([])
        axes[0, 1].set_xticks([])
        axes[0, 0].set_yticks([])
        axes[0, 1].set_yticks([])
        axes[0, 0].set_title('Arc length')
        axes[0, 1].set_title('PC1')
        axes[1, 0].set_xticks([])
        axes[1, 1].set_xticks([])
        axes[1, 0].set_yticks([0, 10])
        # axes[1, 0].set_ylim(0, 25)
        axes[1, 1].set_yticks([])
        output_error_arc = np.zeros((len(self.TASKS), len(self.TASKS)))
        output_error_pc1 = np.zeros((len(self.TASKS), len(self.TASKS)))

        values_range = self.data_params.validation_range
        TASKS_OUTPUTS = [ np.array([task.value(xn=v) for v in values_range]) for task in self.TASKS]

        arc_length_hist = []
        pc1_hist = []

        for task_num, task in enumerate(self.TASKS):
            kappas = KAPPAS[task]
            if np.sum(kappas[:,0]) < 0:
                kappas = -kappas
            new_parameterizations = np.array([point_to_length(new_point, all_kappas, normalized_lengths) for new_point in kappas]).flatten()

            arc_length_hist.append(new_parameterizations)



            axes[0, 0].scatter(new_parameterizations, len(new_parameterizations)*[task_num], color=colors)
            k1d = pca.transform(kappas).flatten()
            pc1_hist.append(k1d)
            axes[0, 1].scatter(k1d, len(k1d)*[task_num], color=colors)

            for task_num2, task2 in enumerate(self.TASKS):
                output_error_arc[task_num, task_num2] = (linear_fit(new_parameterizations, TASKS_OUTPUTS[task_num2]))
                output_error_pc1[task_num, task_num2] = (linear_fit(k1d, TASKS_OUTPUTS[task_num2]))

        n_bins = 13
        self.save_file(arc_length_hist, 'arc_length_hist')
        self.save_file(pc1_hist, 'pc1_hist')
        shared_hist, bin_edges = shared_histogram(arc_length_hist, n_bins)
        # axes[1, 0].bar(bin_edges[:-1], shared_hist, width=np.diff(bin_edges), align='edge', color='black')
        # shared_hist, bin_edges = shared_histogram(pc1_hist, n_bins)
        # axes[1, 1].bar(bin_edges[:-1], shared_hist, width=np.diff(bin_edges), align='edge', color='black')
        index_output = 1
        axes[index_output, 0].imshow(output_error_arc, cmap='binary', vmin=0, vmax=2)
        axes[index_output, 1].imshow(output_error_pc1, cmap='binary', vmin=0, vmax=2)
        axes[index_output, 0].set_xticks([])
        axes[index_output, 1].set_xticks([])
        axes[index_output, 0].set_yticks([])
        axes[index_output, 1].set_yticks([])
        axes[index_output, 0].set_xlabel('output predicted')
        axes[index_output, 0].set_ylabel('attractor fitted')

        self.save_plot(plt, 'weird')