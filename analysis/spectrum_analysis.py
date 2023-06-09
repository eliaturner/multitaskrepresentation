import numpy as np
from analysis.analyzer import Analyzer


def calculate_soft_weighted_average(arr):
    min_range = 0.3
    max_range = 1.3
    diff = max_range - min_range
    weighted_arr = np.where(arr < min_range, arr * 0, np.where(arr <= max_range, (arr - min_range) / diff, arr * 1))
    weighted_sum = np.sum(weighted_arr)
    return weighted_sum / len(arr)


def calculate_hard_average(arr, thresh=1):
    return np.sum(np.where(arr > thresh, 1, 0)) / len(arr)


class SpectrumAnalysis(Analyzer):
    name = 'kappa'

    def run(self):
        eigvals = np.linalg.eigvals(self.rank_analyzer.W_rec)
        soft_magnitude = calculate_soft_weighted_average(np.abs(eigvals))
        hard_magnitude = calculate_hard_average(np.abs(eigvals))
        self.save_file(soft_magnitude, 'soft_average')
        self.save_file(hard_magnitude, 'hard_average')
