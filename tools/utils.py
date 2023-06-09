import pickle as pkl

import os
from pathlib import Path


def load_pickle(file_name):
    # print('loading', file_name)
    with open(file_name, 'rb') as handle:
        file = pkl.load(handle)
    return file


def dump_pickle(file_name, var):
    with open(file_name, 'wb') as handle:
        pkl.dump(var, handle)


def load_score(full_path):
    scores = [0]
    for filename in os.listdir(full_path):
        if 'score' in filename:
            scores.append(load_pickle(full_path + '/' + filename))

    return max(scores)


def create_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)
