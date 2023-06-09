from abc import abstractmethod, ABC
from scipy.stats import logistic
from pathlib import Path
from tools.utils import load_pickle, dump_pickle
import numpy as np
from cached_property import cached_property
from tools.rsg_utils import transition_indices
from collections import defaultdict
from data.data_generator import RNNDataGenerator
MINIMAL_INPUT_LENGTH = 60


class CustomDataGenerator(RNNDataGenerator):
    def __init__(self, n_inputs, n_outputs, n_batch=400, steps=200, min_delay=10, max_delay=40, delta=5, vmin=-1, vmax=1):
        self._steps = steps
        self.n_batch = n_batch
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.delta = delta
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self.vmin = vmin
        self.vmax = vmax
        self.train_noise = True
        self.dontcare_offset = 0
        self.offset = self.delta + self.dontcare_offset

    @property
    def name(self):
        if self.vmin != -1 or self.vmax != 1:
            return f'{self.task_name}_{self._steps}_{self.vmin}_{self.vmax}'
        return f'{self.task_name}_{self._steps}'

    @property
    def steps(self):
        return self._steps

    @property
    def n_inputs(self):
        return self._n_inputs

    @n_inputs.setter
    def n_inputs(self, value):
        self._n_inputs = value

    @property
    def n_outputs(self):
        return self._n_outputs

    @n_outputs.setter
    def n_outputs(self, value):
        self._n_outputs = value

    def generate_train_data(self):
        x = np.zeros((self.n_batch, self.steps, self.n_inputs))
        y = np.zeros((self.n_batch, self.steps, self.n_outputs))
        for i in range(self.n_batch):
            x[i], y[i] = self.generate_training_trial()

        # if self.train_noise:
        #     x = x + np.random.normal(0, 0.05, x.shape)
        return x, y

    def generate_validation_data(self):
        return self.generate_train_data()

    @abstractmethod
    def generate_training_trial(self):
        pass


    def generate_ff_output(self, transitions, values):
        y_channel = np.empty(self.steps)
        y_channel[:] = np.nan
        for i in range(len(transitions) - 1):
            value = values[i]
            y_channel[transitions[i] + self.offset:transitions[i + 1]] = value

        return y_channel

    def generate_wave_output(self, transitions, values):
        y_channel = np.empty(self.steps)
        y_channel[:] = np.nan
        sample_points = 4 * np.pi * np.linspace(0, 1, self.max_delay)
        for i in range(len(transitions) - 1):
            seq = 0.5 * np.sin(sample_points) + values[i]
            length = transitions[i + 1] - transitions[i] - self.offset
            total_seq = seq
            while len(total_seq) < length:
                total_seq = np.append(total_seq, seq[1:])
            y_channel[transitions[i] + self.offset:transitions[i + 1]] = total_seq[:length]

        return y_channel

    def generate_output_sequence(self, x_seq, output_generator):
        transitions = np.argwhere(x_seq).flatten()[::5]
        values = [x_seq[t] for t in transitions]
        if self.steps - transitions[-1] > self.offset:
            transitions = np.append(transitions, self.steps)

        return output_generator(transitions, values)


    def get_minimal_input(self):
        pass


class MemoryBitsBaseGenerator(CustomDataGenerator):
    def __init__(self, n_bits=2):
        super().__init__(n_inputs=n_bits, n_outputs=n_bits)
        self.n_bits = n_bits

    @property
    def task_name(self):
        return f'{self.n_bits}bits'

    def in_channel(self, bit):
        return bit

    def get_output_generator(self, task_num):
        pass

    def randomize_binary_value(self):
        return np.random.choice([self.vmin, self.vmax])

    def randomize_continuous_value(self):
        return np.random.uniform(self.vmin + 0.0, self.vmax - 0.0)

    def randomize_bit(self):
        return np.random.randint(0, self.n_bits)

    def generate_random_sequence(self, value_type='binary', val=False):
        x_random = np.zeros((self.steps))
        t = 0
        while t < len(x_random):
            if val:
                delay = self.max_delay

            else:
                delay = np.random.randint(self.min_delay, self.max_delay)

            if value_type == 'binary':
                value = self.randomize_binary_value()
            else:
                value = self.randomize_continuous_value()

            x_random[t:t+self.delta] = value
            t += delay+self.delta

        return x_random


class MemoryBitsGenerator(MemoryBitsBaseGenerator):
    def generate_training_trial(self):
        task_num = np.random.randint(self.n_bits)
        x_seq = self.generate_random_sequence()
        return self.generate_trial(task_num, x_seq)

    def generate_trial(self, task_num, x_seq):
        xx = np.zeros((self.steps, self.n_inputs))
        xx[:,task_num] = x_seq
        # xx += np.random.normal(0, 0.2, xx.shape)
        yy = np.empty((self.steps, self.n_outputs))
        yy[:] = np.nan
        yy[:, task_num] = self.generate_output_sequence(x_seq, self.get_output_generator(task_num))
        return xx, yy

    def task_num_to_value_type(self, task_num):
        return 'binary'

    def generate_validation_data(self):
        reps = 100
        n_trials = reps*self.n_bits
        x = np.zeros((n_trials, self.steps, self.n_inputs))
        y = np.zeros((n_trials, self.steps, self.n_outputs))
        idx = 0
        for rep in range(reps):
            x_seq = self.generate_random_sequence(value_type=self.task_num_to_value_type(0), val=True)
            for i in range(self.n_bits):
                x[idx], y[idx] = self.generate_trial(task_num=i, x_seq=x_seq)
                idx += 1

        return x, y

    def get_minimal_input(self):
        X = np.zeros((2*self.n_bits, MINIMAL_INPUT_LENGTH, self.n_inputs))
        for bit in range(self.n_bits):
            X[2*bit, :self.delta, bit] = self.vmin
            X[2*bit+1, :self.delta, bit] = self.vmax

        return X


class FlipFlopGenerator(MemoryBitsGenerator):

    @property
    def task_name(self):
        return f'flipflop' + super().task_name

    def get_output_generator(self, task_num):
        return self.generate_ff_output



class CyclesGenerator(MemoryBitsGenerator):
    @property
    def task_name(self):
        return f'cycles' + super().task_name

    def get_output_generator(self, task_num):
        return self.generate_wave_output

class LinesGenerator(MemoryBitsGenerator):
    @property
    def task_name(self):
        return f'lines' + super().task_name

    def task_num_to_value_type(self, task_num):
        return 'continuous'

    def get_output_generator(self, task_num):
        return self.generate_ff_output

    def get_minimal_input(self):
        X = []
        for task_num in range(self.n_bits):
            x = np.zeros((10, MINIMAL_INPUT_LENGTH, self.n_inputs))
            values_range = np.linspace(self.vmin, self.vmax, 10)
            for i in range(10):
                x[i, :self.delta, task_num] = values_range[i]

            X.append(x)

        return np.vstack(X)

    def generate_training_trial(self):
        task_num = np.random.randint(self.n_bits)
        x_seq = self.generate_random_sequence(value_type='continuous')
        return self.generate_trial(task_num, x_seq)


class FlipFlopCycleGenerator(MemoryBitsGenerator):
    @property
    def task_name(self):
        return f'flipflopcycle' + super().task_name

    def get_output_generator(self, task_num):
        if task_num % 2 == 0:
            return self.generate_ff_output
        else:
            return self.generate_wave_output


class MemoryBitsHybridGenerator(MemoryBitsBaseGenerator):
    def task_num_to_value_type(self, task_num):
        pass

    def generate_training_trial(self):
        task_num = np.random.randint(self.n_bits)
        x_seq = self.generate_random_sequence(value_type=self.task_num_to_value_type(task_num))
        return self.generate_trial(task_num, x_seq)

    def generate_trial(self, task_num, x_seq):
        xx = np.zeros((self.steps, self.n_inputs))
        xx[:,task_num] = x_seq
        # xx += np.random.normal(0, 0.2, xx.shape)
        yy = np.empty((self.steps, self.n_outputs))
        yy[:] = np.nan
        yy[:, task_num] = self.generate_output_sequence(x_seq, self.get_output_generator(task_num))
        return xx, yy

    def generate_validation_data(self):
        reps = 20
        n_trials = reps*self.n_bits
        x = np.zeros((n_trials, self.steps, self.n_inputs))
        y = np.zeros((n_trials, self.steps, self.n_outputs))
        idx = 0
        for rep in range(reps):
            for task_num in range(self.n_bits):
                x_seq = self.generate_random_sequence(value_type=self.task_num_to_value_type(task_num), val=True)
                x[idx], y[idx] = self.generate_trial(task_num=task_num, x_seq=x_seq)
                idx += 1

        return x, y

    def get_minimal_input(self):
        X = np.zeros((2 + 10, MINIMAL_INPUT_LENGTH, self.n_inputs))
        X[0, :self.delta, 0] = self.vmin
        X[1, :self.delta, 0] = self.vmax

        values_range = np.linspace(self.vmin , self.vmax , 10)
        for i in range(10):
            X[i+2, :self.delta, 1] = values_range[i]

        return X


class FlipFlopLineGenerator(MemoryBitsHybridGenerator):
    @property
    def task_name(self):
        return f'flipflop_line' + super().task_name

    def task_num_to_value_type(self, task_num):
        if task_num == 0:
            return 'binary'
        else:
            return 'continuous'

    def get_output_generator(self, task_num):
        return self.generate_ff_output



class LimitCycleLineGenerator(MemoryBitsHybridGenerator):
    @property
    def task_name(self):
        return f'cycle_line' + super().task_name

    def task_num_to_value_type(self, task_num):
        if task_num == 0:
            return 'binary'
        else:
            return 'continuous'

    def get_output_generator(self, task_num):
        if task_num == 0:
            return self.generate_wave_output
        return self.generate_ff_output

    # def get_minimal_input(self):
    #     X = np.zeros((2 + 10, self.max_delay, self.n_inputs))
    #     X[0, :self.delta, 0] = self.vmin
    #     X[1, :self.delta, 0] = self.vmax
    #
    #     values_range = np.linspace(self.vmin, self.vmax, 10)
    #     for i in range(10):
    #         X[i + 2, :self.delta, 1] = values_range[i]
    #
    #     return X


class SineWaveGenerator(CustomDataGenerator):
    def __init__(self):
        super(SineWaveGenerator, self).__init__(n_inputs=1, n_outputs=1, steps=100)

    @property
    def task_name(self):
        return f'sinewave'

    def generate_training_trial(self):
        return self.generate_trial()

    def generate_trial(self):
        x = np.zeros((self.steps, self.n_inputs))
        y = np.zeros((self.steps, self.n_outputs))
        y[:,0] = np.sin(8 * np.pi * np.linspace(0, 1, self.steps))
        return x, y

    def generate_validation_data(self):
        reps = 100
        n_trials = reps
        x = np.zeros((n_trials, self.steps, self.n_inputs))
        y = np.zeros((n_trials, self.steps, self.n_outputs))
        for rep in range(reps):
            x[rep], y[rep] = self.generate_trial()

        return x, y

    def generate_train_data(self):
        return self.generate_validation_data()



class RingGenerator(CustomDataGenerator):
    def __init__(self):
        super(RingGenerator, self).__init__(n_inputs=2, n_outputs=2, steps=200)

    @property
    def task_name(self):
        return f'ring'


    def generate_training_trial(self):
        return self.generate_trial(val=False)

    def generate_trial(self, val=False):
        y = np.zeros((self.steps, self.n_outputs))
        x = self.generate_random_sequence(val)

        transitions = np.argwhere(x[:,0]).flatten()[::5]
        values = [np.arctan((x[t, 1]-2)/(x[t, 0] - 2)) for t in transitions]
        sines = [np.sin(v) for v in values]
        cosines = [np.cos(v) for v in values]
        if self.steps - transitions[-1] > 5:
            transitions = np.append(transitions, self.steps)

        y[:, 0] = self.generate_ff_output(transitions, sines)
        y[:, 1] = self.generate_ff_output(transitions, cosines)
        return x, y

    def generate_random_sequence(self, val=False):
        x_random = np.zeros((self.steps, self.n_inputs))
        t = 0
        while t < len(x_random):
            if val:
                delay = self.max_delay

            else:
                delay = np.random.randint(self.min_delay, self.max_delay)

            value1 = np.random.uniform(1, 3)
            value2 = np.random.uniform(1, 3)
            x_random[t:t+self.delta, 0], x_random[t:t+self.delta, 1] = value1, value2
            t += delay+self.delta

        return x_random

    def generate_validation_data(self):
        reps = 200
        n_trials = reps
        x = np.zeros((n_trials, self.steps, self.n_inputs))
        y = np.zeros((n_trials, self.steps, self.n_outputs))
        for rep in range(reps):
            x[rep], y[rep] = self.generate_trial(val=True)

        return x, y

    def generate_train_data(self):
        return self.generate_validation_data()



class ParallelMemoryBitsGenerator(MemoryBitsBaseGenerator):
    @property
    def task_name(self):
        return f'parallel_' + super().task_name

    def generate_training_trial(self):
        return self.generate_trial()

    def generate_trial(self):
        yy = np.empty((self.steps, self.n_outputs))
        yy[:] = np.nan
        xx = self.generate_parallel_random_sequence()
        for bit in range(self.n_bits):
            yy[:, bit] = self.generate_output_sequence(xx[:, bit], self.get_output_generator(bit))

        # yy[np.any(np.isnan(yy), axis=1)] = np.nan
        return xx, yy

    def task_num_to_value_type(self, task_num):
        return 'binary'


    def generate_parallel_random_sequence(self, val=False):
        x_random = np.zeros((self.steps, self.n_bits))
        t = 0
        while t < len(x_random):
            if t == 0:
                for bit in range(self.n_bits):
                    x_random[:self.delta, bit] = self.randomize_binary_value()

            else:
                bit = self.randomize_bit()
                value_type = self.task_num_to_value_type(bit)
                if value_type == 'binary':
                    value = self.randomize_binary_value()
                else:
                    value = self.randomize_continuous_value()
                x_random[t:t+self.delta, bit] = value

            if val:
                delay = self.max_delay
            else:
                delay = np.random.randint(self.min_delay, self.max_delay-10)

            t += delay+self.delta

        return x_random

    def generate_validation_data(self):
        reps = 100
        n_trials = reps * self.n_bits
        x = np.zeros((n_trials, self.steps, self.n_inputs))
        y = np.zeros((n_trials, self.steps, self.n_outputs))
        idx = 0
        for rep in range(reps):
            for i in range(self.n_bits):
                x[idx], y[idx] = self.generate_trial()
                idx += 1

        return x, y

    def get_minimal_input(self):
        X = np.zeros((self.n_bits**2, MINIMAL_INPUT_LENGTH, self.n_inputs))
        from itertools import product
        values = [self.vmin, self.vmax]
        options = product(values, repeat=self.n_bits)

        for i, bits in enumerate(options):
            for j, bit in enumerate(bits):
                X[i, :self.delta, j] = bit

        return X

class ParallelFlipFlopGenerator(ParallelMemoryBitsGenerator):

    @property
    def task_name(self):
        return f'flipflop' + super().task_name

    def get_output_generator(self, task_num):
        return self.generate_ff_output



class ParallelCyclesGenerator(ParallelMemoryBitsGenerator):
    @property
    def task_name(self):
        return f'cycles' + super().task_name

    def get_output_generator(self, task_num):
        return self.generate_wave_output


class ParallelLineGenerator(ParallelMemoryBitsGenerator):
    def task_num_to_value_type(self, task_num):
        return 'continous'

    @property
    def task_name(self):
        return f'lines' + super().task_name

    def get_output_generator(self, task_num):
        return self.generate_ff_output

    def get_minimal_input(self):
        X = []
        for task_num in range(self.n_bits):
            x = np.zeros((10, MINIMAL_INPUT_LENGTH, self.n_inputs))
            values_range = np.linspace(self.vmin, self.vmax, 10)
            for i in range(10):
                x[i, :self.delta, task_num] = values_range[i]

            X.append(x)

        return np.vstack(X)


class ParallelFlipFlopCycleGenerator(ParallelMemoryBitsGenerator):
    @property
    def task_name(self):
        return f'flipflopcycle' + super().task_name

    def get_output_generator(self, task_num):
        if task_num % 2 == 0:
            return self.generate_ff_output
        else:
            return self.generate_wave_output


class ParallelCycleLineGenerator(ParallelMemoryBitsGenerator):
    @property
    def task_name(self):
        return f'cycle_line_' + super().task_name

    def task_num_to_value_type(self, task_num):
        if task_num == 0:
            return 'binary'
        else:
            return 'continuous'

    def get_output_generator(self, task_num):
        if task_num % 2 == 0:
            return self.generate_wave_output
        else:
            return self.generate_ff_output


class ParallelFlipFlopLineGenerator(ParallelMemoryBitsGenerator):
    @property
    def task_name(self):
        return f'flipflop_line_' + super().task_name

    def task_num_to_value_type(self, task_num):
        if task_num == 0:
            return 'binary'
        else:
            return 'continuous'

    def get_output_generator(self, task_num):
        return self.generate_ff_output





class OrthogonalMemoryBitsGenerator(MemoryBitsBaseGenerator):
    @property
    def task_name(self):
        return f'orthogonal_' + super().task_name

    def generate_training_trial(self):
        task_num = self.randomize_bit()
        x_seq = self.generate_random_sequence()
        return self.generate_trial(task_num, x_seq)

    def generate_trial(self, task_num, x_seq):
        xx = np.zeros((self.steps, self.n_inputs))
        yy = np.zeros((self.steps, self.n_outputs))
        xx[:, task_num] = x_seq
        yy[:, task_num] = self.generate_output_sequence(x_seq, self.get_output_generator(task_num))
        # yy[np.isnan(yy[:,task_num]), :] = np.nan
        return xx, yy

    def generate_validation_data(self):
        reps = 100
        n_trials = reps*self.n_bits
        x = np.zeros((n_trials, self.steps, self.n_inputs))
        y = np.zeros((n_trials, self.steps, self.n_outputs))
        idx = 0
        for rep in range(reps):
            x_seq = self.generate_random_sequence(val=True)
            for i in range(self.n_bits):
                x[idx], y[idx] = self.generate_trial(task_num=i, x_seq=x_seq)
                idx += 1

        return x, y

    def get_minimal_input(self):
        X = np.zeros((2*self.n_bits, MINIMAL_INPUT_LENGTH, self.n_inputs))
        for bit in range(self.n_bits):
            X[2*bit, :self.delta, bit] = self.vmin
            X[2*bit+1, :self.delta, bit] = self.vmax

        return X

class OrthogonalFlipFlopGenerator(OrthogonalMemoryBitsGenerator):

    @property
    def task_name(self):
        return f'flipflop' + super().task_name

    def get_output_generator(self, task_num):
        return self.generate_ff_output


class OrthogonalLineGenerator(OrthogonalMemoryBitsGenerator):
    def generate_random_sequence(self, value_type='binary', val=False):
        return super().generate_random_sequence(value_type='continuous', val=val)

    @property
    def task_name(self):
        return f'lines' + super().task_name

    def get_output_generator(self, task_num):
        return self.generate_ff_output

    def get_minimal_input(self):
        X = []
        for task_num in range(self.n_bits):
            x = np.zeros((10, MINIMAL_INPUT_LENGTH, self.n_inputs))
            values_range = np.linspace(self.vmin, self.vmax, 10)
            for i in range(10):
                x[i, :self.delta, task_num] = values_range[i]

            X.append(x)

        return np.vstack(X)



class OrthogonalCyclesGenerator(OrthogonalMemoryBitsGenerator):
    @property
    def task_name(self):
        return f'cycles' + super().task_name

    def get_output_generator(self, task_num):
        return self.generate_wave_output


class OrthogonalFlipFlopCycleGenerator(OrthogonalMemoryBitsGenerator):
    @property
    def task_name(self):
        return f'flipflopcycle' + super().task_name

    def get_output_generator(self, task_num):
        if task_num % 2 == 0:
            return self.generate_ff_output
        else:
            return self.generate_wave_output

class OrthogonalCycleLineGenerator(OrthogonalMemoryBitsGenerator):
    @property
    def task_name(self):
        return f'cycle_line_' + super().task_name

    def generate_random_sequence(self, value_type='binary', val=False):
        return super().generate_random_sequence(value_type=value_type, val=val)

    def task_num_to_value_type(self, task_num):
        if task_num == 0:
            return 'binary'
        else:
            return 'continuous'

    def get_output_generator(self, task_num):
        if task_num % 2 == 0:
            return self.generate_wave_output
        else:
            return self.generate_ff_output

    def generate_training_trial(self):
        task_num = np.random.randint(self.n_bits)
        x_seq = self.generate_random_sequence(value_type=self.task_num_to_value_type(task_num))
        return self.generate_trial(task_num, x_seq)

    def generate_validation_data(self):
        reps = 20
        n_trials = reps*self.n_bits
        x = np.zeros((n_trials, self.steps, self.n_inputs))
        y = np.zeros((n_trials, self.steps, self.n_outputs))
        idx = 0
        for rep in range(reps):
            for task_num in range(self.n_bits):
                x_seq = self.generate_random_sequence(value_type=self.task_num_to_value_type(task_num), val=True)
                x[idx], y[idx] = self.generate_trial(task_num=task_num, x_seq=x_seq)
                idx += 1

        return x, y

    def get_minimal_input(self):
        X = np.zeros((2 + 10, MINIMAL_INPUT_LENGTH, self.n_inputs))
        X[0, :self.delta, 0] = self.vmin
        X[1, :self.delta, 0] = self.vmax

        values_range = np.linspace(self.vmin , self.vmax , 10)
        for i in range(10):
            X[i+2, :self.delta, 1] = values_range[i]

        return X


class OrthogonalFlipFlopLineGenerator(OrthogonalMemoryBitsGenerator):
    @property
    def task_name(self):
        return f'flipflop_line_' + super().task_name

    def generate_random_sequence(self, value_type='binary', val=False):
        return super().generate_random_sequence(value_type=value_type, val=val)

    def task_num_to_value_type(self, task_num):
        if task_num == 0:
            return 'binary'
        else:
            return 'continuous'

    def get_output_generator(self, task_num):
        return self.generate_ff_output

    def generate_training_trial(self):
        task_num = np.random.randint(self.n_bits)
        x_seq = self.generate_random_sequence(value_type=self.task_num_to_value_type(task_num))
        return self.generate_trial(task_num, x_seq)

    def generate_validation_data(self):
        reps = 20
        n_trials = reps*self.n_bits
        x = np.zeros((n_trials, self.steps, self.n_inputs))
        y = np.zeros((n_trials, self.steps, self.n_outputs))
        idx = 0
        for rep in range(reps):
            for task_num in range(self.n_bits):
                x_seq = self.generate_random_sequence(value_type=self.task_num_to_value_type(task_num), val=True)
                x[idx], y[idx] = self.generate_trial(task_num=task_num, x_seq=x_seq)
                idx += 1

        return x, y

    def get_minimal_input(self):
        X = np.zeros((2 + 10, MINIMAL_INPUT_LENGTH, self.n_inputs))
        X[0, :self.delta, 0] = self.vmin
        X[1, :self.delta, 0] = self.vmax

        values_range = np.linspace(self.vmin , self.vmax , 10)
        for i in range(10):
            X[i+2, :self.delta, 1] = values_range[i]

        return X



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # #
    # plt.clf()
    # plt.plot(x[1])
    # plt.plot(y[1])
    # plt.show()
    # plt.clf()
    # plt.plot(x[2])
    # plt.plot(y[2])
    # plt.show()
    # plt.clf()
    # plt.plot(x[3])
    # plt.plot(y[3])
# # for gen in [RingGenerator, SineWaveGenerator]:
# # for gen in [FlipFlopGenerator, CyclesGenerator, FlipFlopCycleGenerator, ParallelFlipFlopGenerator, ParallelCyclesGenerator, ParallelFlipFlopCycleGenerator, OrthogonalFlipFlopGenerator, OrthogonalCyclesGenerator, OrthogonalFlipFlopCycleGenerator]:
#     gen = gen()
#     print(gen.name)
#     x, y = gen.data
#     fig, axes = plt.subplots(4, 1)
#     axes[0].plot(x[0])
#     axes[1].plot(y[0])
#     # axes[2].plot(x[1])
#     # axes[3].plot(y[1])
#     plt.show()
