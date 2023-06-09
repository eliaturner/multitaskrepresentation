from abc import abstractmethod, ABC

import numpy as np
from cached_property import cached_property


class RNNDataGenerator(ABC):
    @property
    @abstractmethod
    def n_inputs(self):
        pass

    @property
    @abstractmethod
    def n_outputs(self):
        pass

    @property
    @abstractmethod
    def steps(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def task_name(self):
        pass

    @abstractmethod
    def generate_train_data(self):
        pass

    @abstractmethod
    def generate_validation_data(self):
        pass

    def preprocess_data(self, x):
        return x

    def postprocess_data(self, x):
        return x

    def get_data(self):
        return self.data

    def generate_data(self):
        x = np.zeros((self.n_batch, self.steps, self.n_inputs))
        y = np.zeros((self.n_batch, self.steps, self.n_outputs))
        for i in range(self.n_batch):
            x[i], y[i] = self.generate_trial()

        return x, y

    def predict_on_data(self, model):
        x, y = self.get_data()
        pred = model.run_system_from_input(x)
        x = self.postprocess_data(x)
        y = self.postprocess_data(y)
        outputs = self.postprocess_data(pred['output'])
        states = self.postprocess_data(pred['state'])
        return outputs, states

    @cached_property
    def data(self):
        self.x, self.y = self.generate_validation_data()
        return self.x, self.y

    def initialize_data_placeholder(self, n_samples):
        x = np.zeros([n_samples, self.steps, self.n_inputs])
        y = np.zeros([n_samples, self.steps, self.n_outputs])

        return x, y


class FamilyOfTasksGenerator(RNNDataGenerator):
    def __init__(self, tasks, input_type='multi', output_type='multiple', steps=250, n_tasks_total=None, extra=0,
                 transient_amplitude=1, train_last=None, n_values_training=None, orthogonal=False):
        self.tasks = tasks
        self.n_tasks = len(tasks)
        if n_tasks_total:
            self.n_tasks_total = n_tasks_total
        else:
            self.n_tasks_total = self.n_tasks
        self.input_type = input_type
        self.output_type = output_type  # single, multi, single_phase
        self.n_batch = 1024
        self._steps = steps
        self.n_values = 15
        self.vmin = self.tasks[0].vmin
        self.vmax = self.tasks[0].vmax
        self.validation_range = np.linspace(self.vmin, self.vmax + extra, self.n_values)
        self.n_values_training = n_values_training
        # self.validation_range = np.linspace(self.vmax, self.vmax + extra, self.n_values)
        self.extra = extra
        self.min_delay = 10
        self.max_delay = 40
        self.transient_amplitude = transient_amplitude
        self.delta = 5
        if train_last is None:
            self.train_last = self.n_tasks
        else:
            self.train_last = train_last

        self.orthogonal = orthogonal

    @property
    def task_name(self):
        names = [task.name for task in self.tasks]
        s = '_'.join(names)
        s += ('_' + self.input_type)
        s += ('_' + self.output_type)
        return 'family_' + s

    @property
    def steps(self):
        return self._steps

    @property
    def name(self):
        name = f'{self.task_name}_{self.n_tasks_total}tasks_{self.steps}'
        name += f'_{self.vmin}_{self.vmax}'
        if self.extra:
            name += f'_extra{self.extra}'
        if self.transient_amplitude != 1:
            name += f'_trans{self.transient_amplitude}'
        if self.train_last != self.n_tasks:
            name += f'last{self.train_last}'

        if self.orthogonal:
            name += '_orthogonal'

        return name

    @property
    def n_inputs(self):
        if self.input_type == 'none':
            return 1
        if self.input_type == 'multi':
            return self.n_tasks_total

        return self.n_tasks_total + 1

    @property
    def n_outputs(self):
        if self.input_type == 'none':
            return self.n_tasks

        if self.output_type == 'single':
            return 1
        if self.input_type == 'single_phase':
            return self.n_tasks + 1

        return self.n_tasks_total

    def get_input_channel(self, task_num):
        if self.input_type == 'multi':
            in_channel = task_num
        else:
            in_channel = -1

        return in_channel

    def get_output_channel(self, task_num):
        if self.output_type == 'multiple' or self.output_type == 'all':
            output_channel = task_num
        else:
            output_channel = -1

        return output_channel

    def generate_random_sequence(self, val=False):
        x_random = np.zeros((self.steps))
        t = 0
        while t < len(x_random):
            if val:
                value = np.random.choice(self.validation_range)
                # value = np.random.choice(np.linspace(self.vmin, self.vmax + self.extra, self.n_values))
                delay = self.max_delay + 0
                # delay = int(np.random.choice(np.linspace(self.min_delay, self.max_delay, 20)))
                # delay = np.random.randint(self.min_delay, self.max_delay)

            else:
                if self.n_values_training is None:
                    value = np.random.uniform(self.vmin, self.vmax + self.extra)
                else:
                    value = np.random.choice(np.linspace(self.vmin, self.vmax, self.n_values_training))

                delay = np.random.randint(self.min_delay, self.max_delay)

            x_random[t:t + self.delta] = value
            t += delay + self.delta

        return x_random

    def construct_input(self, task_num, x_seq):
        xi = np.zeros((self.steps, self.n_inputs))
        in_channel = -1
        if self.input_type == 'tonic':
            xi[:, task_num] = 1
        elif self.input_type == 'transient':
            xi[:self.delta, task_num] = self.transient_amplitude
        elif self.input_type == 'multi':
            in_channel = task_num

        xi[:, in_channel] = x_seq
        return xi

    def generate_task_output(self, xi, task_num):
        in_channel = self.get_input_channel(task_num)
        output_channel = self.get_output_channel(task_num)
        yi = np.empty((self.steps, self.n_outputs))

        if self.orthogonal:
            yi[:] = 0
        else:
            yi[:] = np.nan

        transitions = np.argwhere(xi[:, in_channel])[::5].flatten()
        values = [None] + [xi[t, in_channel] for t in transitions]
        transitions = np.append(transitions, self.steps)

        for i in range(len(transitions) - 1):
            interval = transitions[0] if i == 0 else transitions[i] - transitions[i - 1]
            value = self.tasks[task_num].value(xn=values[i + 1], xnm1=values[i], interval=interval)
            yi[transitions[i] + self.delta:transitions[i + 1], output_channel] = value

        yi[np.argwhere(xi[:, in_channel]), output_channel] = np.nan
        if self.output_type == 'all':
            for c in range(self.n_outputs):
                yi[:, c] = yi[:, output_channel]

        return yi

    def generate_all_output(self, xi):
        transitions = np.argwhere(xi[:, -1])[::5].flatten()
        yi = np.zeros((self.steps, self.n_outputs))
        values = [0] + [xi[t, -1] for t in transitions]
        np.append(transitions, self.steps)
        for task_num in range(len(self.tasks)):
            for i in range(len(transitions) - 1):
                value = self.tasks[task_num].value(xn=values[i + 1], xnm1=values[i])
                yi[transitions[i]:transitions[i + 1], task_num] = value

            value = self.tasks[task_num].value(values[-1], values[-2])
            yi[transitions[i + 1]:, task_num] = value

        return yi

    def generate_trial(self, task_num=None, x_seq=None):
        if task_num is None:
            task_num = np.random.randint(self.n_tasks - self.train_last, self.n_tasks)

        if x_seq is None:
            x_seq = self.generate_random_sequence()

        xx = self.construct_input(task_num, x_seq)
        if self.input_type == 'none':
            yy = self.generate_all_output(xx)
        else:
            yy = self.generate_task_output(xx, task_num)

        return xx, yy

    def generate_training_trial(self):
        return self.generate_trial()

    def generate_validation_data(self):
        reps = 100
        if self.input_type is None:
            n_trials = reps
        else:
            n_trials = reps * self.n_tasks

        x = np.zeros((n_trials, self.steps, self.n_inputs))
        y = np.zeros((n_trials, self.steps, self.n_outputs))
        idx = 0
        for rep in range(reps):
            x_seq = self.generate_random_sequence(val=True)
            if self.input_type is None:
                x[idx], y[idx] = self.generate_trial(task_num=-1, x_seq=x_seq)
                idx += 1
            else:
                for i in range(self.n_tasks):
                    x[idx], y[idx] = self.generate_trial(task_num=i, x_seq=x_seq)
                    idx += 1

        return x, y

    def generate_train_data(self):
        return self.generate_data()

    def get_data(self):
        return self.data

    def get_minimal_input(self):
        X = []
        from data.functions import Function1D
        for task_num in range(self.n_tasks):
            if isinstance(self.tasks[task_num], Function1D):
                x = np.zeros((10, self.max_delay + 10, self.n_inputs))
                values_range = np.linspace(self.vmin, self.vmax, 10)
                for i in range(10):
                    x[i, :self.delta, task_num] = values_range[i]

            else:
                x = np.zeros((25, self.max_delay + 10, self.n_inputs))
                values_range = np.linspace(self.vmin, self.vmax, 5)
                i = 0
                for i1 in range(5):
                    for i2 in range(5):
                        x[i, :self.delta, task_num] = values_range[i1]
                        x[i, 20:20 + self.delta, task_num] = values_range[i2]
                        i += 1

            X.append(x)

        return np.vstack(X)


class SimplifiedTask1DFamilyGenerator(FamilyOfTasksGenerator):
    @property
    def name(self):
        return 'simple_' + super().name

    def generate_validation_data(self):
        reps = len(self.validation_range)
        if self.input_type is None:
            n_trials = reps
        else:
            n_trials = reps * self.n_tasks

        x = np.zeros((n_trials, self.steps, self.n_inputs))
        y = np.zeros((n_trials, self.steps, self.n_outputs))
        idx = 0
        for an in self.validation_range:
            for i in range(self.n_tasks):
                x[idx], y[idx] = self.generate_trial(task_num=i, an=an)
                idx += 1

        return x, y

    def generate_trial(self, task_num=None, an=None):
        min_offset, max_offset = 0 * self.delta, 4 * self.delta
        if an is None:
            an = np.random.uniform(self.vmin, self.vmax)
            offset = np.random.randint(min_offset, max_offset)

        else:
            offset = max_offset

        if task_num is None:
            task_num = np.random.randint(self.n_tasks - self.train_last, self.n_tasks)

        xx = np.zeros((self.steps, self.n_inputs))
        yy = np.empty((self.steps, self.n_outputs))
        yy[:] = np.nan
        if self.input_type == 'tonic':
            xx[:, task_num] = self.transient_amplitude
        else:
            xx[:self.delta, task_num] = self.transient_amplitude
        # xx[:self.delta, -1] = an
        xx[offset: offset + self.delta, -1] = an
        offset += 2 * self.delta
        duration = np.random.randint(2 * self.delta, self.steps + 1 - offset)
        # yy[self.delta:, task_num] = self.tasks[task_num].value(xn=an)
        if self.n_outputs == 1:
            yy[offset:offset + duration, 0] = self.tasks[task_num].value(xn=an)
        else:
            yy[offset:offset + duration, task_num] = self.tasks[task_num].value(xn=an)

        return xx, yy

    @property
    def n_inputs(self):
        return self.n_tasks_total + 1
