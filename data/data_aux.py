import numpy as np

def generate_ff_output(steps, offset, transitions, values):
    y_channel = np.empty(steps)
    y_channel[:] = np.nan
    for i in range(len(transitions) - 1):
        value = values[i]
        y_channel[transitions[i] + offset:transitions[i + 1]] = value

    return y_channel


def generate_wave_output(steps, offset, transitions, values):
    y_channel = np.empty(steps)
    y_channel[:] = np.nan
    sample_points = 4 * np.pi * np.linspace(0, 1, self.max_delay)
    for i in range(len(transitions) - 1):
        seq = 0.5 * np.sin(sample_points) + values[i]
        length = transitions[i + 1] - transitions[i] - self.offset
        y_channel[transitions[i] + self.offset:transitions[i + 1]] = seq[:length]

    return y_channel


def generate_output_sequence(self, x_seq, output_generator):
    transitions = np.argwhere(x_seq).flatten()[::5]
    values = [x_seq[t] for t in transitions]
    if self.steps - transitions[-1] > self.offset:
        transitions = np.append(transitions, self.steps)

    return output_generator(transitions, values)