import numpy as np
from gen_data.Task import Task

class Sine_Wave(Task):
    """Class for the sine wave task.

    There are two input dimensions, one of which specifies whether the sine wave
    is "on" (1) or "off" (0). The second dimension specifies the period of
    the sine wave (in time steps) to be produced by the network."""

    def __init__(self, p_transition, periods, never_off=False, **kwargs):
        """Initializes an instance of sine wave task by specifying transition
        probability (between on and off states) and periods to sample from.

        Args:
            p_transition (float): The probability of switching between on and off
                modes.
            periods (list): The sine wave periods to sample from, by default
                uniformly.
            never_off (bool): If true, the is no "off" period, and the network
                simply switches from period to period.
        Keyword args:
            p_periods (list): Must be same length as periods, specifying probability
                for each sine wave period.
            amplitude (float): Amplitude of all sine waves, by default 0.1 if
                not specified.
            method (string): Must be either "random" or "regular", the former for
                transitions randomly sampled according to p_transition and the
                latter for deterministic transitions every 1 / p_transition
                time steps."""

        allowed_kwargs = {'p_periods', 'amplitude', 'method'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to Sine_Wave.__init__: ' + str(k))

        super().__init__(2, 2)

        self.p_transition = p_transition
        self.method = 'random'
        self.amplitude = 0.1
        self.periods = periods
        self.p_periods = np.ones_like(periods) / len(periods)
        self.never_off = never_off
        self.__dict__.update(kwargs)

    def gen_dataset(self, N):
        """Generates a dataset for the sine wave task."""

        X = np.zeros((N, 2))
        Y = np.zeros((N, 2))

        self.switch_cond = False

        active = False
        t = 0
        X[0, 0] = 1
        for i in range(1, N):

            if self.method == 'regular':
                if i % int(1 / self.p_transition) == 0:
                    self.switch_cond = True
            elif self.method == 'random':
                if np.random.rand() < self.p_transition:
                    self.switch_cond = True

            if self.switch_cond:

                t = 0

                if active and not self.never_off:
                    X[i, 0] = 1
                    X[i, 1] = 0
                    Y[i, :] = 0

                if not active or self.never_off:
                    X[i, 0] = np.random.choice(self.periods, p=self.p_periods)
                    X[i, 1] = 1
                    Y[i, 0] = self.amplitude * np.cos(2 * np.pi / X[i, 0] * t)
                    Y[i, 1] = self.amplitude * np.sin(2 * np.pi / X[i, 0] * t)

                active = not active

            else:

                t += 1
                X[i, :] = X[i - 1, :]
                theta = 2 * np.pi / X[i, 0] * t
                on_off = (active or self.never_off)
                Y[i, 0] = self.amplitude * np.cos(theta) * on_off
                Y[i, 1] = self.amplitude * np.sin(theta) * on_off

            self.switch_cond = False

        X[:, 0] = -np.log(X[:, 0])

        return X, Y, None, None, None
