import numpy as np
from gen_data.Task import Task

class Discrete_Integration_Task(Task):
    """Class for the 1-dimensional discrete integration task."""

    def __init__(self, p_bit=0.05, p_reset=0.005, tau_task=1,
                 reset_mode='random'):
        """Later

        Args:
            p_bit (float): The probability of integration input being nonzero.
            p_reset (float): The probability of the integration being reset."""

        n_in = 2
        n_out = 2

        super().__init__(n_in, n_out)

        self.p_bit = p_bit
        self.p_reset = p_reset
        self.tau_task = tau_task
        self.reset_mode = reset_mode
        self.probe_inputs = [np.array([1, 0]),
                             np.array([-1, 0]),
                             np.array([0, 1])]

    def gen_dataset(self, N):

        #N = N // self.tau_task

        probability = [self.p_bit / 2, 1 - self.p_bit, self.p_bit / 2]
        choices = [-1, 0, 1]
        x_bits = np.random.choice(choices, size=N, p=probability)

        if self.reset_mode == 'random':
            x_resets = np.random.binomial(1, self.p_reset, size=N)
        if self.reset_mode == 'regular':
            x_resets = np.zeros_like(x_bits)
            period = int(1 / self.p_reset)
            x_resets[::period] = 1

        X = np.array([x_bits, x_resets]).T
        Y = np.zeros_like(X)

        t_resets = list(np.where(x_resets > 0)[0]) + [None]
        t_reset_prev = 0

        for i_t_reset, t_reset in enumerate(t_resets):

            x_interval = x_bits[t_reset_prev:t_reset]
            Y[t_reset_prev:t_reset, 0] = np.sign(np.cumsum(x_interval))

            t_reset_prev = t_reset

        return X, Y, None