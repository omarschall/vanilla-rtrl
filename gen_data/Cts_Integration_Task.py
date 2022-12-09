import numpy as np
from gen_data.Task import Task

class Cts_Integration_Task(Task):
    """Class for the 1-dimensional discrete integration task."""

    def __init__(self):
        """Later

        Args:
            p_bit (float): The probability of integration input being nonzero.
            p_reset (float): The probability of the integration being reset."""

        n_in = 3
        n_out = 2

        super().__init__(n_in, n_out)

        self.p_bit = p_bit
        self.p_reset = p_reset
        self.tau_task = tau_task
        self.reset_mode = reset_mode
        self.report_count = report_count
        self.report_both = report_both
        self.probe_inputs = [np.array([1, 0, 0]),
                             np.array([-1, 0, 0]),
                             np.array([0, 1, 0]),
                             np.array([0, 0, delay_hint])]
        self.uniform_count_stats = uniform_count_stats
        self.max_count = max_count
        self.max_total_inputs = max_total_inputs
        self.delay = delay
        self.BPR_integrate_mask = BPR_integrate_mask
        self.BPR_delay_mask = BPR_delay_mask
        self.BPT_integrate_mask = BPT_integrate_mask
        self.BPT_delay_mask = BPT_delay_mask
        self.delay_hint = delay_hint

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
            counts = np.cumsum(x_interval)
            if self.report_count:
                Y[t_reset_prev:t_reset, 0] = counts
            else:
                Y[t_reset_prev:t_reset, 0] = np.sign(counts)

            t_reset_prev = t_reset

        return X, Y, None