import numpy as np
from gen_data.Task import Task
from math import floor

class Discrete_Integration_Task(Task):
    """Class for the 1-dimensional discrete integration task."""

    def __init__(self, p_bit=0.05, p_reset=0.005, tau_task=1,
                 reset_mode='random', report_count=False,
                 uniform_count_stats=False, max_count=5,
                 max_total_inputs=12):
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
        self.report_count = report_count
        self.probe_inputs = [np.array([1, 0]),
                             np.array([-1, 0]),
                             np.array([0, 1])]
        self.uniform_count_stats = uniform_count_stats
        self.max_count = max_count
        self.max_total_inputs = max_total_inputs

    def gen_dataset_1(self, N):

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

    def get_dataset_2(self, N):

        period = int(1 / self.p_reset)
        N_trials = N // period

        X = []
        Y = []

        for i_trial in range(N_trials):
            final_net_count = np.random.uniform([-self.max_count,
                                                 self.max_count])
            min_positive = np.maximum(0, final_net_count)
            max_positive = 0.5 * floor(self.max_total_inputs + final_net_count)
            final_positive_count = np.random.randint(low=min_positive,
                                                     high=max_positive + 1)
            final_negative_count = final_positive_count - final_net_count

            #Randomly put these positive and negative counts into boxes
            x_trial = np.zeros((period, 2))
            time_indices = list(range(period))
            positive_input_indices = np.random.choice(time_indices,
                                                      size=final_positive_count,
                                                      replace=False)
            time_indices = [t for t in time_indices
                            if t not in positive_input_indices]
            negative_input_indices = np.random.choice(time_indices,
                                                      size=final_negative_count,
                                                      replace=False)

            x_trial[positive_input_indices, 0] = 1
            x_trial[negative_input_indices, 0] = -1
            x_trial[0, 1] = 1

            X.append(x_trial)

            y_trial = np.zeros((period, 2))
            y_trial[:, 0] = np.cumsum(x_trial[:, 0])
            if not self.report_count:
                y_trial[:, 0] = np.sign(y_trial[:, 0])

            Y.append(y_trial)

        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)

        return X, Y, None

    def gen_dataset(self, N):

        if self.uniform_count_stats:
            return self.get_dataset_1(N)
        else:
            return self.gen_dataset_2(N)