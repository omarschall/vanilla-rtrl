import numpy as np
from gen_data.Task import Task

class Cts_Integration_Task(Task):
    """Class for the 1-dimensional continuous integration task."""

    def __init__(self, T_trial,  input_var=1, sensitivity=0.4,
                 c_values=[-0.512, -0.256, -0.128, -0.064, -0.032,
                           0, 0.032, 0.064, 0.128, 0.256, 0.512]):
        """Later

        Args:
            """

        n_in = 2
        n_out = 2

        super().__init__(n_in, n_out)

        self.T_trial = T_trial
        self.c_values = c_values
        self.input_var = input_var
        self.sensitivity = sensitivity

    def gen_dataset(self, N):

        N_trials = N // self.T_trial

        X = []
        Y = []
        trial_type = []
        trial_switch = []

        for i_trial in range(N_trials):

            c = np.random.choice(self.c_values)
            mu = self.sensitivity * c
            x_trial = np.random.normal(mu, self.input_var, self.T_trial)
            y_trial = np.cumsum(x_trial)
            X.append(np.array([x_trial, x_trial]).T)
            Y.append(np.array([y_trial, y_trial]).T)

            trial_type_ = c * np.ones_like(x_trial)
            trial_type.append(trial_type_)

            trial_switch_ = np.zeros_like(x_trial)
            trial_switch_[-1] = 1
            trial_switch.append(trial_switch_)

        try:
            X = np.concatenate(X, axis=0)
            Y = np.concatenate(Y, axis=0)
            trial_type = np.concatenate(trial_type)
            trial_switch = np.concatenate(trial_switch, axis=0)
        except ValueError:
            X = None
            Y = None
            trial_type = None
            trial_switch = None

        return X, Y, trial_type, trial_switch, None