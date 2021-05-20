import numpy as np
from .Task import Task

class Sensorimotor_Mapping(Task):
    """Class for a simple working memory task, where a binary input is given
    to the network, which it must report back some number of time steps later.

    In more detail, on a given trial, the input channel are 0 except during the
    stimulus (a user-defined time window), when it is either 1 or -1; or during
    the report period (another user-defined time window), when it is 1. The
    label channel is always 0 except during the

    Unlike other tasks, this task has a *trial* structure where the user has
    the option of resetting the network and/or learning algorithm states between
    trials."""

    def __init__(self, t_stim=1, stim_duration=3,
                 t_report=20, report_duration=3,
                 off_report_loss_factor=0.1):
        """Initializes an instance by specifying, within a trial, the time step
        where the stimulus occurs, the stimulus duration, the time of the
        report, the report duration, and the factor by which the loss is scaled
        for on- vs. off-report time steps.

        Args:
            t_stim (int): The time step when the stimulus starts
            stim_duration (int): The duration of the stimulus in time steps
            t_report (int): The time step when the report period starts
            report_duration (int): The report duration in time steps.
            off_report_loss_factor (float): A number between 0 and 1 that scales
                the loss function on time steps outside the report period."""

        super().__init__(2, 2)

        self.t_stim = t_stim
        self.stim_duration = stim_duration
        self.t_report = t_report
        self.report_duration = report_duration
        self.time_steps_per_trial = t_report + report_duration
        self.off_report_loss_factor = off_report_loss_factor

        #Make mask for preferential learning within task
        self.trial_mask = np.ones(self.time_steps_per_trial)
        self.trial_mask *= self.off_report_loss_factor
        self.trial_mask[self.t_report:self.t_report + self.report_duration] = 1

    def gen_dataset(self, N):
        """Generates a dataset by hardcoding in the two trial types, randomly
        generating a sequence of choices, and concatenating them at the end."""

        X = []
        Y = []

        #Trial type 1
        x_1 = np.zeros((self.time_steps_per_trial, 2))
        y_1 = np.ones_like(x_1)*0.5
        x_1[self.t_stim:self.t_stim + self.stim_duration, 0] = 1
        x_1[self.t_report:self.t_report + self.report_duration, 1] = 1
        y_1[self.t_report:self.t_report + self.report_duration, 0] = 1
        y_1[self.t_report:self.t_report + self.report_duration, 1] = 0

        #Trial type 2
        x_2 = np.zeros((self.time_steps_per_trial, 2))
        y_2 = np.ones_like(x_2)*0.5
        x_2[self.t_stim:self.t_stim + self.stim_duration, 0] = -1
        x_2[self.t_report:self.t_report + self.report_duration, 1] = 1
        y_2[self.t_report:self.t_report + self.report_duration, 0] = 0
        y_2[self.t_report:self.t_report + self.report_duration, 1] = 1

        x_trials = [x_1, x_2]
        y_trials = [y_1, y_2]

        N_trials = N // self.time_steps_per_trial
        for i in range(N_trials):

            trial_type = np.random.choice([0, 1])
            X.append(x_trials[trial_type])
            Y.append(y_trials[trial_type])

        if N_trials > 0:
            X = np.concatenate(X, axis=0)
            Y = np.concatenate(Y, axis=0)

        return X, Y
