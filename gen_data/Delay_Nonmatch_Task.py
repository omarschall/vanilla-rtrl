import numpy as np
from gen_data.Task import Task

class Delay_Nonmatch_Task(Task):
    """Class for a simple working memory task, where two binary inputs are
    given with a delay, and the network reports whether they match or not.

    In more detail, on a given trial, the input channel are 0 until the first
    stimulus begins, during which one randomly selected input channel turns
    to 1. Then a delay period occurs, in which both channels are 0, until
    a second stimulus arrives. Then another delay period, following a report
    period where the network reports either np.array([1, 0]) or np.array([0, 1])
    for match and non-match, respectively.

    Unlike other tasks, this task has a *trial* structure where the user has
    the option of resetting the network and/or learning algorithm states between
    trials."""

    def __init__(self, t_stim_1=1, stim_duration=3,
                 delay_duration=10, report_duration=3,
                 off_report_loss_factor=0.1):
        """Initializes an instance by specifying, within a trial, the time step
        where the stimulus occurs, the stimulus duration, the delay duration,
        the report duration, and the factor by which the loss is scaled
        for on- vs. off-report time steps.

        Args:
            t_stim (int): The time step when the first stimulus starts
            stim_duration (int): The duration of each stimulus in time steps
            delay_duration (int): The duration of each delay period in time
                steps
            report_duration (int): The report duration in time steps.
            off_report_loss_factor (float): A number between 0 and 1 that scales
                the loss function on time steps outside the report period."""

        super().__init__(2, 2)

        self.t_stim_1 = t_stim_1
        self.stim_duration = stim_duration
        self.delay_duration = delay_duration
        self.report_duration = report_duration
        self.time_steps_per_trial = (t_stim_1
                                     + 2 * stim_duration
                                     + 2 * delay_duration
                                     + report_duration)
        self.t_stim_2 = t_stim_1 + stim_duration + delay_duration
        self.t_report = self.t_stim_2 + stim_duration + delay_duration
        self.off_report_loss_factor = off_report_loss_factor

        #Make mask for preferential learning within task
        self.trial_mask = np.ones(self.time_steps_per_trial)
        self.trial_mask *= self.off_report_loss_factor
        self.trial_mask[self.t_report:self.t_report + self.report_duration] = 1

    def gen_dataset(self, N):
        """Generates a dataset by hardcoding in the four trial types, randomly
        generating a sequence of choices, and concatenating them at the end."""

        X = []
        Y = []
        trial_type = []

        #Trial type 1 AA
        x_1 = np.zeros((self.time_steps_per_trial, 2))
        y_1 = np.zeros((self.time_steps_per_trial, 2))
        x_1[self.t_stim_1:self.t_stim_1 + self.stim_duration, 0] = 1
        x_1[self.t_stim_2:self.t_stim_2 + self.stim_duration, 0] = 1
        y_1[self.t_report:self.t_report + self.report_duration, 0] = 1

        #Trial type 2 AB
        x_2 = np.zeros((self.time_steps_per_trial, 2))
        y_2 = np.zeros((self.time_steps_per_trial, 2))
        x_2[self.t_stim_1:self.t_stim_1 + self.stim_duration, 0] = 1
        x_2[self.t_stim_2:self.t_stim_2 + self.stim_duration, 1] = 1
        y_2[self.t_report:self.t_report + self.report_duration, 1] = 1

        #Trial type 3 BA
        x_3 = np.zeros((self.time_steps_per_trial, 2))
        y_3 = np.zeros((self.time_steps_per_trial, 2))
        x_3[self.t_stim_1:self.t_stim_1 + self.stim_duration, 1] = 1
        x_3[self.t_stim_2:self.t_stim_2 + self.stim_duration, 0] = 1
        y_3[self.t_report:self.t_report + self.report_duration, 1] = 1

        #Trial type 4 BB
        x_4 = np.zeros((self.time_steps_per_trial, 2))
        y_4 = np.zeros((self.time_steps_per_trial, 2))
        x_4[self.t_stim_1:self.t_stim_1 + self.stim_duration, 1] = 1
        x_4[self.t_stim_2:self.t_stim_2 + self.stim_duration, 1] = 1
        y_4[self.t_report:self.t_report + self.report_duration, 0] = 1

        x_trials = [x_1, x_2, x_3, x_4]
        y_trials = [y_1, y_2, y_3, y_4]

        N_trials = N // self.time_steps_per_trial
        for i in range(N_trials):

            trial_type_ = np.random.choice([0, 1, 2, 3])
            trial_type_array = np.ones(self.time_steps_per_trial) * trial_type_
            trial_type.append(trial_type_array.astype(np.int))
            X.append(x_trials[trial_type_])
            Y.append(y_trials[trial_type_])

        if N_trials > 0:
            X = np.concatenate(X, axis=0)
            Y = np.concatenate(Y, axis=0)
            trial_type = np.concatenate(trial_type, axis=0)

        return X, Y, trial_type
