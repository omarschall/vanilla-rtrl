import numpy as np
from gen_data.Task import Task

class Context_Dependent_Integration_Task(Task):
    """Class for the 1-dimensional continuous context-dependent integration task."""

    def __init__(self, T_trial,  input_var=1, sensitivity=0.4, a=0.025,
                 c_values=[-0.512, -0.256, -0.128, -0.064, -0.032,
                           0, 0.032, 0.064, 0.128, 0.256, 0.512],
                 B=None):
        """Later

        Args:
            """

        n_in = 4
        n_out = 2

        super().__init__(n_in, n_out)

        self.T_trial = T_trial
        self.c_values = c_values
        self.input_var = input_var
        self.sensitivity = sensitivity
        self.a = a
        self.B = B
        self.probe_inputs = [np.eye(4)[0], np.eye(4)[1], -np.eye(4)[0], -np.eye(4)[1],
                             np.eye(4)[2], np.eye(4)[3]]
        self.probe_dataset = self.gen_probe_dataset()

    def gen_probe_dataset(self):
        #Generate probe dataset
        X = []
        Y = []
        trial_type = []
        trial_switch = []

        #c_values_ds = self.c_values[::2]
        for context in [0, 1]:
            for c_motion, c_color in zip(self.c_values, self.c_values):

                #Motion information
                mu_motion = self.sensitivity * c_motion
                x_motion = np.random.normal(mu_motion, 0, self.T_trial)

                #Color information
                mu_color= self.sensitivity * c_color
                x_color = np.random.normal(mu_color, 0, self.T_trial)

                if context == 0:
                    y_trial = self.a * np.cumsum(x_motion)
                    c0 = np.ones(self.T_trial)
                    c1 = np.zeros(self.T_trial)
                if context == 1:
                    y_trial = self.a * np.cumsum(x_color)
                    c0 = np.zeros(self.T_trial)
                    c1 = np.ones(self.T_trial)

                if self.B is not None:
                    y_trial = np.maximum(y_trial, -self.B)
                    y_trial = np.minimum(y_trial, self.B)
                X.append(np.array([x_motion, x_color, c0, c1]).T)
                Y.append(np.array([y_trial, np.zeros_like(y_trial)]).T)

                trial_type_ = np.tile(np.array([c_motion, c_color, context]), self.T_trial).reshape(-1, 3)
                trial_type.append(trial_type_)

                trial_switch_ = np.zeros(self.T_trial)
                trial_switch_[0] = 1
                trial_switch.append(trial_switch_)

        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)
        trial_type = np.concatenate(trial_type, axis=0)
        trial_switch = np.concatenate(trial_switch, axis=0)

        probe_dataset = {'X': X,
                         'Y': Y,
                         'trial_type': trial_type,
                         'trial_switch': trial_switch,
                         'loss_mask': None}

        return probe_dataset


    def gen_dataset(self, N):

        N_trials = N // self.T_trial

        X = []
        Y = []
        trial_type = []
        trial_switch = []

        for i_trial in range(N_trials):

            #Motion information
            c_motion = np.random.choice(self.c_values)
            mu_motion = self.sensitivity * c_motion
            x_motion = np.random.normal(mu_motion, self.input_var, self.T_trial)

            #Color information
            c_color = np.random.choice(self.c_values)
            mu_color= self.sensitivity * c_color
            x_color = np.random.normal(mu_color, self.input_var, self.T_trial)

            #Pick context
            context = np.random.choice([0, 1])

            if context == 0:
                y_trial = self.a * np.cumsum(x_motion)
                c0 = np.ones(self.T_trial)
                c1 = np.zeros(self.T_trial)
            if context == 1:
                y_trial = self.a * np.cumsum(x_color)
                c0 = np.zeros(self.T_trial)
                c1 = np.ones(self.T_trial)

            if self.B is not None:
                y_trial = np.maximum(y_trial, -self.B)
                y_trial = np.minimum(y_trial, self.B)
            X.append(np.array([x_motion, x_color, c0, c1]).T)
            Y.append(np.array([y_trial, np.zeros_like(y_trial)]).T)

            trial_type_ = np.tile(np.array([c_motion, c_color, context]), self.T_trial).reshape(-1, 3)
            trial_type.append(trial_type_)

            trial_switch_ = np.zeros(self.T_trial)
            trial_switch_[0] = 1
            trial_switch.append(trial_switch_)

        try:
            X = np.concatenate(X, axis=0)
            Y = np.concatenate(Y, axis=0)
            trial_type = np.concatenate(trial_type, axis=0)
            trial_switch = np.concatenate(trial_switch, axis=0)
        except ValueError:
            X = None
            Y = None
            trial_type = None
            trial_switch = None

        return X, Y, trial_type, trial_switch, None

class Context_Dependent_Decision_Task(Task):
    """Class for the 1-dimensional continuous context-dependent integration task."""

    def __init__(self, T_trial,  input_var=1, sensitivity=0.4,
                 c_values=[-0.512, -0.256, -0.128, -0.064, -0.032,
                           0.032, 0.064, 0.128, 0.256, 0.512],
                 fixation_steps=5, report_steps=5, output_scale=1,
                 report_cue=False, T_context=None, fixed_context=None,
                 tonic_report=False):
        """Later

        Args:
            """

        n_in = 4 + int(report_cue)
        n_out = 2

        super().__init__(n_in, n_out)

        self.T_trial = T_trial
        self.c_values = c_values
        self.input_var = input_var
        self.sensitivity = sensitivity
        self.fixation_steps = fixation_steps
        self.report_steps = report_steps
        self.report_cue = report_cue
        self.output_scale = output_scale
        self.T_context = T_context
        self.fixed_context = fixed_context
        self.probe_inputs = [np.eye(n_in)[0], np.eye(n_in)[1], -np.eye(n_in)[0], -np.eye(n_in)[1],
                             np.eye(n_in)[2], np.eye(n_in)[3]]
        self.tonic_report = tonic_report
        if report_cue:
            self.probe_inputs += [np.eye(n_in)[4]]
        self.probe_dataset = self.gen_probe_dataset()

    def gen_probe_dataset(self):
        #Generate probe dataset
        X = []
        Y = []
        trial_type = []
        trial_switch = []

        #c_values_ds = self.c_values[::2]
        for context in [0, 1]:
            for c_motion, c_color in zip(self.c_values, self.c_values):

                #Motion information
                mu_motion = self.sensitivity * c_motion
                if context == 1:
                    mu_motion = 0
                x_motion = np.random.normal(mu_motion, 0, self.T_trial)

                #Color information
                mu_color= self.sensitivity * c_color
                if context == 0:
                    mu_color = 0
                x_color = np.random.normal(mu_color, 0, self.T_trial)

                #Report cue time
                t_low = self.T_trial // 2
                t_high = self.T_trial - self.report_steps
                #t_report_cue = np.random.randint(t_low, t_high)
                t_report_cue = t_high - 1
                if self.report_cue:
                    cue = np.zeros(self.T_trial)
                    cue[t_report_cue:t_report_cue + self.report_steps] = 1

                # Set output and context inputs depending on context outcome
                y_trial = np.zeros(self.T_trial)
                if context == 0:
                    if self.report_cue:
                        if self.tonic_report:
                            y_trial[t_report_cue+1:] = self.output_scale * np.sign(c_motion)
                        else:
                            y_trial[t_report_cue+1:t_report_cue+self.report_steps] = self.output_scale * np.sign(c_motion)
                    else:
                        y_trial[-self.report_steps:] = self.output_scale * np.sign(c_motion)
                    c0 = np.ones(self.T_trial)
                    if self.T_context is not None:
                        c0[self.T_context:] = 0
                    c1 = np.zeros(self.T_trial)
                if context == 1:
                    if self.report_cue:
                        if self.tonic_report:
                            y_trial[t_report_cue + 1:] = self.output_scale * np.sign(c_color)
                        else:
                            y_trial[t_report_cue + 1:t_report_cue + self.report_steps] = self.output_scale * np.sign(c_color)
                    else:
                        y_trial[-self.report_steps:] = self.output_scale * np.sign(c_color)
                    c0 = np.zeros(self.T_trial)
                    c1 = np.ones(self.T_trial)
                    if self.T_context is not None:
                        c1[self.T_context:] = 0

                if self.report_cue:
                    X.append(np.array([x_motion, x_color, c0, c1, cue]).T)
                else:
                    X.append(np.array([x_motion, x_color, c0, c1]).T)
                Y.append(np.array([y_trial, np.zeros_like(y_trial)]).T)

                trial_type_ = np.tile(np.array([c_motion, c_color, context]), self.T_trial).reshape(-1, 3)
                trial_type.append(trial_type_)

                trial_switch_ = np.zeros(self.T_trial)
                trial_switch_[0] = 1
                trial_switch.append(trial_switch_)

        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)
        trial_type = np.concatenate(trial_type, axis=0)
        trial_switch = np.concatenate(trial_switch, axis=0)

        probe_dataset = {'X': X,
                         'Y': Y,
                         'trial_type': trial_type,
                         'trial_switch': trial_switch,
                         'loss_mask': None}

        return probe_dataset


    def gen_dataset(self, N):

        N_trials = N // self.T_trial

        X = []
        Y = []
        trial_type = []
        trial_switch = []
        loss_mask = []

        for i_trial in range(N_trials):

            #Motion information
            c_motion = np.random.choice(self.c_values)
            mu_motion = self.sensitivity * c_motion
            x_motion = np.random.normal(mu_motion, self.input_var, self.T_trial)

            #Color information
            c_color = np.random.choice(self.c_values)
            mu_color= self.sensitivity * c_color
            x_color = np.random.normal(mu_color, self.input_var, self.T_trial)

            #Pick context
            if self.fixed_context is None:
                context = np.random.choice([0, 1])
            else:
                context = self.fixed_context

            # Report cue time
            t_low = self.T_trial // 2
            t_high = self.T_trial - self.report_steps
            t_report_cue = np.random.randint(t_low, t_high)
            if self.report_cue:
                cue = np.zeros(self.T_trial)
                cue[t_report_cue:t_report_cue + self.report_steps] = 1

            #Set output and context inputs depending on context outcome
            y_trial = np.zeros(self.T_trial)
            if context == 0:
                if self.report_cue:
                    if self.tonic_report:
                        y_trial[t_report_cue + 1:] = self.output_scale * np.sign(c_motion)
                    else:
                        y_trial[t_report_cue + 1:t_report_cue + self.report_steps] = self.output_scale * np.sign(c_motion)
                else:
                    y_trial[-self.report_steps:] = self.output_scale * np.sign(c_motion)
                c0 = np.ones(self.T_trial)
                if self.T_context is not None:
                    c0[self.T_context:] = 0
                c1 = np.zeros(self.T_trial)
            if context == 1:
                if self.report_cue:
                    if self.tonic_report:
                        y_trial[t_report_cue + 1:] = self.output_scale * np.sign(c_color)
                    else:
                        y_trial[t_report_cue + 1:t_report_cue + self.report_steps] = self.output_scale * np.sign(c_color)
                else:
                    y_trial[-self.report_steps:] = self.output_scale * np.sign(c_color)
                c0 = np.zeros(self.T_trial)
                c1 = np.ones(self.T_trial)
                if self.T_context is not None:
                    c1[self.T_context:] = 0

            if self.report_cue:
                X.append(np.array([x_motion, x_color, c0, c1, cue]).T)
            else:
                X.append(np.array([x_motion, x_color, c0, c1]).T)
            Y.append(np.array([y_trial, np.zeros_like(y_trial)]).T)

            trial_type_ = np.tile(np.array([c_motion, c_color, context]), self.T_trial).reshape(-1, 3)
            trial_type.append(trial_type_)

            trial_switch_ = np.zeros(self.T_trial)
            trial_switch_[0] = 1
            trial_switch.append(trial_switch_)

            loss_mask_ = np.ones((self.T_trial, 2))
            loss_mask_[self.fixation_steps:-self.report_steps] = np.array([0, 1])
            loss_mask.append(loss_mask_)

        try:
            X = np.concatenate(X, axis=0)
            Y = np.concatenate(Y, axis=0)
            trial_type = np.concatenate(trial_type, axis=0)
            trial_switch = np.concatenate(trial_switch, axis=0)
            loss_mask = np.concatenate(loss_mask, axis=0)
        except ValueError:
            X = None
            Y = None
            trial_type = None
            trial_switch = None
            loss_mask = None

        return X, Y, trial_type, trial_switch, loss_mask