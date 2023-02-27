import numpy as np
from gen_data.Task import Task
from copy import copy

class Binary_Addition_Task(Task):
    """ATTEMPT AT WILL TONG TASK"""

    def __init__(self, max_args=3, max_binary_digits=3, max_noops=5,
                 T_final_answer=5, output_style='analog', T_probe=50):
        """Not sure what args should be."""

        #Initialize a parent Task object with 4 input and 2 output dimensions.
        if output_style == 'analog':
            n_out = 2
        elif output_style == 'one_hot':
            n_out = max_args * (2 ** max_binary_digits - 1) + 1
            if (T_probe // 2) > n_out:
                T_probe = n_out * 2
        super().__init__(4, n_out)

        self.max_args = max_args
        self.max_binary_digits = max_binary_digits
        self.max_noops = max_noops
        self.T_final_answer = T_final_answer
        self.output_style = output_style
        self.probe_inputs = [np.eye(4)[i] for i in range(4)]
        self.probe_dataset = self.gen_probe_dataset(T_probe)

    def gen_probe_dataset(self, T_probe):

        #Temporarily change instance parameters
        original_max_noops = copy(self.max_noops)
        original_T_final_answer = copy(self.T_final_answer)
        self.max_noops = 0
        self.T_final_answer = 0

        args = [1] * (T_probe // 2)
        x_trial, y_trial = self.symbolic_to_one_hot(args, np.cumsum(args))
        trial_switch = np.zeros(x_trial.shape[0])
        trial_switch[-1] = 1

        probe_dataset = {'X': x_trial,
                         'Y': y_trial,
                         'trial_type': None,
                         'trial_switch': trial_switch,
                         'loss_mask': None}

        #Reset original parameters
        self.max_noops = original_max_noops
        self.T_final_answer = original_T_final_answer

        return probe_dataset

    def gen_dataset(self, N):

        n = 0
        X = []
        Y = []
        trial_switch = []
        while n < N:

            args, partial_sums = self.generate_random_symbolic_sequence()
            x_trial, y_trial = self.symbolic_to_one_hot(args, partial_sums)
            t_trial = x_trial.shape[0]
            trial_switch_ = np.zeros(t_trial).astype(np.int)
            trial_switch_[-1] = 1
            n += t_trial

            X.append(x_trial)
            Y.append(y_trial)
            trial_switch.append(trial_switch_)

        try:
            X = np.concatenate(X, axis=0)
            Y = np.concatenate(Y, axis=0)
            trial_switch = np.concatenate(trial_switch, axis=0)
        except ValueError:
            return None, None, None, None, None

        return X, Y, None, trial_switch, None

    def generate_random_symbolic_sequence(self, n_args=None):
        """Generates a random addition sequence for either given or random
        number of args. Randomly inserts in noops where approrpiate."""

        if n_args is None:
            n_args_ = np.random.randint(1, self.max_args + 1)
        else:
            n_args_ = n_args

        args = [np.random.randint(0, 2**self.max_binary_digits)
                for _ in range(n_args_)]

        partial_sums = np.cumsum(args)

        return args, partial_sums

    def symbolic_to_one_hot(self, arg_sequence, partial_sums):
        """We represent a symbolic sequence as a list of one-hots, which can
        represent the values '0', '1', '+', or 'noop'.

        Returns both the input sequence and the sequence of partial sums."""

        I = np.eye(4)
        n_args = len(arg_sequence)
        zero_pad_ps = np.concatenate([np.array([0]), partial_sums])
        one_hot_seq = []
        expanded_partial_sums = []
        for i_arg, arg in enumerate(arg_sequence):

            brep = [I[int(d)] for d in np.binary_repr(arg)]
            one_hot_seq += brep
            #from pdb import set_trace
            #set_trace()
            expanded_partial_sums += ([zero_pad_ps[i_arg]] * len(brep))
            if i_arg < n_args - 1:

                n_noops = np.random.randint(0, self.max_noops + 1)
                n_repeats = n_noops + 1
                one_hot_seq += ([I[2]] * n_noops)
                one_hot_seq += [I[3]]
            else:
                n_repeats = self.T_final_answer
                one_hot_seq += ([I[2]] * self.T_final_answer)

            expanded_partial_sums += ([zero_pad_ps[i_arg + 1]] * n_repeats)

        x_trial = np.array(one_hot_seq)
        if self.output_style == 'analog':
            y_trial = np.vstack([expanded_partial_sums,
                                 expanded_partial_sums]).T
        elif self.output_style == 'one_hot':
            I = np.eye(self.n_out)
            y_trial = np.array([I[s] for s in expanded_partial_sums])

        return x_trial, y_trial


