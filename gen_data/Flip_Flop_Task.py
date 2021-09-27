import numpy as np
from gen_data.Task import Task

class Flip_Flop_Task(Task):
    """Class for the N-bit flip-flop task.

    For n independent dimensions, an input stream of 0, -1 and 1 is provided,
    and the output is persistently either -1 or 1, flipping to the other only
    if the corresponding input is the opposite. Most inputs are 0, as
    specified by the "p_flip" parameter."""

    def __init__(self, n_bit, p_flip, tau_task=1, p_context_flip=None,
                 input_magnitudes=None,
                 x_dim_mask=[1,1,1], y_dim_mask=[1,1,1]):
        """Initiates an instance of the n-bit flip flop task by specifying the
        probability of a nonzero input and timescale of the task.

        Args:
            n_bit (int): The number of independent task dimensions.
            p_flip (float): The probability of an input being nonzero.
            tau_task (int): The factor by which we temporally "stretch" the task
                (similar to Add Task)."""

        self.n_bit = n_bit

        n_in = np.maximum(n_bit, 2)
        n_out = n_in

        if p_context_flip is not None:
            n_in += 1

        super().__init__(n_in, n_out)


        self.p_flip = p_flip
        self.tau_task = tau_task
        self.p_context_flip = p_context_flip
        self.input_magnitudes = input_magnitudes
        self.x_dim_mask = np.array(x_dim_mask)
        self.y_dim_mask = np.array(y_dim_mask)
        self.probe_inputs = ([np.eye(n_bit)[i] for i in range(n_bit)] +
                             [-np.eye(n_bit)[i] for i in range(n_bit)])

    def gen_dataset(self, N):
        """Generates a dataset for the flip-flop task."""

        N = N // self.tau_task

        if N == 0:
            return np.array([]), np.array([])

        probability = [self.p_flip / 2, 1 - self.p_flip, self.p_flip / 2]
        choices = [-1, 0, 1]
        X = np.random.choice(choices, size=(N, self.n_bit), p=probability)
        X = np.tile(X, self.tau_task).reshape((self.tau_task*N, self.n_bit))
        Y = X.copy()
        for k in range(int(np.ceil(np.log2(N)))):
            Y[2 ** k:] = np.sign(Y[2 ** k:] + Y[:-2 ** k] / 2)

        if self.input_magnitudes is not None:
            mags = np.random.choice(self.input_magnitudes, size=X.shape)
            X = X * mags

        if self.n_bit == 1:
            X = np.tile(X, 2)
            Y = np.tile(Y, 2)

        if self.p_context_flip is not None:
            x_context = []
            init_context = np.random.choice([-1, 1])
            while len(x_context) < N:
                n_same_context = np.random.geometric(self.p_context_flip)
                x_context += ([init_context] * n_same_context)

                init_context *= -1
            x_context = np.array(x_context[:N]).reshape(-1, 1)

            X = np.concatenate([X, x_context], axis=1)

            X[np.where(x_context == -1), :-1] *= -1

        #Mask any dimensions
        X = X * self.x_dim_mask
        Y = Y * self.y_dim_mask

        return X, Y, None
