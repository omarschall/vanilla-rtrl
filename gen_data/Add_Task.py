import numpy as np
from .Task import Task

class Add_Task(Task):
    """Class for the 'Add Task', an input-label mapping with i.i.d. Bernoulli
    inputs (p=0.5) and labels depending additively on previous inputs at
    t_1 and t_2 time steps ago:

    y(t) = 0.5 + 0.5 * x(t - t_1) - 0.25 * x(t - t_2)           (1)

    as inspired by Pitis 2016
    (https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html).

    The inputs and outputs each have a redundant dimension representing the
    complement of the outcome (i.e. x_1 = 1 - x_0), because keeping all
    dimensions above 1 makes python broadcasting rules easier."""

    def __init__(self, t_1, t_2, deterministic=False, tau_task=1):
        """Initializes an instance of this task by specifying the temporal
        distance of the dependencies, whether to use deterministic labels, and
        the timescale of the changes.

        Args:
            t_1 (int): Number of time steps for first dependency
            t_2 (int): Number of time steps for second dependency
            deterministic (bool): Indicates whether to take the labels as
                the exact numbers in Eq. (1) OR to use those numbers as
                probabilities in Bernoulli outcomes.
            tau_task (int): Factor by which we temporally 'stretch' the task.
                For example, if tau_task = 3, each input (and label) is repeated
                for 3 time steps before being replaced by a new random
                sample."""

        #Initialize a parent Task object with 2 input and 2 output dimensions.
        super().__init__(2, 2)

        #Dependencies in coin task
        self.t_1 = t_1
        self.t_2 = t_2
        self.tau_task = tau_task

        #Use coin flip outputs or deterministic probabilities as labels
        self.deterministic = deterministic

    def gen_dataset(self, N):
        """Generates a dataset according to Eq. (1)."""

        #Generate random bernoulli inputs and labels according to Eq. (1).
        N = N // self.tau_task
        x = np.random.binomial(1, 0.5, N)
        y = 0.5 + 0.5 * np.roll(x, self.t_1) - 0.25 * np.roll(x, self.t_2)
        if not self.deterministic:
            y = np.random.binomial(1, y, N)
        X = np.array([x, 1 - x]).T
        Y = np.array([y, 1 - y]).T

        #Temporally stretch according to the desire timescale of change.
        X = np.tile(X, self.tau_task).reshape((self.tau_task*N, 2))
        Y = np.tile(Y, self.tau_task).reshape((self.tau_task*N, 2))

        return X, Y