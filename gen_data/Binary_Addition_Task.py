import numpy as np
from gen_data.Task import Task

class Binary_Addition_Task(Task):
    """ATTEMPT AT WILL TONG TASK"""

    def __init__(self, max_args=3, max_binary_digits=3):
        """Not sure what args should be."""

        #Initialize a parent Task object with 4 input and 2 output dimensions.
        super().__init__(4, 2)

        self.max_args = max_args
        self.max_binary_digits = max_binary_digits

    def gen_dataset(self, N):

        pass

    def generate_random_symbolic_sequence(self, n_args=None):
        """Generates a random addition sequence for either given or random
        number of args. Randomly inserts in noops where approrpiate."""

        if n_args is None:
            n_args_ = np.random.randint(1, self.max_args - 1)
        else:
            n_args_ = n_args

        args = [np.random.randint(0, 2**self.max_binary_digits)
                for _ in range(n_args_)]

        answer = sum(args)

        sequence = []

    def symbolic_to_one_hot(self, symbolic_sequence):
        """We represent a symbolic sequence as a list of strings, which can
        take on the values '0', '1', '+', or 'noop'."""

        pass