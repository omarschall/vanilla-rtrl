from learning_algorithms.Learning_Algorithm import Learning_Algorithm
from utils import *
from functions import *

class Only_Output_Weights(Learning_Algorithm):
    """Updates only the output weights W_out and b_out"""

    def __init__(self, rnn, **kwargs):

        self.name = 'Only_Output_Weights'
        allowed_kwargs_ = set()
        super().__init__(rnn, allowed_kwargs_, **kwargs)

    def update_learning_vars(self):
        """No internal variables to update."""

        pass

    def get_rec_grads(self):
        """Returns all 0s for the recurrent gradients."""

        return np.zeros((self.n_h, self.m))
