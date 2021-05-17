from learning_algorithms.Learning_Algorithm import Learning_Algorithm
from utils import *
from functions import *


class List_of_Gradients(Learning_Algorithm):
    """Simply prescribe a series of updates to the network"""

    def __init__(self, rnn, grads_list_list, allowed_kwargs_=set(), **kwargs):
        """Initializes an instance of learning algorithm by specifying the
        network to be trained, custom allowable kwargs, and kwargs.

        Args:
            rnn (network.RNN): An instance of RNN to be trained by the network.
            allowed_kwargs_ (set): Set of allowed kwargs in addition to those
                common to all child classes of Learning_Algorithm, 'W_FB' and
                'L2_reg'."""

        allowed_kwargs = {}.union(allowed_kwargs_)

        self.rnn = rnn
        self.grads_list_list = grads_list