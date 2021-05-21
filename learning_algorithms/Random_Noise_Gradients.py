from learning_algorithms.Learning_Algorithm import Learning_Algorithm
import numpy as np

class Random_Noise_Gradients(Learning_Algorithm):

    def __init__(self, rnn, sigma, bias=0, **kwargs):
        """Initializes an instance of learning algorithm by specifying the
        network to be trained, custom allowable kwargs, and kwargs.

        Args:
            rnn (network.RNN): An instance of RNN to be trained by the network.
            allowed_kwargs_ (set): Set of allowed kwargs in addition to those
                common to all child classes of Learning_Algorithm, 'W_FB' and
                'L2_reg'."""

        self.name = 'Noise'
        allowed_kwargs_ = set()
        super().__init__(rnn, allowed_kwargs_, **kwargs)

        self.rnn = rnn
        self.sigma = sigma
        self.bias = bias

    def update_learning_vars(self):
        pass

    def get_rec_grads(self):
        shape = (self.rnn.n_h, self.rnn.n_h + self.rnn.n_in + 1)

        return np.random.normal(self.bias, self.sigma, shape)

