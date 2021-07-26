from learning_algorithms.Learning_Algorithm import Learning_Algorithm
import numpy as np

class REINFORCE(Learning_Algorithm):
    def __init__(self, rnn, sigma=0, **kwargs):
        """Inits an instance of REINFORCE by specifying the optimizer used to
        train the A and alpha values and a noise standard deviation for the
        perturbations.
        Args:
            optimizer (optimizers.Optimizer): An instance of the Optimizer class
            sigma_noise (float): Standard deviation for the values, sampled
                i.i.d. from a zero-mean Gaussian, used to perturb the network
                state to noisy_rnn and thus estimate target predictions for
                A and alpha.
        Keyword args:
            decay (numpy float): value of decay for the eligibility trace.
                Must be a value between 0 and 1, default is 0, indicating
                no decay.
            loss_decay (numpy float): time constant of the filtered average of
                the activations."""

        self.name = 'REINFORCE'
        allowed_kwargs_ = {'decay', 'loss_decay'}
        super().__init__(rnn, allowed_kwargs_, **kwargs)
        # Initialize learning variables
        if self.decay is None:
            self.decay = 1
        if self.loss_decay is None:
            self.loss_decay = 0.01
        self.e_trace = 0
        self.loss_avg = 0
        self.loss_prev = 0
        self.loss = 0
        self.sigma = sigma

    def update_learning_vars(self):
        """Updates the eligibility traces used for learning"""
        # presynaptic variables/parameters

        self.a_hat = np.concatenate([self.rnn.a_prev,
                                     self.rnn.x,
                                     np.array([1])])
        # postsynaptic variables/parameters
        self.D = self.rnn.activation.f_prime(self.rnn.h) * self.rnn.noise

        # matrix of pre/post activations
        self.e_immediate = np.outer(self.D, self.a_hat) / self.sigma ** 2
        self.e_trace = ((1 - self.decay) * self.e_trace +
                        self.decay * self.e_immediate)
        self.loss_prev = self.loss
        self.loss = self.rnn.loss_
        self.loss_avg = ((1 - self.loss_decay) * self.loss_avg +
                         self.loss_decay * self.loss_prev)

    def get_rec_grads(self):
        """Combine the eligibility trace and the reward to get an estimate
        of the gradient"""
        return (self.loss - self.loss_avg) * self.e_trace
