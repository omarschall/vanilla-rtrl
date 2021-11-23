from learning_algorithms.Learning_Algorithm import Learning_Algorithm
import numpy as np
from functions import *

class RFLO(Learning_Algorithm):
    """Implements the Random-Feedback Local Online learning algorithm (RFLO)
    from Murray 2019.

    Maintains an eligibility trace B that is updated by temporally filtering
    the immediate influences \phi'(h_i) a_hat_j by the network's inverse time
    constant \alpha:

    B'_{ij} = (1 - \alpha) B_{ij} + \alpha \phi'(h_i) a_hat_j       (1)

    Eq. (1) is implemented by update_learning_vars method. Gradients are then
    calculated according to

    q_i B_{ij}      (2)

    which is implemented in get_rec_grads."""


    def __init__(self, rnn, alpha, **kwargs):
        """Inits an RFLO instance by specifying the inverse time constant for
        the eligibility trace.

        Args:
            alpha (float): Float between 0 and 1 specifying the inverse time
                constant of the eligilibility trace, typically chosen to be
                equal to alpha for the network.

        Keyword args:
            B (numpy array): Initial value for B (all 0s if unspecified)."""

        self.name = 'RFLO'
        allowed_kwargs_ = {'B'}
        super().__init__(rnn, allowed_kwargs_, **kwargs)

        self.alpha = alpha
        if self.B is None:
            if self.rnn.type == 'rnn':
                self.B = np.zeros((self.n_h, self.m))
            elif self.rnn.type == 'gru':
                self.B = np.zeros((3*self.n_h, self.m))

    def update_learning_vars(self):
        """Updates B by one time step of temporal filtration via the invesre
        time constant alpha (see Eq. 1)."""

        #Get relevant values and derivatives from network
        self.a_hat = np.concatenate([self.rnn.a_prev,
                                     self.rnn.x,
                                     np.array([1])])
        if self.rnn.type == 'gru':
            D = self.rnn.activation.f_prime(self.rnn.h)
            Dzz = self.alpha * self.rnn.activation.f_prime(self.rnn.h) * sigmoid.f_prime(self.rnn.zz)
            Dr = self.alpha * self.rnn.activation.f_prime(self.rnn.h) * sigmoid.f_prime(self.rnn.zz) * sigmoid.f_prime(
                self.rnn.r)
            self.M_immediate = self.alpha * np.multiply.outer(np.concatenate([Dzz, Dr, D]), self.a_hat)
            #Update eligibility traces
            self.B = (1 - self.alpha) * self.B + self.M_immediate
        elif self.rnn.type == 'rnn':
            self.D = self.rnn.activation.f_prime(self.rnn.h)
            self.M_immediate = self.alpha * np.multiply.outer(self.D, self.a_hat)

            # Update eligibility traces
            self.B = (1 - self.alpha) * self.B + self.M_immediate


    def get_rec_grads(self):
        """Implements Eq. (2) from above."""

        return (self.q * self.B.T).T

    def reset_learning(self):
        """Reset eligibility trace to 0."""

        self.B *= 0
