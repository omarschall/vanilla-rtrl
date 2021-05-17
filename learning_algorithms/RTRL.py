from learning_algorithms.Learning_Algorithm import Learning_Algorithm
from utils import *
from functions import *

class RTRL(Learning_Algorithm):
    """Implements the Real-Time Recurrent Learning (RTRL) algorithm from
    Williams and Zipser 1989.

    RTRL maintains a long-term "influence matrix" dadw that represents the
    derivative of the hidden state with respect to a flattened vector of
    recurrent update parameters. We concatenate [W_rec, W_in, b_rec] along
    the column axis and order the flattened vector of parameters by stacking
    the columns end-to-end. In other words, w_k = W_{ij} when i = k%n_h and
    j = k//n_h. The influence matrix updates according to the equation

    M' = JM + M_immediate                            (1)

    where J is the network Jacobian and M_immediate is the immediate influence
    of a parameter w on the hidden state a. (See paper for more detailed
    notation.) M_immediate is notated as papw in the code for "partial a partial
    w." For a vanilla network, this can be simply (if inefficiently) computed as
    the Kronecker product of a_hat = [a_prev, x, 1] (a concatenation of the prev
    hidden state, the input, and a constant 1 (for bias)) with the activation
    derivatives organized in a diagonal matrix. The implementation of Eq. (1)
    is in the update_learning_vars method.

    Finally, the algorithm returns recurrent gradients by projecting the
    feedback vector q onto the influence matrix M:

    dL/dw = dL/da da/dw = qM                         (2)

    Eq. (2) is implemented in the get_rec_grads method."""

    def __init__(self, rnn, M_decay=1, **kwargs):
        """Inits an RTRL instance by setting the initial dadw matrix to zero."""

        self.name = 'RTRL' #Algorithm name
        allowed_kwargs_ = set() #No special kwargs for RTRL
        super().__init__(rnn, allowed_kwargs_, **kwargs)

        #Initialize influence matrix
        self.dadw = np.zeros((self.n_h, self.rnn.n_h_params))
        self.M_decay = M_decay

    def update_learning_vars(self):
        """Updates the influence matrix via Eq. (1)."""

        #Get relevant values and derivatives from network.
        self.a_hat = np.concatenate([self.rnn.a_prev,
                                     self.rnn.x,
                                     np.array([1])])
        D = np.diag(self.rnn.activation.f_prime(self.rnn.h))
        self.papw = np.kron(self.a_hat, D) #Calculate M_immediate
        self.rnn.get_a_jacobian() #Get updated network Jacobian

        #Update influence matrix via Eq. (1).
        self.dadw = self.M_decay * self.rnn.a_J.dot(self.dadw) + self.papw

    def get_rec_grads(self):
        """Calculates recurrent grads using Eq. (2), reshapes into original
        matrix form."""

        return self.q.dot(self.dadw).reshape((self.n_h, self.m), order='F')

    def reset_learning(self):
        """Resets learning algorithm by setting influence matrix to 0."""

        self.dadw *= 0
