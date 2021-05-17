from learning_algorithms.Learning_Algorithm import Learning_Algorithm
from utils import *
from functions import *

class Future_BPTT(Learning_Algorithm):
    """Implements the 'F-BPTT' version of backprop we discuss in the paper for
    an RNN.

    Although more expensive than E-BPTT by a factor of the truncation horizon,
    this version covers more 'loss-parmaeter sensitivity' terms in Fig. 3 and
    produces, at each time step, an approximate 'future-facing' gradient up to
    truncation that can be used for comparison with other algorithm's outputs.

    Details of computation are in paper. When a credit assignment estimate is
    calculated, the gradient is ultimately calculated according to

    dL/dW_{ij} = c_i \phi'(h_i) a_hat_j                             (1)."""

    def __init__(self, rnn, T_truncation, **kwargs):
        """Inits an instance of Future_BPTT by specifying the network to
        train and the truncation horizon. No default allowable kwargs."""

        self.name = 'F-BPTT'
        allowed_kwargs_ = set()
        super().__init__(rnn, allowed_kwargs_, **kwargs)

        self.T_truncation = T_truncation

        self.c_history = []
        self.a_hat_history = []
        self.h_history = []

    def update_learning_vars(self):
        """Updates the list of credit assignment vectors according to Section
        4.1.2 in the paper.

        First updates relevant history with latest network variables
        a_hat, h and q. Then backpropagates the latest q to each previous time
        step, adding the result to each previous credit assignment estimate."""

        #Update history
        self.a_hat_history.insert(0, np.concatenate([self.rnn.a_prev,
                                                     self.rnn.x,
                                                     np.array([1])]))
        self.h_history.insert(0, self.rnn.h)
        self.propagate_feedback_to_hidden()
        q = np.copy(self.q)
        #Add immediate credit assignment to front of list
        self.c_history.insert(0, q)

        #Loop over truncation horizon and backpropagate q, pausing along way to
        #update credit assignment estimates
        for i_BPTT in range(1, len(self.c_history)):

            h = self.h_history[i_BPTT - 1]
            J = self.rnn.get_a_jacobian(h=h, update=False)
            q = q.dot(J)
            self.c_history[i_BPTT] += q

    def get_rec_grads(self):
        """Removes the oldest credit assignment value from the c_history list
        and uses it to produce recurrent gradients according to Eq. (1).

        Note: for the first several time steps of the simulation, before
        self.c_history fills up to T_truncation size, 0s are returned for
        the recurrent gradients."""

        if len(self.c_history) >= self.T_truncation:

            #Remove oldest c, h and a_hat from lists
            c = self.c_history.pop(-1)
            h = self.h_history.pop(-1)
            a_hat = self.a_hat_history.pop(-1)

            #Implement Eq. (1)
            D = self.rnn.activation.f_prime(h)
            rec_grads = np.multiply.outer(c * D, a_hat)

        else:

            rec_grads = np.zeros((self.n_h, self.m))

        return rec_grads

    def reset_learning(self):
        """Resets learning by deleting network variable history."""

        self.c_history = []
        self.a_hat_history = []
        self.h_history = []
