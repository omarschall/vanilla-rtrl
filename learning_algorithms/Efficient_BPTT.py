from learning_algorithms.Learning_Algorithm import Learning_Algorithm, Learning_Algorithm_For_Embedding
import numpy as np
from utils import norm

class Efficient_BPTT(Learning_Algorithm):
    """Implements the 'E-BPTT' version of backprop we discuss in the paper for
    an RNN.

    We describe in more detail in the paper. In brief, the network activity is
    'unrolled' for T_trunction time steps in non-overlapping intervals. The
    gradient for each interval is computed using the future-facing relation
    from Section 2. Thus 'update_learning_vars' is called at every step to
    update the memory of relevant network variables, while get_rec_grads only
    returns non-zero elements every T_truncation time steps."""

    def __init__(self, rnn, T_truncation, trial_based_truncation=False,
                 **kwargs):
        """Inits an instance of Efficient_BPTT by specifying the network to
        train and the truncation horizon. No default allowable kwargs."""

        self.name = 'E-BPTT'
        allowed_kwargs_ = {'c_clip_norm'}
        super().__init__(rnn, allowed_kwargs_, **kwargs)

        self.T_truncation = T_truncation
        self.trial_based_truncation = trial_based_truncation
        if self.trial_based_truncation:
            self.T_truncation = 0
            self.compute_gradient = False

        # Initialize lists for storing network data
        self.a_hat_history = []
        self.h_history = []
        self.q_history = []

    def update_learning_vars(self):
        """Updates the memory of the algorithm with the relevant network
        variables for running E-BPTT."""

        # Add latest values to list
        self.a_hat_history.insert(0, np.concatenate([self.rnn.a_prev,
                                                     self.rnn.x,
                                                     np.array([1])]))
        self.h_history.insert(0, self.rnn.h)
        self.propagate_feedback_to_hidden()
        self.q_history.insert(0, self.q)

        if hasattr(self, 'compute_embedding_grad'):
            self.p_history.insert(0, self.rnn.p)
            self.x_task_history.insert(0, self.rnn.x_task)

    def get_rec_grads(self):
        """Using the accumulated history of q, h and a_hat values over the
        truncation interval, computes the recurrent gradient.

        Returns:
            rec_grads (numpy array): Array of shape (n_h, m) representing
                the gradient dL/dW after truncation interval completed,
                otherwise an array of 0s of the same shape."""

        # Once a 'triangle' is formed (see Fig. 3 in paper), compute gradient.
        if self.trial_based_truncation:
            compute_condition = self.compute_gradient
        else:
            compute_condition = (len(self.a_hat_history) >= self.T_truncation)

        if compute_condition:

            # Initialize recurrent grads at 0
            rec_grads = np.zeros((self.n_h, self.m))
            # Start with most recent credit assignment value
            c = self.q_history.pop(0)

            if hasattr(self, 'compute_embedding_grad'):
                embedding_grad = np.zeros((self.n_in, self.task_in))

            for i_BPTT in range(self.T_truncation):

                # Truncate credit assignment norm
                if self.c_clip_norm is not None:
                    if norm(c) > self.c_clip_norm:
                        c = c * (self.c_clip_norm / norm(c))

                # Access present values of h and a_hat
                h = self.h_history.pop(0)
                a_hat = self.a_hat_history.pop(0)

                # Use to get gradients w.r.t. weights from credit assignment
                D = self.rnn.alpha * self.rnn.activation.f_prime(h)
                dLdh = c * D
                rec_grads += np.multiply.outer(dLdh, a_hat)

                if hasattr(self, 'compute_embedding_grad'):

                    p = self.p_history.pop(0)
                    x_task = self.x_task_history.pop(0)
                    dLdp = dLdh.dot(self.rnn.W_in) * self.rnn.activation_embed.f_prime(p)
                    embedding_grad += np.multiply.outer(dLdp, x_task)

                if i_BPTT == self.T_truncation - 1:  # Skip if at end
                    continue

                # Use future-facing relation to backpropagate by one time step.
                q = self.q_history.pop(0)
                J = self.rnn.get_a_jacobian(h=h, update=False)
                c = q + c.dot(J)

            if self.trial_based_truncation:
                self.compute_gradient = False
                self.T_truncation = 0

            return rec_grads

        else:

            if self.trial_based_truncation:
                self.T_truncation += 1

            return np.zeros((self.n_h, self.m))

    def reset_learning(self):

        self.compute_gradient = True

class Efficient_BPTT_For_Embedding(Learning_Algorithm_For_Embedding, Efficient_BPTT):

    def __init__(self, T_truncation, trial_based_truncation=False,
                 **kwargs):

        super().__init__(T_truncation, trial_based_truncation, **kwargs)
        self.compute_embedding_grad = True
        self.task_in = self.rnn.n_in
        self.p_history = []
        self.x_task_history = []

    def get_embedding_grad(self):

        pass

