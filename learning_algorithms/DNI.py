from learning_algorithms.Learning_Algorithm import Learning_Algorithm
from functions import *
import numpy as np

class DNI(Learning_Algorithm):
    """Implements the Decoupled Neural Interface (DNI) algorithm for an RNN from
    Jaderberg et al. 2017.

    Details are in our review, original paper, or Czarnecki et al. 2017.
    Briefly, we linearly approximate the (future-facing) credit assignment
    vector c = dL/da by the variable 'sg' for 'synthetic gradient'
    (of shape (n_h)) using

    c ~ sg = A a_tilde                                                     (1)

    where a_tilde = [a; y^*; 1] is the network state concatenated with the label
    y^* and a constant 1 (for bias), of shape (m_out = n_h + n_in + 1). Then the
    gradient is calculated by combining this estimate with the immediate
    parameter influence \phi'(h) a_hat

    dL/dW_{ij} = dL/da_i da_i/dW_{ij} = sg_i alpha \phi'(h_i) a_hat_j.     (2)

    The matrix A must be updated as well to make sure Eq. (1) is a good
    estimate. Details are in our paper or the original; briefly, a bootstrapped
    credit assignment estimate is formed via

    c^* = q_prev + sg J = q_prev + (A a_tilde_prev) J                      (3)

    where J = da^t/da^{t-1} is the network Jacobian, either calculated exactly
    or approximated (see update_J_approx method). The arrays q_prev and
    a_tilde_prev are q and a_tilde from the previous time step;
    this is because the update requires reference to the "future" (by one time
    step) network state, so we simply wait a time step, such that q and a_tilde
    are now q_prev and a_tilde_prev, respectively. This target for credit
    assignment is then used to calculate a prediction loss gradient for A:

    d/dA_{ij} 0.5 * ||A a_tilde - c^*||^2 = (A a_tilde - c^*)_i a_tilde_j  (4)

    This gradient is used to update A by a given optimizer."""

    def __init__(self, rnn, optimizer, **kwargs):
        """Inits an instance of DNI by specifying the optimizer for the A
        weights and other kwargs.

        Args:
            optimizer (optimizers.Optimizer): An Optimizer instance used to
                update the weights of A based on its credit assignment
                prediction loss gradients.

        Keywords args:
            A (numpy array): Initial value of A weights, must be of shape
                (n_h, m_out). If None, A is initialized with random Gaussian.
            J_lr (float): Learning rate for learning approximate Jacobian.
            activation (functions.Function): Activation function for the
                synthetic gradient function, applied elementwise after A a_tilde
                operation. Default is identity.
            SG_label_activation (functions.Function): Activation function for
                the synthetic gradient function as used in calculating the
                *label* for the
            use_approx_J (bool): If True, trains the network using the
                approximated Jacobian rather than the exact Jacobian.
            SG_L2_reg (float): L2 regularization strength on the A weights, by
                default 0.
            fix_A_interval (int): The number of time steps to wait between
                updating the synthetic gradient method used to bootstrap the
                label estimates. Default is 5."""

        self.name = 'DNI'
        allowed_kwargs_ = {'A', 'J_lr', 'activation', 'SG_label_activation',
                           'use_approx_J', 'SG_L2_reg', 'fix_A_interval'}
        #Default parameters
        self.optimizer = optimizer
        self.SG_L2_reg = 0
        self.fix_A_interval = 5
        self.activation = identity
        self.SG_label_activation = identity
        self.use_approx_J = False
        #Override defaults with kwargs
        super().__init__(rnn, allowed_kwargs_, **kwargs)

        self.m_out = self.n_h + self.n_out + 1
        self.J_approx = np.copy(self.rnn.W_rec)
        self.i_fix = 0
        if self.A is None:
            self.A = np.random.normal(0, np.sqrt(1/self.m_out),
                                      (self.n_h, self.m_out))
        self.A_ = np.copy(self.A)

    def update_learning_vars(self):
        """Updates the A matrix by Eqs. (3) and (4)."""

        if self.use_approx_J: #If using approximate Jacobian, update it.
            self.update_J_approx()
        else: #Otherwise get the exact Jacobian.
            self.rnn.get_a_jacobian()

        #Compute synthetic gradient estimate of credit assignment at previous
        #time step. This is NOT used to drive learning in W but rather to drive
        #learning in A.
        self.a_tilde_prev = np.concatenate([self.rnn.a_prev,
                                            self.rnn.y_prev,
                                            np.array([1])])
        self.sg = self.synthetic_grad(self.a_tilde_prev)

        #Compute the target, error and loss for the synthetic gradient function
        self.sg_target = self.get_sg_target()
        self.A_error = self.sg - self.sg_target
        self.A_loss = 0.5 * np.square(self.A_error).mean()

        #Compute gradients for A
        self.scaled_A_error = self.A_error * self.activation.f_prime(self.sg_h)
        self.A_grad = np.multiply.outer(self.scaled_A_error, self.a_tilde_prev)

        #Apply L2 regularization to A
        if self.SG_L2_reg > 0:
            self.A_grad += self.SG_L2_reg * self.A

        #Update synthetic gradient parameters
        self.A = self.optimizer.get_updated_params([self.A], [self.A_grad])[0]

        #On interval determined by self.fix_A_interval, update A_, the values
        #used to calculate the target in Eq. (3), with the latest value of A.
        if self.i_fix == self.fix_A_interval - 1:
            self.i_fix = 0
            self.A_ = np.copy(self.A)
        else:
            self.i_fix += 1

    def get_sg_target(self):
        """Function for generating the target for training A. Implements Eq. (3)
        using a different set of weights A_, which are static and only
        re-assigned to A  every fix_A_interval time steps.

        Returns:
            sg_target (numpy array): Array of shape (n_out) used to get error
                signals for A in update_learning_vars."""

        #Get latest q value, slide current q value to q_prev.
        self.propagate_feedback_to_hidden()

        self.a_tilde = np.concatenate([self.rnn.a,
                                       self.rnn.y,
                                       np.array([1])])

        #Calculate the synthetic gradient for the 'next' (really the current,
        #but next relative to the previous) time step.
        sg_next = self.synthetic_grad_(self.a_tilde)
        #Backpropagate by one time step and add to q_prev to get sg_target.
        if self.use_approx_J: #Approximate Jacobian
            sg_target = self.q_prev + sg_next.dot(self.J_approx)
        else: #Exact Jacobian
            sg_target = self.q_prev + sg_next.dot(self.rnn.a_J)

        return sg_target

    def update_J_approx(self):
        """Updates the approximate Jacobian by SGD according to a squared-error
        loss function:

        J_loss = 0.5 * || J a_prev - a ||^2.                     (6)

        Thus the gradient for the Jacobian is

        dJ_loss/dJ_{ij} = (J a_prev - a)_i a_prev_j              (7)."""

        self.J_error = self.J_approx.dot(self.rnn.a_prev) - self.rnn.a
        self.J_loss = 0.5 * np.square(self.J_error).mean()
        self.J_approx -= self.J_lr * np.multiply.outer(self.J_error,
                                                       self.rnn.a_prev)

    def synthetic_grad(self, a_tilde):
        """Computes the synthetic gradient using current values of A.

        Retuns:
            An array of shape (n_h) representing the synthetic gradient."""

        self.sg_h = self.A.dot(a_tilde)
        return self.activation.f(self.sg_h)

    def synthetic_grad_(self, a_tilde):
        """Computes the synthetic gradient using A_ and with an extra activation
        function (for DNI(b)), only for computing the label in Eq. (3).

        Retuns:
            An array of shape (n_h) representing the synthetic gradient."""

        self.sg_h_ = self.A_.dot(a_tilde)
        return self.SG_label_activation.f((self.activation.f(self.sg_h_)))

    def get_rec_grads(self):
        """Computes the recurrent grads for the network by implementing Eq. (2),
        using the current synthetic gradient function.

        Note: assuming self.a_tilde already calculated by calling get_sg_target,
        which should happen by default since update_learning_vars is always
        called before get_rec_grads.

        Returns:
            An array of shape (n_h, m) representing the network gradient for
                the recurrent parameters."""

        #Calculate synthetic gradient
        self.sg = self.synthetic_grad(self.a_tilde)
        #Combine the first 3 factors of the RHS of Eq. (2) into sg_scaled
        D = self.rnn.activation.f_prime(self.rnn.h)
        self.sg_scaled = self.sg * self.rnn.alpha * D

        self.a_hat = np.concatenate([self.rnn.a_prev,
                                     self.rnn.x,
                                     np.array([1])])
        #Final result of Eq. (2)
        return np.multiply.outer(self.sg_scaled, self.a_hat)
