from learning_algorithms.Stochastic_Algorithm import Stochastic_Algorithm
from utils import *
from functions import *

class Reverse_KF_RTRL(Stochastic_Algorithm):
    """Implements the "Reverse" KF-RTRL (R-KF-RTRL) algorithm.

    Full details in our review paper. Broadly, an approximation of M in the form
    of a Kronecker product between a matrix B and a (row) vector A is maintained
    by the update

    A'_i = p0 A_i + p1 \nu_i        (1)
    B'_{kj} = (1/p0 \sum_{k'} J+{kk'}B_{k'j} +
               1/p1 \sum_i \nu_i M_immediate_{kij})      (2)

    where \nu is a vector of zero-mean iid samples. p0 and p1 are calculated by

    p0 = \sqrt{norm(B)/norm(A)}       (3)
    p1 = \sqrt{norm(\nu papw)/norm(\nu)}        (4)

    Then the recurrent gradients are calculated by

    dL/dw = qM = A (qB)    (5)

    Eq. (5) is implemented in the get_rec_grads method."""

    def __init__(self, rnn, **kwargs):
        """Inits an R-KF-RTRL instance by setting the initial values of A and B
        to be iid samples from gaussian distributions, to avoid dividing by
        zero in Eqs. (3) and (4).

        Keyword args:
            epsilon (float): Scaling factor on perturbation for forward
                differentiation method. If not provided, exact derivative is
                calculated instead.
            P0 (float): Overrides calculation of p0, instead uses provided value
                of P0. If not provided, p0 is calculated according to Eq. (3).
            P1 (float): Same for p1.
            A (numpy array): Initial value for A.
            B (numpy array): Initial value for B.
            nu_dist (string): Takes on the value of 'gaussian', 'discrete', or
                'uniform' to indicate what type of distribution nu should sample
                 from. Default is 'discrete'."""

        self.name = 'R-KF-RTRL'
        allowed_kwargs_ = {'P0', 'P1', 'A', 'B', 'nu_dist'}
        super().__init__(rnn, allowed_kwargs_, **kwargs)
        self.n_nu = self.n_h

        #Initialize A and B arrays
        if self.A is None:
            self.A = np.random.normal(0, 1, self.n_h)
        if self.B is None:
            self.B = np.random.normal(0, 1, (self.n_h, self.m))

    def update_learning_vars(self, update=True):
        """Implements Eqs. (1), (2), (3), and (4) to update the Kron. product
        approximation of the influence matrix by A and B.

        Args:
            update (bool): If True, updates the algorithm's current outer
                product approximation B, A. If False, only prepares for calling
                get_influence_estimate."""

        #Get relevant values and derivatives from network
        self.a_hat = np.concatenate([self.rnn.a_prev,
                                     self.rnn.x,
                                     np.array([1])])
        self.D = self.rnn.activation.f_prime(self.rnn.h)
        #Compact form of M_immediate
        self.papw = np.multiply.outer(self.D, self.a_hat)
        self.rnn.get_a_jacobian() #Get updated network Jacobian
        self.B_forwards = self.rnn.a_J.dot(self.B)

        A, B = self.get_influence_estimate()

        if update:
            self.A, self.B = A, B

    def get_influence_estimate(self):
        """Generates one random Kron.-product estimate of the influence matrix.

        Samples a random vector nu of iid samples with 0 mean from a
        distribution given by nu_dist, and returns an updated estimate
        of A and B from Eqs. (1)-(4).

        Returns:
            Updated A (numpy array of shape (n_h)) and B (numpy array of shape
                (n_h, n_m))."""

        #Sample random vector
        self.nu = self.sample_nu()

        # Get random projection of M_immediate onto \nu
        M_projection = (self.papw.T * self.nu).T

        #Calculate scaling factors
        B_norm = norm(self.B_forwards)
        A_norm = norm(self.A)
        M_norm = norm(M_projection)
        self.p0 = np.sqrt(B_norm/A_norm)
        self.p1 = np.sqrt(M_norm/np.sqrt(self.n_h))

        #Override with fixed P0 and P1 if given
        if self.P0 is not None:
            self.p0 = np.copy(self.P0)
        if self.P1 is not None:
            self.p1 = np.copy(self.P1)

        #Update "inverse" Kronecker product approximation
        A = self.p0 * self.A + self.p1 * self.nu
        B = (1/self.p0) * self.B_forwards + (1/self.p1) * M_projection

        return A, B

    def get_rec_grads(self):
        """Calculates recurrent grads by taking matrix product of q with the
        estimate of the influence matrix.

        First associates q with B to calculate qB, then takes the outer product
        with A to get an estimate of the recurrent gradient.

        Returns:
            An array of shape (n_h, m) representing the recurrent gradient."""

        self.qB = self.q.dot(self.B) #Unit-specific learning signal
        return np.multiply.outer(self.A, self.qB)

    def reset_learning(self):
        """Resets learning by re-randomizing the Kron. product approximation to
        random gaussian samples."""

        self.A = np.random.normal(0, 1, self.n_h)
        self.B = np.random.normal(0, 1, (self.n_h, self.m))
