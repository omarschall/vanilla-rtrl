from learning_algorithms.Stochastic_Algorithm import Stochastic_Algorithm
from utils import *
from functions import *

class KF_RTRL(Stochastic_Algorithm):
    """Implements the Kronecker-Factored Real-Time Recurrent Learning Algorithm
    (KF-RTRL) from Mujika et al. 2018.

    Details in review paper or original Mujika et al. 2018. Broadly, M is
    approximated as a Kronecker product between a (row) vector A and a matrix
    B, which updates as

    A' = \nu_0 p0 A + \nu_1 p1 a_hat        (1)
    B' = \nu_0 1/p0 JB + \nu_1 1/p1 \alpha diag(\phi'(h))      (2)

    where \nu = (\nu_0, \nu_1) is a vector of zero-mean iid samples, a_hat is
    the concatenation [a_prev, x, 1], and p0 and p1 are calculated by

    p0 = \sqrt{norm(JB)/norm(A)}       (3)
    p1 = \sqrt{norm(D)/norm(a_hat)}        (4)

    Then the recurrent gradients are calculated by

    dL/dw = qM = A (qB)    (5)

    Eq. (5) is implemented in the get_rec_grads method.
    """

    def __init__(self, rnn, **kwargs):
        """Inits a KF-RTRL instance by setting the initial values of A and B to
        be iid samples from a gaussian distributions, to avoid dividing by
        zero in Eqs. (3) and (4).

        Keyword args:
            P0 (float): Overrides calculation of p0, instead uses provided value
                of P0. If not provided, p0 is calculated according to Eq. (3).
            P1 (float): Same for p1.
            A (numpy array): Initial value for A.
            B (numpy array): Initial value for B.
            nu_dist (string): Takes on the value of 'gaussian', 'discrete', or
                'uniform' to indicate what type of distribution nu should sample
                 from. Default is 'discrete'."""

        self.name = 'KF-RTRL'
        allowed_kwargs_ = {'P0', 'P1', 'A', 'B', 'nu_dist'}
        super().__init__(rnn, allowed_kwargs_, **kwargs)
        self.n_nu = 2

        #Initialize A and B arrays
        if self.A is None:
            self.A = np.random.normal(0, 1, self.m)
        if self.B is None:
            self.B = np.random.normal(0, 1, (self.n_h, self.n_h))

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
        self.D = np.diag(self.rnn.activation.f_prime(self.rnn.h))
        self.rnn.get_a_jacobian()
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
            Updated A (numpy array of shape (m)) and B (numpy array of shape
                (n_h, n_h))."""

        #Sample random vector (shape (2) in KF-RTRL)
        self.nu = self.sample_nu()

        #Calculate p0, p1 or override with fixed P0, P1 if given
        if self.P0 is None:
            self.p0 = np.sqrt(norm(self.B_forwards)/norm(self.A))
        else:
            self.p0 = np.copy(self.P0)
        if self.P1 is None:
            self.p1 = np.sqrt(norm(self.D)/norm(self.a_hat))
        else:
            self.p1 = np.copy(self.P1)

        #Update Kronecker product approximation
        A = self.nu[0]*self.p0*self.A + self.nu[1]*self.p1*self.a_hat
        B = (self.nu[0]*(1/self.p0)*self.B_forwards +
             self.nu[1]*(1/self.p1)*self.D)

        return A, B

    def get_rec_grads(self):
        """Calculates recurrent grads by taking matrix product of q with the
        estimate of the influence matrix.

        First associates q with B to calculate a vector qB, whose Kron. product
        with A (effectively an outer product upon reshaping) gives the estimated
        recurrent gradient.

        Returns:
            An array of shape (n_h, m) representing the recurrent gradient."""

        self.qB = self.q.dot(self.B) #Unit-specific learning signal
        return np.kron(self.A, self.qB).reshape((self.n_h, self.m), order='F')

    def reset_learning(self):
        """Resets learning by re-randomizing the outer product approximation to
        random gaussian samples."""

        self.A = np.random.normal(0, 1, self.m)
        self.B = np.random.normal(0, 1, (self.n_h, self.n_h))
