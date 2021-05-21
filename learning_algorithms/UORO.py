from learning_algorithms.Stochastic_Algorithm import Stochastic_Algorithm
import numpy as np
from utils import norm

class UORO(Stochastic_Algorithm):
    """Implements the Unbiased Online Recurrent Optimization (UORO) algorithm
    from Tallec et al. 2017.

    Full details in our review paper or in original paper. Broadly, an outer
    product approximation of M is maintained by 2 vectors A and B, which update
    by the equations

    A' = p0 J A + p1 \nu        (1)
    B' = 1/p0 B + 1/p1 \nu M_immediate      (2)

    where \nu is a vector of zero-mean iid samples. p0 and p1 are calculated by

    p0 = \sqrt{norm(B)/norm(A)}       (3)
    p1 = \sqrt{norm(\nu papw)/norm(\nu)}        (4)

    These equations are implemented in update_learning_vars by two different
    approaches. If 'epsilon' is provided as an argument, then the "forward
    differentiation" method from the original paper is used, where the matrix-
    vector product JA is estimated numerically by a perturbation of size
    epsilon in the A direction.

    Then the recurrent gradients are calculated by

    dL/dw = qM = (q A) B    (5)

    Eq. (5) is implemented in the get_rec_grads method."""

    def __init__(self, rnn, **kwargs):
        """Inits an UORO instance by setting the initial values of A and B to be
        iid samples from a standard normal distribution, to avoid dividing by
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

        self.name = 'UORO' #Default algorithm name
        allowed_kwargs_ = {'epsilon', 'P0', 'P1', 'A', 'B', 'nu_dist'}
        super().__init__(rnn, allowed_kwargs_, **kwargs)
        self.n_nu = self.n_h

        #Initialize A and B arrays
        if self.A is None:
            self.A = np.random.normal(0, 1, self.n_h)
        if self.B is None:
            self.B = np.random.normal(0, 1, (self.n_h, self.m))

    def update_learning_vars(self, update=True):
        """Implements Eqs. (1), (2), (3), and (4) to update the outer product
        approximation of the influence matrix by A and B.

        Args:
            update (bool): If True, updates the algorithm's current outer
                product approximation B, A. If False, only prepares for calling
                get_influence_estimate."""

        #Get relevant values and derivatives from network
        self.a_hat = np.concatenate([self.rnn.a_prev,
                                     self.rnn.x,
                                     np.array([1])])
        D = self.rnn.activation.f_prime(self.rnn.h)
        #Compact form of M_immediate
        self.papw = np.multiply.outer(D, self.a_hat)
        self.rnn.get_a_jacobian() #Get updated network Jacobian

        A, B = self.get_influence_estimate()

        if update:
            self.A, self.B = A, B

    def get_influence_estimate(self):
        """Generates one random outer-product estimate of the influence matrix.

        Samples a random vector nu of iid samples with 0 mean from a
        distribution given by nu_dist, and returns an updated estimate
        of A and B from Eqs. (1)-(4).

        Returns:
            Updated A (numpy array of shape (n_h)) and B (numpy array of shape
                (n_h, m))."""

        #Sample random vector
        self.nu = self.sample_nu()

        #Get random projection of M_immediate onto \nu
        M_projection = (self.papw.T*self.nu).T

        if self.epsilon is not None: #Forward differentiation method
            eps = self.epsilon
            #Get perturbed state in direction of A
            self.a_perturbed = self.rnn.a_prev + eps * self.A
            #Get hypothetical next states from this perturbation
            self.a_perturbed_next = self.rnn.next_state(self.rnn.x,
                                                        self.a_perturbed,
                                                        update=False)
            #Get forward-propagated A
            self.A_forwards = (self.a_perturbed_next - self.rnn.a) / eps
            #Calculate scaling factors
            B_norm = norm(self.B)
            A_norm = norm(self.A_forwards)
            M_norm = norm(M_projection)
            self.p0 = np.sqrt(B_norm/(A_norm + eps)) + eps
            self.p1 = np.sqrt(M_norm/(np.sqrt(self.n_h) + eps)) + eps
        else: #Backpropagation method
            #Get forward-propagated A
            self.A_forwards = self.rnn.a_J.dot(self.A)
            #Calculate scaling factors
            B_norm = norm(self.B)
            A_norm = norm(self.A_forwards)
            M_norm = norm(M_projection)
            self.p0 = np.sqrt(B_norm/A_norm)
            self.p1 = np.sqrt(M_norm/np.sqrt(self.n_h))

        #Override with fixed P0 and P1 if given
        if self.P0 is not None:
            self.p0 = np.copy(self.P0)
        if self.P1 is not None:
            self.p1 = np.copy(self.P1)

        #Update outer product approximation
        A = self.p0 * self.A_forwards + self.p1 * self.nu
        B = (1/self.p0) * self.B + (1 / self.p1) * M_projection

        return A, B

    def get_rec_grads(self):
        """Calculates recurrent grads by taking matrix product of q with the
        estimate of the influence matrix.

        First associates q with A to calculate a "global learning signal"
        Q, which multiplies by B to compute the recurrent gradient, which
        is reshaped into original matrix form.

        Returns:
            An array of shape (n_h, m) representing the recurrent gradient."""

        self.Q = self.q.dot(self.A) #"Global learning signal"
        return (self.Q * self.B)

    def reset_learning(self):
        """Resets learning by re-randomizing the outer product approximation to
        random gaussian samples."""

        self.A = np.random.normal(0, 1, self.n_h)
        self.B = np.random.normal(0, 1, (self.n_h, self.m))
