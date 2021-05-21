from learning_algorithms.Learning_Algorithm import Learning_Algorithm
import numpy as np
from copy import deepcopy

class KeRNL(Learning_Algorithm):
    """Implements the Kernel RNN Learning (KeRNL) algorithm from Roth et al.
    2019.

    Details in our review or original paper. Briefly, a matrix A of shape
    (n_h, n_h) and an eligibility trace B of shape (n_h, m) are both
    maintained to approximate the influence matrix as M_{kij} ~ A_{ki} B_{ij}.
    In addition, a set of n_h learned timescales \alpha_i are maintained such
    that da^{(t+ \Delta t)}_k/da^{(t)}_i ~ A_{ki} e^{-\alpha_i \Delta t}. These
    approximations are updated at every time step. B is updated by temporally
    filtering the immediate influence via the learned timescales

    B'_{ij} = (1 - \alpha_i) B_{ij} + \alpha \phi'(h_i) a_hat_j         (1)

    while A and \alpha are updated by SGD on their ability to predict
    perturbative effects. See Algorithm 1: Pseudocode on page 5 of Roth et al.
    2019 for details."""

    def __init__(self, rnn, optimizer, sigma_noise=0.00001, **kwargs):
        """Inits an instance of KeRNL by specifying the optimizer used to train
        the A and alpha values and a noise standard deviation for the
        perturbations.

        Args:
            optimizer (optimizers.Optimizer): An instance of the Optimizer class
                for the training of A and alpha.
            sigma_noise (float): Standard deviation for the values, sampled
                i.i.d. from a zero-mean Gaussian, used to perturb the network
                state to noisy_rnn and thus estimate target predictions for
                A and alpha.

        Keyword args:
            A (numpy array): Initial value of A matrix, must be of shape
                (n_h, n_h). Default is identity.
            B (numpy array): Initial value of B matrix, must be of shape
                (n_h, m). Default is zeros.
            alpha (numpy array): Initial value of learned timescales alpha,
                must be of shape (n_h). Default is all 0.8.
            Omega (numpy array): Initial value of filtered perturbations, must
                be of shape (n_h). Default is 0.
            Gamma (numpy array): Initial value of derivative of filtered
                perturbations, must be of shape (n_h). Default is 0.
            T_reset (int): Number of time steps between regular resets of
                perturbed network to network state and Omega, Gamma, B variables
                to 0. If unspecified, default is no resetting."""

        self.name = 'KeRNL'
        allowed_kwargs_ = {'A', 'B', 'alpha', 'Omega', 'Gamma', 'T_reset'}
        super().__init__(rnn, allowed_kwargs_, **kwargs)

        self.i_t = 0
        self.sigma_noise = sigma_noise
        self.optimizer = optimizer
        self.zeta = np.random.normal(0, self.sigma_noise, self.n_h)

        #Initialize learning variables
        if self.A is None:
            self.A = np.eye(self.n_h)
        if self.B is None:
            self.B = np.zeros((self.n_h, self.m))
        if self.alpha is None:
            self.alpha = np.ones(self.n_h) * 0.8
        if self.Omega is None:
            self.Omega = np.zeros(self.n_h)
        if self.Gamma is None:
            self.Gamma = np.zeros(self.n_h)

        #Initialize noisy network as copy of original network
        self.noisy_rnn = deepcopy(self.rnn)

    def update_learning_vars(self):
        """Updates the matrices A and B, which are ultimately used to drive
        learning. In the process also updates alpha, Omega, and Gamma."""

        #Reset learning variables if desired and on schedule to do so
        if self.T_reset is not None:
            if self.i_t % self.T_reset == 0:
                self.reset_learning()

        #Match noisy network's parameters to latest network parameters
        self.noisy_rnn.W_rec = self.rnn.W_rec
        self.noisy_rnn.W_in = self.rnn.W_in
        self.noisy_rnn.b_rec = self.rnn.b_rec

        #Run perturbed network forwards
        self.noisy_rnn.a += self.zeta
        self.noisy_rnn.next_state(self.rnn.x)

        #Update learning variables (see Pseudocode in Roth et al. 2019)
        self.kernel = np.maximum(0, 1 - self.alpha)
        self.zeta = np.random.normal(0, self.sigma_noise, self.n_h)
        self.Gamma = self.kernel * self.Gamma - self.Omega
        self.Omega = self.kernel * self.Omega + self.zeta

        #Update eligibility trace (Eq. 1)
        self.D = self.rnn.activation.f_prime(self.rnn.h)
        self.a_hat = np.concatenate([self.rnn.a_prev,
                                     self.rnn.x,
                                     np.array([1])])
        self.papw = self.rnn.alpha * np.multiply.outer(self.D, self.a_hat)
        self.B = (self.B.T * self.kernel).T + self.papw

        #Get error in predicting perturbations effect (see Pseudocode)
        self.error_prediction = self.A.dot(self.Omega)
        self.error_observed = self.noisy_rnn.a - self.rnn.a
        self.noise_loss = np.square(self.error_prediction -
                                    self.error_observed).sum()
        self.noise_error = self.error_prediction - self.error_observed

        #Update A and alpha (see Pseudocode)
        self.A_grads = np.multiply.outer(self.noise_error, self.Omega)
        self.alpha_grads = self.noise_error.dot(self.A) * self.Gamma
        params = [self.A, self. alpha]
        grads = [self.A_grads, self.alpha_grads]
        self.A, self.alpha = self.optimizer.get_updated_params(params, grads)

        self.i_t += 1

    def get_rec_grads(self):
        """Using updated A and B, returns recurrent gradients according to
        final line in Pseudocode table in Roth et al. 2019."""

        return (self.B.T * self.q.dot(self.A)).T

    def reset_learning(self):
        """Resets learning variables to 0 and resets the perturbed network
        to the state of the primary network."""

        self.noisy_rnn.a = np.copy(self.rnn.a)
        self.Omega = np.zeros_like(self.Omega)
        self.Gamma = np.zeros_like(self.Gamma)
        self.B = np.zeros_like(self.B)
