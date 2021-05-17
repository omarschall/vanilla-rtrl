from learning_algorithms.Learning_Algorithm import Learning_Algorithm
from utils import *
from functions import *

class Stochastic_Algorithm(Learning_Algorithm):

    def sample_nu(self):
        """Sample nu from specified distribution."""

        if self.nu_dist == 'discrete' or self.nu_dist is None:
            nu = np.random.choice([-1, 1], self.n_nu)
        elif self.nu_dist == 'gaussian':
            nu = np.random.normal(0, 1, self.n_nu)
        elif self.nu_dist == 'uniform':
            nu = np.random.uniform(-1, 1, self.n_nu)

        return nu
