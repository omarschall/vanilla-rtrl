import numpy as np
from gen_data.Task import Task

class Mimic_RNN(Task):
    """Class for the 'Mimic Task,' where the inputs are random i.i.d. Bernoulli
    and the labels are the outputs of a fixed 'target' RNN that is fed these
    inputs."""

    def __init__(self, rnn, p_input, tau_task=1, latent_dim=None):
        """Initializes the task with a target RNN (instance of network.RNN),
        the probability of the Bernoulli inputs, and a time constant of change.

        Args:
            rnn (network.RNN instance): The target RNN
            p_input (float): The probability of any input having value 1
            tau_task (int): The temporal stretching factor for the inputs, see
                tau_task in Add_Task."""

        #Initialize as Task object with dims inherited from the target RNN.
        super().__init__(rnn.n_in, rnn.n_out)

        self.rnn = rnn
        self.p_input = p_input
        self.tau_task = tau_task
        self.latent_dim = latent_dim
        if self.latent_dim is not None:
            self.segment_length = self.n_in // self.latent_dim

    def gen_dataset(self, N):
        """Generates a dataset by first generating inputs randomly by the
        binomial distribution and temporally stretching them by tau_task,
        then feeding these inputs to the target RNN."""

        #Generate inputs
        N = N // self.tau_task
        X = []
        for i in range(N):
            if self.latent_dim is not None:
                outcomes = np.random.binomial(1, self.p_input, self.latent_dim)
                x = [o * np.ones(self.segment_length) for o in outcomes]
                x = np.concatenate(x)
            else:
                x = np.random.binomial(1, self.p_input, self.n_in)
            X.append(x)
        X = np.tile(X, self.tau_task).reshape((self.tau_task*N, self.n_in))

        #Get RNN responses
        Y = []
        self.rnn.reset_network(h=np.zeros(self.rnn.n_h))
        for i in range(len(X)):
            self.rnn.next_state(X[i])
            self.rnn.z_out()
            Y.append(self.rnn.output.f(self.rnn.z))

        return X, np.array(Y), None
