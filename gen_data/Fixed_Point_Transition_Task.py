import numpy as np
from .Task import Task

class Fixed_Point_Transition_Task(Task):
    """A task where a given number of states in output space is provided,
    along with a set of transition matrices as functions of possible inputs."""

    def __init__(self, states, T_dict, p_transition=0.05, deterministic=True,
                 delay=None):
        """Initializes an instance of the task by specifying the states
        (points in output space) to be reported, as well as a dictionary
        specifying the transition patterns for each input.

        Args:
            states (list): A list of length n_states, whose entries are umpy
                arrays with shape (n_out) that indicate in order the fixed
                points to be learned.
            T_dict (dictionary): A dictionary with keys 'input_{}'.format(k)
                for each possible input k, each pointing to a numpy array
                of shape (n_states, n_states) indicating transition probs.
            p_transition (float): A float indicating the probability of
                encountering a transition at each time point.
            deterministic (bool): A boolean indicating True if you know
                T_dict to be deterministic, which simplifies data gen.
            delay (int): An integer indicating the number of time steps to
                delay the indicated transition."""

        self.states = states
        self.n_states = len(states)
        self.T_dict = T_dict
        self.p_transition = p_transition
        self.deterministic = deterministic
        self.delay = delay
        self.probe_inputs = [np.eye(self.n_states)[i]
                             for i in range(self.n_states)]

        super().__init__(len(T_dict.keys()), states[0].shape[0])

        #Assert dimensional consistency
        assert len(np.unique([state.shape[0] for state in states])) == 1
        for val in T_dict.values():
            assert self.n_states == val.shape[0]
            assert self.n_states == val.shape[1]

    def gen_dataset(self, N):

        #Indicate time steps where transition occurs
        p = [1 - self.p_transition, self.p_transition]
        sparsity_mask = np.random.choice([0, 1], size=N, p=p)
        sparsity_mask[0] = 0 #No transition at first time step
        T_times = np.where(sparsity_mask != 0)[0]

        #Sample particular transition types
        I_X = np.random.randint(0, self.n_in, size=N) #* sparsity_mask
        X = np.eye(self.n_in)[I_X] #Turn into one-hot

        #apply sparsity
        I_X *= sparsity_mask
        X = (X.T * sparsity_mask).T

        #Set up initial state
        I_Y = [np.random.randint(self.n_states)]

        last_T_time = 0
        for i_T_time, T_time in enumerate(T_times):

            #Backfill all the last "same" states since last transition
            I_Y += ([I_Y[-1]] * (T_time - last_T_time - 1))

            #Access current transition and associated graph
            i_input = I_X[T_time]
            #set_trace()
            key = 'input_{}'.format(i_input)
            T_matrix = self.T_dict[key]

            if self.deterministic:
                I_Y.append(np.where(T_matrix[I_Y[-1]] > 0)[0][0])
            else:
                p_state = T_matrix[I_Y[-1]]
                I_Y.append(np.random.choice(list(range(self.n_states)),
                                            p=p_state))

            last_T_time = T_time

        #Final state

        I_Y += ([I_Y[-1]] * (len(I_X) - last_T_time - 1))

        Y = np.array([self.states[i_y] for i_y in I_Y])

        if self.delay is not None:
            Y = np.roll(Y, shift=self.delay, axis=0)

        return X, Y
