import sys, os
sys.path.append(os.path.abspath('..'))
from utils import split_weight_matrix
import numpy as np
from core import Simulation
from continual_learning.Continual_Learning import Continual_Learning

class Duncker_Method(Continual_Learning):
    """CL Method that preferentially updates parameters in directions
        that minimally interfere with subspaces used on previous subtasks.

        Ref: Duncker and Sussillo, NeurIPS 2020."""

    def __init__(self, rnn, N_proj_data=500, mode='combined',
                 combined_task=None, proj_tasks=None):
        """Initialize instance by specifyiing RNN to be trained, amount of
        data to use in projections, and method for doing so.

        Args:
            rnn (core.RNN): The RNN instance undergoing continual learning.
            N_proj_data (int): Number of time steps of data to use for
                computing projection matrices.
            mode (str): One of 'combined', 'previous' or 'training':
                'combined' -> Combine all component tasks in a new training
                    simulation to collect projection data.
                'previous' -> Combine all component tasks *the network has
                    seen so far* in new training simulation to collect
                    projection data.
                'training' -> Use last N_proj_data time steps of the actual
                    training activity rather than separate test data (this is
                    the method used in original paper).
            combined_task (gen_data.Task): The task to be used if method is
                'combined'.
            proj_tasks (list): List of gen_data.Task instances to be used
                sequentially at each task break if method is 'previous'."""

        assert mode in ['combined', 'previous', 'training']

        self.rnn = rnn
        self.N_proj_data = N_proj_data
        self.mode = mode
        self.combined_task = combined_task
        self.proj_tasks = proj_tasks
        self.n_switches = 0

        m = rnn.n_h + rnn.n_in + 1

        self.rec_proj_mats = [np.eye(rnn.n_h), np.eye(m)]
        self.out_proj_mats = [np.eye(rnn.n_out), np.eye(rnn.n_h + 1)]

    def mini_update(self, sim):
        """No update necessary at each time step"""

        pass

    def task_switch_update(self, sim):
        """By one of the three methods, input and hidden state data matrices
        are generated or accessed, which are used to compute the projection
        matrices that the gradient is ultimately passed through."""

        self.n_switches += 1

        if self.mode == 'combined':

            proj_data = self.combined_task.gen_data(0, self.N_proj_data)
            proj_sim = Simulation(self.rnn)
            proj_sim.run(proj_data, mode='test',
                         monitors=['rnn.a', 'rnn.x'],
                         verbose=False)

            self.A_Duncker = proj_sim.mons['rnn.a']
            self.X_Duncker = proj_sim.mons['rnn.x']

        elif self.mode == 'previous':

            task = self.proj_tasks.pop(0)
            proj_data = task.gen_data(0, self.N_proj_data)
            proj_sim = Simulation(self.rnn)
            proj_sim.run(proj_data, mode='test',
                         monitors=['rnn.a', 'rnn.x'],
                         verbose=False)

            self.A_Duncker = proj_sim.mons['rnn.a']
            self.X_Duncker = proj_sim.mons['rnn.x']

        elif self.mode == 'training':

            A = np.array(sim.mons['rnn.a'][-self.N_proj_data:])
            X = np.array(sim.mons['rnn.x'][-self.N_proj_data:])

            if not hasattr(self, 'A_Duncker'):
                self.A_Duncker = A
                self.X_Duncker = X
            else:
                self.A_Duncker = np.vstack([self.A_Duncker, A])
                self.X_Duncker = np.vstack([self.X_Duncker, X])

        P_z, P_wz, P_h, P_y = self.get_projections(self.A_Duncker,
                                                   self.X_Duncker,
                                                   n_switches=self.n_switches)

        self.rec_proj_mats = [P_wz, P_z]
        self.out_proj_mats = [P_y, P_h]

    def get_projections(self, A, X, n_switches=1, inv_constant=0.001):
        """Get the continual learning projections from Duncker et al. given the
        data that should be included and the current parameters of the rnn."""

        rnn = self.rnn

        m = rnn.n_h + rnn.n_in + 1
        Z = np.concatenate([A, X, np.ones((A.shape[0], 1))], axis=1).T
        W = np.concatenate([rnn.W_rec,
                            rnn.W_in,
                            rnn.b_rec.reshape((-1, 1))], axis=1)
        WZ = W.dot(Z)

        P_z = np.linalg.inv(Z.dot(Z.T) / (n_switches * inv_constant) + np.eye(m))
        P_wz = np.linalg.inv(WZ.dot(WZ.T) / (n_switches * inv_constant) + np.eye(rnn.n_h))

        H = np.concatenate([A, np.ones((A.shape[0], 1))], axis=1).T
        W_o = np.concatenate([rnn.W_out, rnn.b_out.reshape((-1, 1))], axis=1)
        WH = W_o.dot(H)

        P_h = np.linalg.inv(H.dot(H.T) / (n_switches * inv_constant) + np.eye(rnn.n_h + 1))
        P_y = np.linalg.inv(WH.dot(WH.T) / (n_switches * inv_constant) + np.eye(rnn.n_out))

        return P_z, P_wz, P_h, P_y

    def __call__(self, grads_list):
        """Project gradients through matrices computed via get_projections"""

        grads = grads_list

        n_h = grads[0].shape[0]
        n_in = grads[1].shape[1]

        #Concatenate gradients in relevant directions
        W_grad = np.concatenate([grads[0], grads[1],
                                 grads[2].reshape(-1,1)], axis=1)
        W_out_grad = np.concatenate([grads[3], grads[4].reshape(-1,1)], axis=1)

        #Project along projection matrices
        W_grad_proj = self.rec_proj_mats[0].dot(W_grad)
        W_grad_proj = W_grad_proj.dot(self.rec_proj_mats[1])
        W_out_grad_proj = self.out_proj_mats[0].dot(W_out_grad)
        W_out_grad_proj = W_out_grad_proj.dot(self.out_proj_mats[1])

        ret = split_weight_matrix(W_grad_proj, [n_h, n_in, 1])
        ret += split_weight_matrix(W_out_grad_proj, [n_h, 1])

        return ret