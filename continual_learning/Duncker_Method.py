import sys, os
sys.path.append(os.path.abspath('..'))
from utils import get_Duncker_projections, split_weight_matrix
import numpy as np
from core import Simulation
from continual_learning.Continual_Learning import Continual_Learning

class Duncker_Method(Continual_Learning):

    def __init__(self, rnn, N_proj_data=500, mode='combined',
                 combined_task=None, proj_tasks=None):

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

        pass

    def task_switch_update(self, sim):

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

            A = np.array(sim.mons['rnn.a'][-N_proj_data:])
            X = np.array(sim.mons['rnn.x'][-N_proj_data:])

            if not hasattr(self, 'A_Duncker'):
                self.A_Duncker = A
                self.X_Duncker = X
            else:
                self.A_Duncker = np.vstack([self.A_Duncker, A])
                self.X_Duncker = np.vstack([self.X_Duncker, X])

        P_z, P_wz, P_h, P_y = get_Duncker_projections(self.A_Duncker,
                                                      self.X_Duncker,
                                                      rnn=self.rnn,
                                                      n_switches=self.n_switches)

        self.rec_proj_mats = [P_wz, P_z]
        self.out_proj_mats=[P_y, P_h]

    def __call__(self, grads_list):

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