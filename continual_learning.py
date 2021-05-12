#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 15:38:55 2021

@author: omarschall
"""

import numpy as np
from utils import get_Duncker_projections, split_weight_matrix
from simulation import Simulation

class Continual_Learning_Method:
    
    def __init__(self, rnn):
        
        pass
    
    def mini_update(self, sim):
        
        pass
    
    def task_switch_update(self, sim):
        
        pass
    
    def __call__(self, grads_list):
        
        pass
    
class Duncker_Method(Continual_Learning_Method):
    
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

class Synaptic_Intelligence(Continual_Learning_Method):
    
    def __init__(self, rnn, c, epsilon=0.001):
        
        self.rnn = rnn
        self.c = c
        self.epsilon = epsilon
        
        self.SI_Theta = [[p.copy() for p in self.rnn.params]]
        self.SI_Delta = []
        self.SI_Omega = [np.zeros(s) for s in self.rnn.shapes]
        self.SI_omega = [np.zeros(s) for s in self.rnn.shapes]
        
    def mini_update(self, sim):
        
        #Update omegas
        for i_param in range(len(self.SI_omega)):
            v = sim.optimizer.vel[i_param]
            g = sim.grads_list[i_param]
            self.SI_omega[i_param] -= v * g
        
    def task_switch_update(self, sim):
        
        #Refresh omegas, update Omegas
        if sim.task_marker[sim.i_t] != sim.task_marker[sim.i_t - 1]:
            
            self.SI_Theta.append([p.copy() for p in self.rnn.params])
            self.SI_Delta = [p - q for p, q in zip(self.SI_Theta[-1],
                                                   self.SI_Theta[-2])]
            
            for i_param in range(len(self.SI_omega)):
                omega = self.SI_omega[i_param]
                Delta = self.SI_Delta[i_param]
                self.SI_Omega[i_param] += omega / (Delta**2 + self.epsilon)
            
            self.SI_omega = [np.zeros(s) for s in self.rnn.shapes]
            
    def __call__(self, grads_list):
        
        new_grads = []
        diff = []
        for i_param, grad in enumerate(grads_list):
            
            Omega = self.SI_Omega[i_param]
            Theta = self.rnn.params[i_param]
            Theta_tilde = self.SI_Theta[-1][i_param]
            new_grads.append(grad + 2 * self.c * Omega * (Theta - Theta_tilde))
            diff.append((Theta - Theta_tilde))
        
        self.loss = np.mean([O * np.square(d).mean() for d, O in zip(diff,
                                                                     Omega)])
            
        return new_grads
        