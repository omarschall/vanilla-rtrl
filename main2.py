#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 00:03:45 2021

@author: omarschall
"""
import numpy as np
from network import *
from simulation import *
from gen_data import *
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    pass
from optimizers import *
from analysis_funcs import *
from learning_algorithms import *
from continual_learning import *
from functions import *
from itertools import product
import os
import pickle
from time import time
from copy import deepcopy
from scipy.ndimage.filters import uniform_filter1d
from sklearn import linear_model
from state_space import *
from dynamics import *
import multiprocessing as mp
from functools import partial
from sklearn.cluster import DBSCAN
from distances import *

if os.environ['HOME'] == '/home/oem214':
    n_seeds = 3
    try:
        i_job = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
    except KeyError:
        i_job = 0
        
    macro_configs = config_generator(N_Duncker_data=[2000, 4000, 8000],
                                     lr=[0.005, 0.01, 0.05, 0.08],
                                     N1=[10000, 50000, 100000],
                                     N2=[10000, 50000, 100000],
                                     N3=[10000, 50000, 100000])
    micro_configs = tuple(product(macro_configs, list(range(n_seeds))))

    params, i_seed = micro_configs[i_job]
    i_config = i_job//n_seeds
    np.random.seed(i_job)
    
if os.environ['HOME'] == '/Users/omarschall':
    params = {'N_Duncker_data': 2000,
              'lr': 0.05,
              'N1': 100000,
              'N2': 10000,
              'N3': 10000}
    #i_seed = 26
    #np.random.seed(i_seed)
    i_job = 0
    save_dir = '/Users/omarschall/vanilla-rtrl/library'

task_1 = Flip_Flop_Task(3, 0.05, x_dim_mask=[1,0,0], y_dim_mask=[1,0,0])
task_2 = Flip_Flop_Task(3, 0.05, x_dim_mask=[0,1,0], y_dim_mask=[0,1,0])
task_3 = Flip_Flop_Task(3, 0.05, x_dim_mask=[0,0,1], y_dim_mask=[0,0,1])
task = Multi_Task([task_1, task_2, task_3], context_input=False)
combined_task = Flip_Flop_Task(3, 0.05)
proj_task_1 = Flip_Flop_Task(3, 0.05, x_dim_mask=[1,0,0])
proj_task_2 = Flip_Flop_Task(3, 0.05, x_dim_mask=[1,1,0])
proj_tasks = [proj_task_1, proj_task_2]
N_train = [{'task_id': 0, 'N': params['N1']},
           {'task_id': 1, 'N': params['N2']},
           {'task_id': 2, 'N': params['N3']}]
N_test = 5000
checkpoint_interval = None
data = task.gen_data(N_train, N_test)

n_in = task.n_in
n_hidden = 32
n_out = task.n_out
W_in  = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
W_rec = np.random.normal(0, np.sqrt(1/n_hidden), (n_hidden, n_hidden))
W_out = np.random.normal(0, np.sqrt(1/(n_hidden)), (n_out, n_hidden))

b_rec = np.zeros(n_hidden)
b_out = np.zeros(n_out)

alpha = 1
sigma = 0

rnn = RNN(W_in, W_rec, W_out, b_rec, b_out,
          activation=tanh,
          alpha=alpha,
          output=identity,
          loss=mean_squared_error)

#cl_method = Synaptic_Intelligence(rnn, c=0.01)
cl_method = Duncker_Method(rnn, N_proj_data=params['N_Duncker_data'],
                           mode='previous', proj_tasks=proj_tasks)
learn_alg = RFLO(rnn, alpha=alpha, L2_reg=0.0001, L1_reg=0.0001,
                 CL_method=cl_method)

comp_algs = []
monitors = ['learn_alg.CL_method.loss']

### --- SIMULATION 1 --- ####    

#optimizer = Stochastic_Gradient_Descent(lr=0.005, clip_norm=None)
optimizer = SGD_Momentum(lr=params['lr'], mu=0.6)

sim = Simulation(rnn)
sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
        sigma=sigma,
        comp_algs=comp_algs,
        monitors=monitors,
        verbose=True,
        report_accuracy=False,
        report_loss=True,
        checkpoint_interval=checkpoint_interval,
        N_Duncker_data=None,
        combined_task=None,
        lr_Duncker=None,
        Duncker_proj_tasks=None)


data_1 = task_1.gen_data(0, N_test)
data_2 = task_2.gen_data(0, N_test)
data_3 = task_3.gen_data(0, N_test)
individual_task_data = [data_1, data_2, data_3]
data_combined = combined_task.gen_data(0, N_test)

test_sim_1 = Simulation(rnn)
test_sim_1.run(data_1, mode='test', monitors=['rnn.loss_'], verbose=False)

test_sim_2 = Simulation(rnn)
test_sim_2.run(data_2, mode='test', monitors=['rnn.loss_'], verbose=False)

test_sim_3 = Simulation(rnn)
test_sim_3.run(data_3, mode='test', monitors=['rnn.loss_'], verbose=False)

test_sim_combined = Simulation(rnn)
test_sim_combined.run(data_combined, mode='test', monitors=['rnn.loss_'], verbose=False)


processed_data = {'task_1': test_sim_1.mons['rnn.loss_'].mean(),
                  'task_2': test_sim_2.mons['rnn.loss_'].mean(),
                  'task_3': test_sim_3.mons['rnn.loss_'].mean(),
                  'combined_task': test_sim_combined.mons['rnn.loss_'].mean()}

if os.environ['HOME'] == '/Users/omarschall':
    
    test_sim = Simulation(rnn)
    test_sim.run(data_combined, mode='test', monitors=['rnn.y_hat', 'rnn.a'], verbose=False)
    
    plt.figure()
        
    for i in range(task.n_in):
        
        plt.plot((data_combined['test']['X'][:1000, i] + i*2.5), color=('0.6'))
        
    for j in range(task.n_out):
        plt.plot(data_combined['test']['Y'][:1000, j] + j*2.5, color='C0')
        plt.plot(test_sim.mons['rnn.y_hat'][:1000, j] + j*2.5, color='C2')
        
    for i in range(3):
        
        data_ = individual_task_data[i]
        
        test_sim = Simulation(rnn)
        test_sim.run(data_, mode='test', monitors=['rnn.y_hat', 'rnn.a'], verbose=False)
        
        plt.figure()
        plt.title('Task {}'.format(i + 1))
            
        for i in range(task.n_in):
            
            plt.plot((data_['test']['X'][:1000, i] + i*2.5), color=('0.6'))
            
        for j in range(task.n_out):
            plt.plot(data_['test']['Y'][:1000, j] + j*2.5, color='C0')
            plt.plot(test_sim.mons['rnn.y_hat'][:1000, j] + j*2.5, color='C2')

if os.environ['HOME'] == '/home/oem214':

    result = {'sim': sim, 'i_seed': i_seed, 'task': task,
              'config': params, 'i_config': i_config, 'i_job': i_job,
              'processed_data': processed_data}
    result['i_job'] = i_job
    result['config'] = params
    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'result_'+str(i_job))

    with open(save_path, 'wb') as f:
        pickle.dump(result, f)












