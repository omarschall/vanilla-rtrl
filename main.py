#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 16:30:58 2018

@author: omarschall
"""
# %%
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
from meta_algorithm import *
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
    n_seeds = 1
    try:
        i_job = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
    except KeyError:
        i_job = 0
    #macro_configs = config_generator(i_start=list(range(0, 200000, 1000)))
    macro_configs = config_generator(algorithm=['RFLO'],
                                     name=['combined'],
                                     i_start=list(range(0, 10000, 100)))
    #macro_configs = config_generator(algorithm=['RFLO'])
    micro_configs = tuple(product(macro_configs, list(range(n_seeds))))

    params, i_seed = micro_configs[i_job]
    i_config = i_job//n_seeds
    np.random.seed(i_job)
    
if os.environ['HOME'] == '/Users/omarschall':
    params = {'algorithm': 'RFLO', 'name': 'combined'}
    i_job = 0
    save_dir = '/Users/omarschall/vanilla-rtrl/library'

np.random.seed(1)

task_1 = Add_Task(t_1=3, t_2=5, deterministic=True)
task_2 = Add_Task(t_1=5, t_2=8, deterministic=True)
task_3 = Add_Task(t_1=2, t_2=10, deterministic=True)
tasks = [task_1, task_2, task_3]
task = Multi_Task(tasks, context_input=False)
N_train = 100000
N_test = 5000
checkpoint_interval = None

data = task.gen_data(N_train, N_test)

n_in = task.n_in
n_hidden = 4
n_out = task.n_out
W_in  = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
#W_rec = np.linalg.qr(np.random.normal(0, 1, (n_hidden, n_hidden)))[0]
W_rec = np.random.normal(0, np.sqrt(1/n_hidden), (n_hidden, n_hidden))
W_out = np.random.normal(0, np.sqrt(1/(n_hidden)), (n_out, n_hidden))

b_rec = np.zeros(n_hidden)
b_out = np.zeros(n_out)

alpha = 1
sigma = 0

rnn = RNN(W_in, W_rec, W_out, b_rec, b_out,
          activation=tanh,
          alpha=alpha,
          output=softmax,
          loss=softmax_cross_entropy)


optimizer = Private_LR_SGD(rnn,init_lr=0.001)
meta_optimizer = Stochastic_Gradient_Descent(lr=0)
#learn_alg = KF_RTRL(rnn, L2_reg=0.0001, L1_reg=0.0001)
learn_alg = RTRL(rnn)
meta_learn_alg = Meta_Learning_Algorithm(rnn,learn_alg,optimizer,clip_norm= 100)

comp_algs = []
monitors = ['rnn.loss_','meta_learn_alg.rec_grads-norm',
'meta_learn_alg.dwdlam-norm',
'meta_learn_alg.H-norm',]
# 'meta_learn_alg.F1-norm',
# 'meta_learn_alg.F2-norm',
# 'meta_learn_alg.G1-norm',
# 'meta_learn_alg.G2-norm',
# 'meta_learn_alg.qB_diag-norm',
# 'meta_learn_alg.A_diag-norm',]
# 'meta_learn_alg.eta-norm',
# 'meta_learn_alg.gamma-norm',
# 'meta_learn_alg.G1_beta-norm',
# 'meta_learn_alg.G1',
# 'meta_learn_alg.beta',
# 'meta_learn_alg.G2_beta-norm',
# 'meta_learn_alg.v1-norm',
# 'meta_learn_alg.v2-norm',
# 'meta_learn_alg.v3-norm',
# 'meta_learn_alg.w1-norm',
# 'meta_learn_alg.w2-norm',
# 'meta_learn_alg.p0',
# 'meta_learn_alg.p1',
# 'meta_learn_alg.p2',
# 'meta_learn_alg.p3',
# 'meta_learn_alg.F1_alpha-norm',
# 'meta_learn_alg.F2_alpha-norm']
# 'meta_learn_alg.tA-norm',
# 'meta_learn_alg.tB-norm']


sim = Simulation(rnn)
sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
        sigma=sigma,
        comp_algs=comp_algs,
        monitors=monitors,
        verbose=True,
        report_accuracy=False,
        report_loss=True,
        checkpoint_interval=checkpoint_interval,
        outer_loop = True,
        meta_learn_alg= meta_learn_alg,
        meta_optimizer = meta_optimizer
        )

# %%


plt.figure(figsize=(10,10))
for name in monitors:
    if np.mean(sim.mons[name][:1000])>0:
        plt.plot(sim.mons[name][:1000],label=name)
plt.legend()
plt.show()




# %% 

if os.environ['HOME'] == '/Users/omarschall' and MODE == 'TRAIN':

    rnn = sim.checkpoints[N_train - 1]['rnn']
    test_sim = Simulation(rnn)
    test_sim.run(data,
                  mode='test',
                  monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
                  verbose=False)

    plt.figure()
    plt.plot(data['test']['X'][:, 0] + 2.5, (str(0.6)), linestyle='--')
    plt.plot(data['test']['Y'][:, 0] + 2.5, 'C0')
    plt.plot(test_sim.mons['rnn.y_hat'][:, 0] + 2.5, 'C3')
    plt.plot(data['test']['X'][:, 1], (str(0.6)), linestyle='--')
    plt.plot(data['test']['Y'][:, 1], 'C0')
    plt.plot(test_sim.mons['rnn.y_hat'][:, 1], 'C3')
    plt.plot(data['test']['X'][:, 2] - 2.5, (str(0.6)), linestyle='--')
    plt.plot(data['test']['Y'][:, 2] - 2.5, 'C0')
    plt.plot(test_sim.mons['rnn.y_hat'][:, 2] - 2.5, 'C3')
    plt.xlim([0, 100])
    plt.yticks([])
    plt.xlabel('time steps')
    
    plt.figure()
    for key in monitors:
        plt.plot(sim.mons[key])
    

if os.environ['HOME'] == '/home/oem214':

    # result = {'sim': sim, 'i_seed': i_seed, 'task': task,
    #           'config': params, 'i_config': i_config, 'i_job': i_job,
    #           'processed_data': processed_data}
    result['i_job'] = i_job
    result['config'] = params
    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'result_'+str(i_job))

    with open(save_path, 'wb') as f:
        pickle.dump(result, f)












