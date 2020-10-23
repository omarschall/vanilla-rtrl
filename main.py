#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 16:30:58 2018

@author: omarschall
"""

import numpy as np
from network import *
from simulation import *
from gen_data import *
import matplotlib.pyplot as plt
from optimizers import *
from learning_algorithms import *
from functions import *
from scipy.ndimage.filters import uniform_filter1d

### --- Define task and generate training and test data --- ###

task = Add_Task(5, 9, deterministic=True, tau_task=1)
N_train = 30000
N_test = 10000
data = task.gen_data(N_train, N_test)

### --- Initialize RNN object --- ###

n_in = task.n_in
n_hidden = 32
n_out = task.n_out

W_in  = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
W_rec = np.linalg.qr(np.random.normal(0, 1, (n_hidden, n_hidden)))[0]
W_out = np.random.normal(0, np.sqrt(1/(n_hidden)), (n_out, n_hidden))
W_FB = np.random.normal(0, np.sqrt(1/n_out), (n_out, n_hidden))

b_rec = np.zeros(n_hidden)
b_out = np.zeros(n_out)

alpha = 1

rnn = RNN(W_in, W_rec, W_out, b_rec, b_out,
          activation=tanh,
          alpha=alpha,
          output=softmax,
          loss=softmax_cross_entropy)

### --- Choose optimizer and learning algorithm --- ###

optimizer = Stochastic_Gradient_Descent(lr=0.03)
learn_alg = RTRL(rnn)


### --- Pick variables to track and run simulation --- ###

monitors = ['rnn.loss_']

sim = Simulation(rnn)
sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
        monitors=monitors,
        verbose=True,
        report_accuracy=False,
        report_loss=True,
        checkpoint_interval=None)

### --- Plot filtered training loss --- ###

plt.figure()
n_filter = 100
filtered_loss = uniform_filter1d(sim.mons['rnn.loss_'], n_filter)
plt.plot(filtered_loss)
plt.xlabel('training time (steps)')
plt.title('filtered training loss (cross entropy)')
plt.axhline(y=0.66, linestyle='--', color='C1')
plt.axhline(y=0.45, linestyle='--', color='C9')
plt.legend(['network performance', 'baseline', 'optimal'])

test_sim = Simulation(rnn)
test_sim.run(data,
              mode='test',
              monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
              verbose=False)

### --- Plot predictions and labels during test run --- ###

plt.figure()
plt.plot(test_sim.mons['rnn.y_hat'][:100,0], color='C3')
plt.plot(data['test']['Y'][:100,0], '--', color=('0.6'))
plt.legend(['prediction', 'label'])
plt.title('test performance')
plt.xlabel('time steps')

























