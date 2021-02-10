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
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    pass
from optimizers import *
from analysis_funcs import *
from learning_algorithms import *
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
    macro_configs = config_generator(name=['seq_reg'],
                                     i_start=list(range(0, 40000, 1000)))
    #macro_configs = config_generator(algorithm=['RFLO'])
    micro_configs = tuple(product(macro_configs, list(range(n_seeds))))

    params, i_seed = micro_configs[i_job]
    i_config = i_job//n_seeds
    np.random.seed(i_job)
    
if os.environ['HOME'] == '/Users/omarschall':
    params = {'name': 'seq_reg'}
    i_job = 0
    save_dir = '/Users/omarschall/vanilla-rtrl/library'

np.random.seed(0)
#task = Flip_Flop_Task(3, 0.05, input_magnitudes=None)
#task = Add_Task(4, 6, deterministic=True)
N_train = 12000
N_test = 2000
checkpoint_interval = 200
sigma = 0.01
name = params['name']
file_name = '{}'.format(name)
analysis_job_name = '{}'.format(name)
compare_job_name = 'comp_' + analysis_job_name
MODE = ['TRAIN', 'CHECK', 'ANALYZE', 'COMPARE', 'PLOT'][0]

""" -----------------------------------------"""
""" --- TRAIN MODEL AND SAVE CHECKPOINTS --- """
""" -----------------------------------------"""

if MODE == 'TRAIN': #Do this locally
    
    task_1 = Add_Task(6, 12, deterministic=True)
    task_2 = Add_Task(4, 9, deterministic=True)
    task_3 = Add_Task(8, 11, deterministic=True)
    task_4 = Add_Task(5, 6, deterministic=True)
    task_5 = Add_Task(6, 9, deterministic=True)
    task_6 = Add_Task(9, 12, deterministic=True)
    task = Multi_Task([task_1, task_2, task_3, task_4, task_5, task_6], context_input=True)
    data = task.gen_data(N_train, N_test)
    
    n_in = task.n_in
    n_hidden = 32
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
    
    #optimizer = SGD_Momentum(lr=0.01, mu=0.6, clip_norm=None)
    #learn_alg = RFLO(rnn, alpha=alpha, L2_reg=0.0001, L1_reg=0.0001)
    learn_alg = RTRL(rnn)
    #optimizer = Stochastic_Gradient_Descent(lr=0.005)
    
    comp_algs = []
    monitors = ['rnn.loss_']

    ### --- SIMULATION 1 --- ####    
    
    optimizer = Stochastic_Gradient_Descent(lr=0.01, clip_norm=None)
    
    sim = Simulation(rnn)
    sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
            sigma=sigma,
            comp_algs=comp_algs,
            monitors=monitors,
            verbose=True,
            report_accuracy=False,
            report_loss=True,
            checkpoint_interval=checkpoint_interval)
    
    ### --- SIMULATION 2 --- ####    
    
    #Create data
    # task = Add_Task(6, 10, deterministic=True)
    # data = task.gen_data(N_train, N_test)
    
    # learn_alg = Only_Output_Weights(rnn)
    # optimizer = Stochastic_Gradient_Descent(lr=0.01, clip_norm=None)
    
    # sim = Simulation(rnn)
    # sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
    #         sigma=sigma,
    #         comp_algs=comp_algs,
    #         monitors=monitors,
    #         verbose=True,
    #         report_accuracy=False,
    #         report_loss=True,
    #         checkpoint_interval=checkpoint_interval)
    
    # with open(os.path.join('saved_runs', file_name), 'wb') as f:
    #     pickle.dump(sim, f)


""" -----------------------"""
""" --- QUICK ANALYSIS --- """
""" -----------------------"""

if MODE == 'CHECK':

    
    with open(os.path.join('saved_runs', file_name), 'rb') as f:
        sim = pickle.load(f)
        
    data = task.gen_data(100, 10000)
    i_checkpoint = max(sim.checkpoints.keys())
    checkpoint = sim.checkpoints[i_checkpoint]
    
    #analyze context 0
    t1 = time.time()
    analyze_checkpoint(checkpoint, data, verbose=False,
                       sigma_pert=0.5, N=400, parallelize=False,
                       N_iters=3000, same_LR_criterion=1000,
                       context=None)
    t2 = time.time()
    print('Time for checkpoint analysis: {}'.format(t2 - t1))
    get_graph_structure(checkpoint, parallelize=False, epsilon=0.01, background_input=0)
    inputs = [np.array([1, 0, 0]), np.array([0, 1, 0]),
              np.array([0, 0, 1]), np.array([-1, 0, 0]),
              np.array([0, -1, 0]), np.array([0, 0, -1])]
    #inputs = [np.array([1, 0]), np.array([0, 1])]
    t3 = time.time()
    print('Time for autonomous graph: {}'.format(t3 - t2))
    get_input_dependent_graph_structure(checkpoint, inputs=inputs, contexts=None)
    t4 = time.time()
    print('Time for input graphs: {}'.format(t4 - t3))
    
    #PLOT INPUT
    i_input = 2
    transform = partial(np.dot, b=checkpoint['V'])
    ssa = State_Space_Analysis(checkpoint, data, transform=transform)
    ssa = plot_checkpoint_results(checkpoint, data, ssa=ssa,
                                  plot_cluster_means=False,
                                  eig_norm_color=False,
                                  plot_test_points=True,
                                  plot_fixed_points=True,
                                  plot_graph_structure=True,
                                  n_test_samples=None,
                                  plot_uncategorized_points=True,
                                  graph_key='adjmat_input_{}'.format(i_input))
    ssa_2 = State_Space_Analysis(checkpoint, data, transform=transform)
    ssa_2 = plot_checkpoint_results(checkpoint, data, ssa=ssa_2,
                                  plot_cluster_means=False,
                                  eig_norm_color=False,
                                  plot_test_points=True,
                                  plot_fixed_points=False,
                                  plot_graph_structure=True,
                                  n_test_samples=None,
                                  graph_key='adjmat_input_{}'.format(i_input+3))
    plot_input_dependent_topology(checkpoint, i_input=i_input)
    plot_input_dependent_topology(checkpoint, i_input=i_input + 3)
    
    plot_input_dependent_topology(checkpoint, i_input=None)

""" ----------------------------"""
""" --- ANALYZE CHECKPOINTS --- """
""" ----------------------------"""
   
if MODE == 'ANALYZE': #Do this on cluster with array jobs on 'i_start'
    
    
    name = params['name']
    print('Analyzing checkpoints...')
    
    #contexts = [np.array([0, 0, 0, 1]), np.array([0, 0, 0, -1])]
    contexts = [None]
    # inputs = [np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]),
    #           np.array([0, 0, 1, 0]), np.array([-1, 0, 0, 0]),
    #           np.array([0, -1, 0, 0]), np.array([0, 0, -1, 0])]
    inputs = [np.array([1, 0, 0]), np.array([0, 1, 0]),
              np.array([0, 0, 1]), np.array([-1, 0, 0]),
              np.array([0, -1, 0]), np.array([0, 0, -1])]
    
    result = {}
    
    data = task.gen_data(100, 30000)
    #big_data = task.gen_data(100, 500000)
    
    #Progress logging
    scratch_path = '/scratch/oem214/vanilla-rtrl/'
    log_path = os.path.join(scratch_path, 'log/' + analysis_job_name) + '_{}.log'.format(i_job)
    
    #Retrieve data
    with open(os.path.join('saved_runs', file_name), 'rb') as f:
        #sim = pickle.load(f)
        checkpoints = pickle.load(f)
    
    for i_checkpoint in range(params['i_start'], params['i_start'] + 1000, 100):

        with open(log_path, 'a') as f:
            f.write('Analyzing chekpoint {}\n'.format(i_checkpoint))
        
        #checkpoint = sim.checkpoints[i_checkpoint]
        checkpoint = checkpoints[i_checkpoint]
        #checkpoint_flip = deepcopy(checkpoint)
        
        #analyze context 0
        analyze_checkpoint(checkpoint, data, verbose=False,
                           sigma_pert=0.5, N=500, parallelize=False,
                           N_iters=6000, same_LR_criterion=5000,
                           context=contexts[0], sigma=sigma)
        
        get_graph_structure(checkpoint, parallelize=False, epsilon=0.01, background_input=0)
        get_input_dependent_graph_structure(checkpoint, inputs=inputs, contexts=None)
        #train_VAE(checkpoint, big_data, T=10, latent_dim=128, lr=0.001)
        
        
        if name == 'interp':
            i_checkpoint_ = i_checkpoint + 10000
        else:
            i_checkpoint_ = i_checkpoint
        result['checkpoint_{}'.format(i_checkpoint_)] = deepcopy(checkpoint)
        
        #analyze context 1
        # analyze_checkpoint(checkpoint_flip, data, verbose=False,
        #                    sigma_pert=0.5, N=600, parallelize=False,
        #                    N_iters=8000, same_LR_criterion=7000,
        #                    context=contexts[1])
        
        # get_graph_structure(checkpoint_flip, parallelize=False, epsilon=0.01, background_input=contexts[1])

        # # inputs = [np.array([1, 0, 0]), np.array([0, 1, 0]),
        # #           np.array([0, 0, 1]), np.array([-1, 0, 0]),
        # #           np.array([0, -1, 0]), np.array([0, 0, -1])]
        # get_input_dependent_graph_structure(checkpoint_flip, inputs=inputs, contexts=contexts)
        # result['checkpoint_{}_flip'.format(i_checkpoint)] = deepcopy(checkpoint_flip)

""" ----------------------------"""
""" --- CALCULATE DISTANCES --- """
""" ----------------------------"""

if MODE == 'COMPARE': #Do this on cluster with single job (for now)

    print('Comparing checkpoints...')
    
    result = {}
    
    n_off_diagonals = 1
    
    #big_data = task.gen_data(100, 20000)
    data = task.gen_data(100, 10000)
    scratch_path = '/scratch/oem214/vanilla-rtrl/'
    if os.environ['HOME'] == '/home/oem214':
        data_path = os.path.join(scratch_path, 'library/' + analysis_job_name)
    else:
        data_path = os.path.join('/Users/omarschall/cluster_results/vanilla-rtrl', analysis_job_name)
    log_path = os.path.join(scratch_path, 'log/' + compare_job_name) + '.log'
    
    #print(data_path)
    
    #Unpack data
    indices, checkpoints = unpack_analysis_results(data_path)
    
    print(indices)
    #set_trace()
    
    #Initialize dissimilarity matrices
    n_checkpoints = len(checkpoints)
    wasserstein_distances = np.zeros((n_checkpoints, n_checkpoints))
    #VAE_distances = np.zeros((n_checkpoints, n_checkpoints))
    PC1_distances = np.zeros((n_checkpoints, n_checkpoints))
    PC2_distances = np.zeros((n_checkpoints, n_checkpoints))
    SVCCA_distances = np.zeros((n_checkpoints, n_checkpoints))
    #graph_distances = np.zeros((n_checkpoints, n_checkpoints))
    #input_graph_distances = np.zeros(n_checkpoints)
    aligned_graph_distances = np.zeros((n_checkpoints, n_checkpoints))
    node_diff_distances = np.zeros((n_checkpoints, n_checkpoints))
    rec_weight_distances = np.zeros((n_checkpoints, n_checkpoints))
    # output_weight_distances = np.zeros(n_checkpoints)
    # participation_coefficients = np.zeros(n_checkpoints)
    
    for i in range(len(indices) - 1):
        
        #print(i)
        
        if i % 10 == 0 and os.environ['HOME'] == '/home/oem214':
            with open(log_path, 'a') as f:
                f.write('Calculating distance row {}\n'.format(i))
        elif i % 10 == 0:
            print(i)
        
        for j in range(i + 1, i + 2):
        
            # try:
            i_index = indices[i]
            j_index = indices[j]
            # except IndexError:
            #     continue
            
            try:
                checkpoint_1 = checkpoints['checkpoint_{}'.format(i_index)]
                checkpoint_2 = checkpoints['checkpoint_{}'.format(j_index)]
            except KeyError:
                continue
            
            #align checkpoints
            if True:
                align_checkpoints(checkpoint_2, checkpoint_1)
                align_checkpoints(checkpoint_2, checkpoint_1)
                #print('checkpoint 2 keys:/n', checkpoint_2.keys())
                #print('checkpoint 1 keys:/n', checkpoint_1.keys())
            
            wasserstein_distances[i, j] = wasserstein_distance(checkpoint_1,
                                                                 checkpoint_2)
            
            # VAE_distances[i, i+j] = VAE_distance(checkpoint_1, checkpoint_2,
            #                                       big_data=big_data)
            
            PC1_distances[i, j] = PC_distance_1(checkpoint_1,
                                                  checkpoint_2)
            PC2_distances[i, j] = PC_distance_2(checkpoint_1,
                                                  checkpoint_2)
            SVCCA_distances[i, j] = SVCCA_distance(checkpoint_1,
                                                       checkpoint_2,
                                                       data=data, R=3)
            # graph_distances[i, i+j] = graph_distance(checkpoint_1,
            #                                           checkpoint_2)
            # input_graph_distances[i, i+j] = input_dependent_graph_distance(checkpoint_1,
            #                                                                 checkpoint_2)
            aligned_graph_distances[i, j] = aligned_graph_distance(checkpoint_1,
                                                                checkpoint_2,
                                                                node_diff_penalty=0)
            node_diff_distances[i, j] = node_diff_distance(checkpoint_1, checkpoint_2)
            rec_weight_distances[i, j] = rec_weight_distance(checkpoint_1,
                                                          checkpoint_2)
            #output_weight_distances[i] = output_weight_distance(checkpoint_1,
            #                                                    checkpoint_2)
            #participation_coefficients[i] = checkpoint_1['participation_coef']
        
    result['wasserstein_distances'] = wasserstein_distances + wasserstein_distances.T
    #result['VAE_distances'] = VAE_distances + VAE_distances.T
    result['PC1_distances'] = PC1_distances + PC1_distances.T
    result['PC2_distances'] = PC2_distances + PC2_distances.T
    result['SVCCA_distances'] = SVCCA_distances + SVCCA_distances.T
    #result['graph_distances'] = graph_distances
    #result['input_graph_distances'] = input_graph_distances
    result['aligned_graph_distances'] = aligned_graph_distances + aligned_graph_distances.T
    result['node_diff_distances'] = node_diff_distances + node_diff_distances.T
    result['rec_weight_distances'] = rec_weight_distances + rec_weight_distances.T
    #result['output_weight_distances'] = output_weight_distances
    #result['participation_coefficients'] = participation_coefficients

""" ---------------------"""
""" --- PLOT RESULTS --- """
""" ---------------------"""

if MODE == 'PLOT':
    
    local_results = '/Users/omarschall/cluster_results/vanilla-rtrl/'
    figs_path = '/Users/omarschall/weekly-reports/report_10-29-2020/figs'
    comp_data_path = os.path.join(local_results, compare_job_name, 'result_0')
    data_path = os.path.join(local_results, analysis_job_name)
    indices, checkpoints = unpack_analysis_results(data_path)
    node_distances = []
    losses = []
    #grad_norms = []
    rec_params = []
    for i_index, i in enumerate(indices):
        ref_checkpoint = checkpoints['checkpoint_{}'.format(indices[i_index - 1])]
        checkpoint = checkpoints['checkpoint_{}'.format(indices[i_index])]
        align_checkpoints(checkpoint, ref_checkpoint)
        align_checkpoints(checkpoint, ref_checkpoint)
        d = norm(np.array(checkpoint['corr_node_distances']))
        node_distances.append(d)
        losses.append(checkpoint['test_loss'])
        rnn = checkpoint['rnn']
        params = np.concatenate([rnn.W_rec.flatten(), rnn.W_in.flatten(), rnn.b_rec])
        rec_params.append(params)
        #grad_norms.append(norm(checkpoint['learn_alg'].rec_grads))
    losses = np.array(losses)
    rec_params = np.array(rec_params)
    norms = np.square(rec_params[1:] - rec_params[:-1]).sum(1)
    node_distances = np.array(node_distances)
    
    #collect all signals
    signals = {'losses': losses, 'node_distances': node_distances, 'norms': norms}
    try:
        with open(comp_data_path, 'rb') as f:
            result = pickle.load(f)
    
        for i_key, key in enumerate(result.keys()):
            
            if 'distances' in key or 'participation' in key:
                
                x = np.diag(result[key][:-1,1:])
                signals[key] = x.copy()

    except FileNotFoundError:
        pass
    
    #Plot all signals
    
    fig1 = plt.figure(figsize=(10,10))
    leg = []
    for i_key, key in enumerate(signals):
        
        x = signals[key].copy()
        
        x_max = np.amax(x)
        x_min = np.amin(x)
        x = (x - x_min) / (x_max - x_min)
        
        plt.plot(x - 1.2 * i_key)
        leg.append(key)
                
    plt.legend(leg)
    plt.yticks([])
    
    
    
    data = task.gen_data(100, 10000)
    # sparse_inputs_task = Flip_Flop_Task(task.n_bit, 0.001)
    
    i_checkpoint = 7000
    checkpoint = checkpoints['checkpoint_{}'.format(i_checkpoint)]
    transform = Vanilla_PCA(checkpoint, data)
    ssa_2 = State_Space_Analysis(checkpoint, data, transform=transform)
    ssa_2 = plot_checkpoint_results(checkpoint, data, ssa=ssa_2,
                                    plot_cluster_means=False,
                                    eig_norm_color=False,
                                    plot_test_points=True,
                                    plot_fixed_points=False,
                                    plot_graph_structure=True,
                                    n_test_samples=2,
                                    graph_key='adjacency_matrix')
    
        
    fig2 = plot_output_from_checkpoint(checkpoint, data, n_PCs=3)



if os.environ['HOME'] == '/Users/omarschall' and MODE == 'TRAIN':
    
    pass
    # task = Flip_Flop_Task(3, 0.05, input_magnitudes=None, dim_mask=[1,1,0])
    # data = task.gen_data(N_train, N_test)
    
    # i_checkpoint = max(sim.checkpoints.keys())
    # plot_output_from_checkpoint(sim.checkpoints[i_checkpoint], data, n_PCs=rnn.n_out)
    #i_checkpoint = max(noisy_sim.checkpoints.keys())
    #plot_output_from_checkpoint(sim.checkpoints[N_train - 1], data, n_PCs=rnn.n_out)
    
    # rnn = sim.checkpoints[N_train - 1]['rnn']
    # test_sim = Simulation(rnn)
    # test_sim.run(data,
    #               mode='test',
    #               monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
    #               verbose=False)

    # plt.figure()
    # plt.plot(data['test']['X'][:, 0] + 2.5, (str(0.6)), linestyle='--')
    # plt.plot(data['test']['Y'][:, 0] + 2.5, 'C0')
    # plt.plot(test_sim.mons['rnn.y_hat'][:, 0] + 2.5, 'C3')
    # plt.plot(data['test']['X'][:, 1], (str(0.6)), linestyle='--')
    # plt.plot(data['test']['Y'][:, 1], 'C0')
    # plt.plot(test_sim.mons['rnn.y_hat'][:, 1], 'C3')
    # try:
    #     plt.plot(data['test']['X'][:, 2] - 2.5, (str(0.6)), linestyle='--')
    #     plt.plot(data['test']['Y'][:, 2] - 2.5, 'C0')
    #     plt.plot(test_sim.mons['rnn.y_hat'][:, 2] - 2.5, 'C3')
    # except IndexError:
    #     pass
    # plt.xlim([0, 100])
    # plt.yticks([])
    # plt.xlabel('time steps')
    
    plt.figure()
    for key in monitors:
        plt.plot(sim.mons[key])
    
    losses = get_multitask_loss_from_checkpoints(sim, task, 300)
    plt.figure()
    for i in range(task.n_tasks):
        plt.plot(losses['task_{}_loss'.format(i)])
    
    # test_loss_A = []
    # test_loss_B = []
    # for i_checkpoint in range(len(sim.checkpoints)):
    #     rnn = sim.checkpoints[i_checkpoint]
        
    #     test_sim_A = 
        
    #     test_loss_A
        
    

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












