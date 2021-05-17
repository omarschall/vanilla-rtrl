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
    #i_job = 61
    #macro_configs = config_generator(i_start=list(range(0, 200000, 1000)))
    macro_configs = config_generator(name=['best_grid_2'],
                                     i_start=list(range(0, 2000000, 5000)))
    #macro_configs = config_generator(algorithm=['RFLO'])
    micro_configs = tuple(product(macro_configs, list(range(n_seeds))))

    params, i_seed = micro_configs[i_job]
    i_config = i_job//n_seeds
    np.random.seed(i_job)
    
if os.environ['HOME'] == '/Users/omarschall':
    params = {'name': 'diag',
              'algorithm': 'RFLO',
              'tasks': 'all'}
    i_job = 0
    save_dir = '/Users/omarschall/vanilla-rtrl/library'

i_seed = np.random.randint(0, 1000)
np.random.seed(i_seed)
task = Flip_Flop_Task(3, 0.05, input_magnitudes=None)
#task = Add_Task(4, 6, deterministic=True)
N_train = 2000000
N_test = 2000
checkpoint_interval = 5000
sigma = 0
name = params['name']
#algorithm = params['algorithm']
file_name = '{}'.format(name)
analysis_job_name = '{}'.format(name)
compare_job_name = 'comp_{}'.format(analysis_job_name)
task_name = '{}_task'.format(name)
MODE = ['TRAIN', 'CHECK', 'ANALYZE', 'COMPARE', 'PLOT'][4]
#print(sklearn.__version__)

""" -----------------------------------------"""
""" --- TRAIN MODEL AND SAVE CHECKPOINTS --- """
""" -----------------------------------------"""

if MODE == 'TRAIN': #Do this locally
    
    if params['tasks'] == 'diag':
        task_1 = Flip_Flop_Task(3, 0.05, dim_mask=[1,0,0])
        task_2 = Flip_Flop_Task(3, 0.05, dim_mask=[0,1,0])
        task_3 = Flip_Flop_Task(3, 0.05, dim_mask=[0,0,1])
        task = Multi_Task([task_1, task_2, task_3], context_input=False)
        combined_task = Flip_Flop_Task(3, 0.05, dim_mask=[1,1,1])
        proj_task_1 = Flip_Flop_Task(3, 0.05, dim_mask=[1,0,0])
        proj_task_2 = Flip_Flop_Task(3, 0.05, dim_mask=[1,1,0])
        proj_tasks = [proj_task_1, proj_task_2]
        N_train = [{'task_id': 0, 'N': 20000},
                    {'task_id': 1, 'N': 10000},
                    {'task_id': 2, 'N': 10000}]
        data = task.gen_data(N_train, N_test)
        
    if params['tasks'] == 'staircase':
        task_1 = Flip_Flop_Task(3, 0.05, dim_mask=[1,0,0])
        task_2 = Flip_Flop_Task(3, 0.05, dim_mask=[1,1,0])
        task_3 = Flip_Flop_Task(3, 0.05, dim_mask=[1,1,1])
        task = Multi_Task([task_1, task_2, task_3], context_input=False)
        combined_task = Flip_Flop_Task(3, 0.05, dim_mask=[1,1,1])
        proj_tasks = None
        N_train = [{'task_id': 0, 'N': 5000},
                    {'task_id': 1, 'N': 10000},
                    {'task_id': 2, 'N': 20000}]
        data = task.gen_data(N_train, N_test)
        
    if params['tasks'] == 'all':
        task = Flip_Flop_Task(3, 0.05, dim_mask=[1,1,1])
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
              output=identity,
              loss=mean_squared_error)
    
    ###
    # 1. Compare different ways of building cube, standard learning vs.
    #       CL with [1,0,0]->[1,1,0]->[1,1,1] or [1,0,0]->[0,1,0]->[0,0,1]
    #       and also with faiure case
    # 2. Maybe figure out a  more automated visualization method?
    # 3. Validate a PC metric
    # 4. FIGURE OUT THE MASK THING
    
    #optimizer = SGD_Momentum(lr=0.01, mu=0.6, clip_norm=None)
    
    if params['algorithm'] == 'RFLO':
        learn_alg = RFLO(rnn, alpha=alpha, L2_reg=0.0001, L1_reg=0.0001)
    if params['algorithm'] == 'REIN':
        learn_alg = REINFORCE(rnn, sigma=sigma)
    
    comp_algs = []
    monitors = ['rnn.a', 'rnn.x', 'rnn.W_rec']
    monitors = ['rnn.W_rec']
    monitors = []

    ### --- SIMULATION 1 --- ####    
    
    #optimizer = Stochastic_Gradient_Descent(lr=0.05, clip_norm=None)
    optimizer = SGD_Momentum(lr=0.01, mu=0.6)
    
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
            Duncker_proj_tasks=None)
            #lr_Duncker=0.1)
    
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
    
    #Progress logging
    scratch_path = '/scratch/oem214/vanilla-rtrl/'
    log_path = os.path.join(scratch_path, 'log/' + analysis_job_name) + '_{}.log'.format(i_job)
    
    #Retrieve data
    with open(os.path.join('saved_runs', file_name), 'rb') as f:
        #checkpoints = pickle.load(f)
        sim = pickle.load(f)
        
    with open(os.path.join('fp_tasks', task_name), 'rb') as f:
        #checkpoints = pickle.load(f)
        task = pickle.load(f)
    
    contexts = [None]
    # inputs = [np.array([1, 0, 0]), np.array([0, 1, 0]),
    #           np.array([0, 0, 1]), np.array([-1, 0, 0]),
    #           np.array([0, -1, 0]), np.array([0, 0, -1])]
    inputs = [np.eye(task.n_in)[i] for i in range(task.n_in)]
    result = {}
    
    data = task.gen_data(100, 30000)
    
    for i_checkpoint in range(params['i_start'], params['i_start'] + checkpoint_interval, checkpoint_interval):

        with open(log_path, 'a') as f:
            f.write('Analyzing chekpoint {}\n'.format(i_checkpoint))
            
        #checkpoint = checkpoints[i_checkpoint]
        checkpoint = sim.checkpoints[i_checkpoint]
        analyze_checkpoint(checkpoint, data, verbose=False,
                           sigma_pert=0.5, N=500, parallelize=False,
                           N_iters=6000, same_LR_criterion=5000,
                           context=contexts[0], sigma=sigma)
        
        get_graph_structure(checkpoint, parallelize=False, epsilon=0.01, background_input=0)
        get_input_dependent_graph_structure(checkpoint, inputs=inputs, contexts=None)

        result['checkpoint_{}'.format(i_checkpoint)] = deepcopy(checkpoint)

""" ----------------------------"""
""" --- CALCULATE DISTANCES --- """
""" ----------------------------"""

if MODE == 'COMPARE': #Do this on cluster with single job (for now)

    print('Comparing checkpoints...')
    
    result = {}
    
    n_off_diagonals = 1
    
    #big_data = task.gen_data(100, 20000)
    #data = task.gen_data(100, 10000)
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
    n_checkpoints = len(indices)
    #wasserstein_distances = np.zeros((n_checkpoints, n_checkpoints))
    #VAE_distances = np.zeros((n_checkpoints, n_checkpoints))
    #PC1_distances = np.zeros((n_checkpoints, n_checkpoints))
    #PC2_distances = np.zeros((n_checkpoints, n_checkpoints))
    PC3_distances = np.zeros((n_checkpoints, n_checkpoints))
    #SVCCA_distances = np.zeros((n_checkpoints, n_checkpoints))
    #graph_distances = np.zeros((n_checkpoints, n_checkpoints))
    #input_graph_distances = np.zeros(n_checkpoints)
    aligned_graph_distances = np.zeros((n_checkpoints, n_checkpoints))
    node_diff_distances = np.zeros((n_checkpoints, n_checkpoints))
    rec_weight_distances = np.zeros((n_checkpoints, n_checkpoints))
    # output_weight_distances = np.zeros(n_checkpoints)
    #participation_coefficients = np.zeros(n_checkpoints)
    
    #checkpoint_final = checkpoints['checkpoint_{}'.format(indices[-1])]
    
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
            try:
                align_checkpoints(checkpoint_2, checkpoint_1, n_inputs=4)
                align_checkpoints(checkpoint_2, checkpoint_1, n_inputs=4)
            except ValueError:
                break
                #print('checkpoint 2 keys:/n', checkpoint_2.keys())
                #print('checkpoint 1 keys:/n', checkpoint_1.keys())
            
            # wasserstein_distances[i, j] = wasserstein_distance(checkpoint_1,
            #                                                      checkpoint_2)
            
            # VAE_distances[i, i+j] = VAE_distance(checkpoint_1, checkpoint_2,
            #                                       big_data=big_data)
            
            # PC1_distances[i, j] = PC_distance_1(checkpoint_1,
            #                                       checkpoint_2)
            # PC2_distances[i, j] = PC_distance_2(checkpoint_1,
            #                                       checkpoint_2)
            # PC3_distances[i, j] = PC_distance_3(checkpoint_1,
            #                                     checkpoint_2,
            #                                     N_avg=25, N_test=1000,
            #                                     task=task)
            # SVCCA_distances[i, j] = SVCCA_distance(checkpoint_1,
            #                                            checkpoint_2,
            #                                            data=data, R=3)
            # graph_distances[i, i+j] = graph_distance(checkpoint_1,
            #                                           checkpoint_2)
            # input_graph_distances[i, i+j] = input_dependent_graph_distance(checkpoint_1,
            #                                                                 checkpoint_2)
            aligned_graph_distances[i, j] = aligned_graph_distance(checkpoint_1,
                                                                checkpoint_2,
                                                                node_diff_penalty=0,
                                                                n_inputs=4)
            node_diff_distances[i, j] = node_diff_distance(checkpoint_1, checkpoint_2)
            rec_weight_distances[i, j] = rec_weight_distance(checkpoint_1,
                                                          checkpoint_2)
            #output_weight_distances[i] = output_weight_distance(checkpoint_1,
            #                                                    checkpoint_2)
            #participation_coefficients[i] = checkpoint_1['participation_coef']
        
    #result['wasserstein_distances'] = wasserstein_distances + wasserstein_distances.T
    #result['VAE_distances'] = VAE_distances + VAE_distances.T
    #result['PC1_distances'] = PC1_distances + PC1_distances.T
    #result['PC2_distances'] = PC2_distances + PC2_distances.T
    #result['PC3_distances'] = PC3_distances + PC3_distances.T
    #result['SVCCA_distances'] = SVCCA_distances + SVCCA_distances.T
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
    
    #meeting notes:
    #- do windowed standard deviation for normalized trace comparison
    #
    
    #with open(os.path.join('fp_tasks', task_name), 'rb') as f:
        #checkpoints = pickle.load(f)
    #    task = pickle.load(f)
    
    local_results = '/Users/omarschall/cluster_results/vanilla-rtrl/'
    figs_path = '/Users/omarschall/career-stuff/thesis_committee/meeting_april-2021'
    comp_data_path = os.path.join(local_results, compare_job_name, 'result_0')
    data_path = os.path.join(local_results, analysis_job_name)
    indices, checkpoints = unpack_analysis_results(data_path)
    node_distances = []
    losses = []
    #grad_norms = []
    rec_params = []
    CCA_distances = []
    largest_evals = []
    weight_stds = []
    CCA_data = task.gen_data(10, 1000)
    for i_index in range(1, len(indices)):
        ref_checkpoint = checkpoints['checkpoint_{}'.format(indices[i_index - 1])]
        checkpoint = checkpoints['checkpoint_{}'.format(indices[i_index])]
        if i_index % 100 == 0:
            print(i_index)
        try:
            align_checkpoints(checkpoint, ref_checkpoint, n_inputs=4)
            align_checkpoints(checkpoint, ref_checkpoint, n_inputs=4)
        except ValueError:
            continue
        d = norm(np.array(checkpoint['corr_node_distances']))
        node_distances.append(d)
        losses.append(checkpoint['test_loss'])
        rnn = checkpoint['rnn']
        params = np.concatenate([rnn.W_rec.flatten(), rnn.W_in.flatten(), rnn.b_rec])
        rec_params.append(params)
        eigs, vecs = np.linalg.eig(checkpoint['rnn'].W_rec)
        largest_evals.append(np.abs(eigs[0]))
        weight_stds.append(rnn.W_rec.std())
        #CCA_distances.append(SVCCA_distance(checkpoint, ref_checkpoint, CCA_data))
        #grad_norms.append(norm(checkpoint['learn_alg'].rec_grads))
    losses = np.array(losses)
    rec_params = np.array(rec_params)
    largest_evals = np.array(largest_evals)
    norms = np.square(rec_params[1:] - rec_params[:-1]).sum(1)
    node_distances = np.array(node_distances)
    weight_stds = np.array(weight_stds)
    
    #collect all signals
    signals = {'losses': losses, 'node_distances': node_distances,
               'norms': norms, 'largest_evals': largest_evals,
               'weight_stds': weight_stds}
    signals = {'losses': losses}
    try:
        with open(comp_data_path, 'rb') as f:
            result = pickle.load(f)
    
        for i_key, key in enumerate(result.keys()):
            
            #if 'distances' in key or 'participation' in key:
            if 'aligned' in key:
                
                x = np.diag(result[key][:-1,1:])
                signals[key] = x.copy()

    except FileNotFoundError:
        print('file not found')
        pass
    
    #Plot all signals
    
    fig1 = plt.figure(figsize=(10,2))
    leg = []
    colors = ['#89949B', '#E89C15']
    for i_key, key in enumerate(signals):
        
        x = signals[key].copy()
        
        x_max = np.amax(x)
        x_min = np.amin(x)
        
        # if 'PC' in key:
        #     x_max = 1
        #     x_min = 0
        
        x = (x - x_min) / (x_max - x_min)
        
        plt.plot(x - 1.2 * i_key, color=colors[i_key])
        if key == 'largest_evals':
            unit_circle = (1 - x_min) / (x_max - x_min)
            plt.axhline(y=unit_circle - 1.2 * i_key, color=('0.6'), linestyle='--', label='_nolegend_')
        leg.append(key)
                
    #plt.legend(leg)
    plt.xticks([0, 100, 200, 300, 400], ['0', '500k', '1M', '1.5M', '2M'])
    plt.yticks([])
    
    
    
    data = task.gen_data(100, 10000)
    # sparse_inputs_task = Flip_Flop_Task(task.n_bit, 0.001)
    
    i_checkpoint = max(indices)
    checkpoint = checkpoints['checkpoint_{}'.format(i_checkpoint)]
    transform = Vanilla_PCA(checkpoint, data)
    ssa_2 = State_Space_Analysis(checkpoint, data, transform=transform)
    ssa_2 = plot_checkpoint_results(checkpoint, data, ssa=ssa_2,
                                    plot_cluster_means=False,
                                    eig_norm_color=False,
                                    plot_test_points=True,
                                    plot_fixed_points=False,
                                    plot_graph_structure=True,
                                    n_test_samples=None,
                                    graph_key='adjacency_matrix')
    
    plot_input_dependent_topology(checkpoint)
        
    fig2 = plot_output_from_checkpoint(checkpoint, data, n_PCs=3)



if os.environ['HOME'] == '/Users/omarschall' and MODE == 'TRAIN':
    
    pass
    #task = Flip_Flop_Task(3, 0.05, input_magnitudes=None, dim_mask=[1,1,0])
    #test_data = combined_task.gen_data(0, 2000)
    test_data = task.gen_data(0, 2000)
    
    i_checkpoint = max(sim.checkpoints.keys())
    plot_output_from_checkpoint(sim.checkpoints[i_checkpoint], test_data, n_PCs=rnn.n_out)
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
    
    # plt.figure()
    # for key in monitors:
    #     plt.plot(sim.mons[key])
    
    # losses = get_multitask_loss_from_checkpoints(sim, task, 300)
    # plt.figure()
    # for i in range(task.n_tasks):
    #     plt.plot(losses['task_{}_loss'.format(i)], color='C{}'.format(i+1))
    # plt.legend(['Task {} Loss'.format(i) for i in range(task.n_tasks)])
        
    # losses = get_loss_from_checkpoints(sim, combined_task, 1000)
    # plt.figure()
    # plt.plot(losses)
    # plt.legend(['Combined Task Loss'])
    
    # participation_coeffs = []
    # norms = []
    # for i in range(len(sim.mons['rnn.W_rec'])):
        
    #     eigs, vecs = np.linalg.eig(sim.mons['rnn.W_rec'][i])
    #     eig_norms = np.abs(eigs)
    #     #participation_coeffs.append(np.sum(np.square(np.abs(eigs))) / np.square(np.abs(np.sum(eigs))))
    #     participation_coeffs.append(np.sqrt(eig_norms).sum() / np.sqrt(eig_norms.sum()))
    #     norms.append(norm(sim.mons['rnn.W_rec'][i]))
        
    # plt.figure()
    # plt.plot(participation_coeffs, color='C8')
    # plt.plot(norms, color='C9')
    # plt.ylim([0, 8])
    # plt.legend(['W participation coefficient', 'W norm'])
    
    checkpoint_1 = checkpoints['checkpoint_10000']; checkpoint_2 = checkpoints['checkpoint_10010']
    
    Vs_1 = []
    Vs_2 = []
    N = 4
    t = time()
    for _ in range(N):
        print(time() - t)
        checkpoint_1['rnn'].a = np.random.normal(0, 1, 32)
        test_data = combined_task.gen_data(0, 5000)
        transform = Vanilla_PCA(checkpoint_1, test_data)
        Vs_1.append(transform(np.eye(32)))
        checkpoint_1['rnn'].a = np.random.normal(0, 1, 32)
        transform = Vanilla_PCA(checkpoint_1, test_data)
        Vs_2.append(transform(np.eye(32)))
        
    Vs_total = Vs_1 + Vs_2
        
    Ds = np.zeros((2*N, 2*N))
    for i in range(2*N):
        for j in range(2*N):
            #Ds[i,j] = np.sum(np.abs(Vs_total[i].T.dot(Vs_total[j]))) / 3
            #Ds[i,j] = np.abs(np.sum(Vs_total[i] * Vs_total[j])) / 3
            V_1 = Vs_total[i]
            V_2 = Vs_total[j]
            Ds[i,j] = 1 - np.sort(np.abs(V_1.T.dot(V_2)).flatten())[-3:].sum() / 3
            #Ds[i,j] = np.sum(np.abs(Vs_total[i].T.dot(Vs_total[j]))) / 3

    # off_diags = Ds.copy()
    # d = np.diag(np.ones(N) * np.nan)
    # d_1 = np.vstack([d,d])
    # d_2 = np.hstack([d_1, d_1])
    # off_diags += d_2
    
    # Ds = np.zeros((5, 8))
    # for i_c2, c2 in enumerate(range(11000, 19000, 1000)):
    #     for i_n, n in enumerate([100, 200, 500, 1000, 2000]):
    #         checkpoint_2 = checkpoints['checkpoint_{}'.format(c2)]
    #         Ds[i_n, i_c2] = PC_distance_3(checkpoint_1, checkpoint_2, N_avg=10, N_test=n, task=task)
        

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












