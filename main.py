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
    macro_configs = config_generator(algorithm=['RFLO'],
                                     name=['combined'],
                                     i_start=list(range(0, 10000, 100)))
    #macro_configs = config_generator(algorithm=['RFLO'])
    micro_configs = tuple(product(macro_configs, list(range(n_seeds))))

    params, i_seed = micro_configs[i_job]
    i_config = i_job//n_seeds
    np.random.seed(i_job)
    
if os.environ['HOME'] == '/Users/omarschall':
    params = {'algorithm': 'REIN', 'name': 'fail'}
    i_job = 0
    save_dir = '/Users/omarschall/vanilla-rtrl/library'

np.random.seed(1)
task = Flip_Flop_Task(3, 0.05, input_magnitudes=None)
N_train = 20000
N_test = 5000
checkpoint_interval = 100
name = params['name']
file_name = '{}_{}'.format(name, params['algorithm'])
analysis_job_name = '{}_{}'.format(name, params['algorithm'])
compare_job_name = 'comp_' + analysis_job_name
figs_path = '/Users/omarschall/weekly-reports/report_08-19-2020/figs'
MODE = ['TRAIN', 'CHECK', 'ANALYZE', 'COMPARE', 'PLOT'][0]

""" -----------------------------------------"""
""" --- TRAIN MODEL AND SAVE CHECKPOINTS --- """
""" -----------------------------------------"""

if MODE == 'TRAIN': #Do this locally

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
    sigma = 0.5
    
    rnn = RNN(W_in, W_rec, W_out, b_rec, b_out,
              activation=tanh,
              alpha=alpha,
              output=identity,
              loss=mean_squared_error)
    
    optimizer = SGD_Momentum(lr=0.0005, mu=0.6, clip_norm=10)
    #optimizer = Stochastic_Gradient_Descent(lr=0.001)
    if params['algorithm'] == 'E-BPTT':
        learn_alg = Efficient_BPTT(rnn, 5, L2_reg=0.0001, L1_reg=0.0001)
    elif params['algorithm'] == 'RFLO':
        learn_alg = RFLO(rnn, alpha=alpha, L2_reg=0.0001, L1_reg=0.0001)
    elif params['algorithm'] == 'RTRL':
        learn_alg = RTRL(rnn, M_decay=0.3, L2_reg=0.0001, L1_reg=0.0001)
    elif params['algorithm'] == 'REIN':
        learn_alg = REINFORCE(rnn, sigma=sigma, L2_reg=0.001, L1_reg=0.001,
                              decay=0.1)
    elif params['algorithm'] == 'KF-RTRL':
        learn_alg = KF_RTRL(rnn, L2_reg=0.0001, L1_reg=0.0001)
    
    comp_algs = []
    monitors = []
    
    sim = Simulation(rnn)
    sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
            sigma=sigma,
            comp_algs=comp_algs,
            monitors=monitors,
            verbose=True,
            report_accuracy=False,
            report_loss=True,
            checkpoint_interval=checkpoint_interval)
    
    # if True:
    #     sim_ = Simulation(rnn)
    #     sim_ = linearly_interpolate_checkpoints(sim_,
    #                                             start_checkpoint=sim.checkpoints[0],
    #                                             end_checkpoint=sim.checkpoints[99000],
    #                                             density=100)
        
    with open(os.path.join('saved_runs', file_name), 'wb') as f:
        pickle.dump(sim, f)
        
    # with open(os.path.join('saved_runs', 'interp_RFLO'), 'wb') as f:
    #     pickle.dump(sim_, f)

""" -----------------------"""
""" --- QUICK ANALYSIS --- """
""" -----------------------"""

if MODE == 'CHECK':

    
    with open(os.path.join('saved_runs', file_name), 'rb') as f:
        sim = pickle.load(f)
        
    data = task.gen_data(100, 10000)
    checkpoint = sim.checkpoints[N_train - 1]
    
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
    t3 = time.time()
    print('Time for autonomous graph: {}'.format(t3 - t2))
    get_input_dependent_graph_structure(checkpoint, inputs=inputs, contexts=None)
    t4 = time.time()
    print('Time for input graphs: {}'.format(t4 - t3))
    
    #PLOT INPUT
    i_input = 1
    transform = partial(np.dot, b=checkpoint['V'])
    ssa = State_Space_Analysis(checkpoint, data, transform=transform)
    ssa = plot_checkpoint_results(checkpoint, data, ssa=ssa,
                                  plot_cluster_means=False,
                                  eig_norm_color=False,
                                  plot_test_points=True,
                                  plot_fixed_points=False,
                                  plot_graph_structure=True,
                                  n_test_samples=None,
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
    
    data = task.gen_data(100, 10000)
    #big_data = task.gen_data(100, 500000)
    
    #Progress logging
    scratch_path = '/scratch/oem214/vanilla-rtrl/'
    log_path = os.path.join(scratch_path, 'log/' + analysis_job_name) + '_{}.log'.format(i_job)
    
    #Retrieve data
    with open(os.path.join('saved_runs', file_name), 'rb') as f:
        sim = pickle.load(f)
    
    for i_checkpoint in range(params['i_start'], params['i_start'] + 100, 10):

        with open(log_path, 'a') as f:
            f.write('Analyzing chekpoint {}\n'.format(i_checkpoint))
        
        checkpoint = sim.checkpoints[i_checkpoint]
        #checkpoint_flip = deepcopy(checkpoint)
        
        #analyze context 0
        analyze_checkpoint(checkpoint, data, verbose=False,
                           sigma_pert=0.5, N=500, parallelize=False,
                           N_iters=6000, same_LR_criterion=5000,
                           context=contexts[0])
        
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
    
    big_data = task.gen_data(100, 20000)
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
    
    #set_trace()
    
    #Initialize dissimilarity matrices
    n_checkpoints = len(checkpoints)
    # wasserstein_distances = np.zeros(n_checkpoints)
    # VAE_distances = np.zeros(n_checkpoints)
    PC1_distances = np.zeros((n_checkpoints, n_checkpoints))
    # PC2_distances = np.zeros(n_checkpoints)
    # SVCCA_distances = np.zeros(n_checkpoints)
    # graph_distances = np.zeros(n_checkpoints)
    # input_graph_distances = np.zeros(n_checkpoints)
    aligned_graph_distances = np.zeros((n_checkpoints, n_checkpoints))
    node_diff_distances = np.zeros((n_checkpoints, n_checkpoints))
    # rec_weight_distances = np.zeros(n_checkpoints)
    # output_weight_distances = np.zeros(n_checkpoints)
    # participation_coefficients = np.zeros(n_checkpoints)
    
    for i in range(len(indices)):
        
        #print(i)
        
        if i % 10 == 0 and os.environ['HOME'] == '/home/oem214':
            with open(log_path, 'a') as f:
                f.write('Calculating distance row {}\n'.format(i))
        elif i % 10 == 0:
            print(i)
        
        for j in range(i + 1, len(indices)):
        
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
            
            #wasserstein_distances[i] = wasserstein_distance(checkpoint_1,
            #                                                      checkpoint_2)
            
            # VAE_distances[i, i+j] = VAE_distance(checkpoint_1, checkpoint_2,
            #                                       big_data=big_data)
            
            PC1_distances[i, j] = PC_distance_1(checkpoint_1,
                                                  checkpoint_2)
            #PC2_distances[i] = PC_distance_2(checkpoint_1,
            #                                      checkpoint_2)
            # SVCCA_distances[i, i+j] = SVCCA_distance(checkpoint_1,
            #                                           checkpoint_2,
            #                                           data=data, R=2)
            # graph_distances[i, i+j] = graph_distance(checkpoint_1,
            #                                           checkpoint_2)
            # input_graph_distances[i, i+j] = input_dependent_graph_distance(checkpoint_1,
            #                                                                 checkpoint_2)
            aligned_graph_distances[i, j] = aligned_graph_distance(checkpoint_1,
                                                                checkpoint_2,
                                                                node_diff_penalty=0)
            node_diff_distances[i, j] = node_diff_distance(checkpoint_1, checkpoint_2)
            #rec_weight_distances[i] = rec_weight_distance(checkpoint_1,
            #                                                    checkpoint_2)
            #output_weight_distances[i] = output_weight_distance(checkpoint_1,
            #                                                    checkpoint_2)
            #participation_coefficients[i] = checkpoint_1['participation_coef']
        
    #result['wasserstein_distances'] = wasserstein_distances
    #result['VAE_distances'] = VAE_distances
    result['PC1_distances'] = PC1_distances + PC1_distances.T
    #result['PC2_distances'] = PC2_distances
    #result['SVCCA_distances'] = SVCCA_distances
    #result['graph_distances'] = graph_distances
    #result['input_graph_distances'] = input_graph_distances
    result['aligned_graph_distances'] = aligned_graph_distances + aligned_graph_distances.T
    result['node_diff_distances'] = node_diff_distances + node_diff_distances.T
    #result['rec_weight_distances'] = rec_weight_distances
    #result['output_weight_distances'] = output_weight_distances
    #result['participation_coefficients'] = participation_coefficients

""" ---------------------"""
""" --- PLOT RESULTS --- """
""" ---------------------"""

if MODE == 'PLOT':
    
    local_results = '/Users/omarschall/cluster_results/vanilla-rtrl/'
    figs_path = '/Users/omarschall/weekly-reports/report_09-30-2020/figs'
    data_path = os.path.join(local_results, compare_job_name, 'result_0')
    
    signals = {}
    try:
        with open(data_path, 'rb') as f:
            result = pickle.load(f)
    
        fig1 = plt.figure(figsize=(10,10))
        leg = []
        
        #norm_ = result['rec_weight_distances'][:-1]
        #norm_max = np.amax(norm_)
        #norm_min = np.amin(norm_)
        #norm_ = np.exp((norm_ - norm_min) / (norm_max - norm_min))
    
        for i_key, key in enumerate(result.keys()):
            
            if 'distances' in key or 'participation' in key:
                
                #x = np.array(result[key][:-1])
                
                
                # if key == 'participation_coefficients':
                    
                #     x = np.array(result[key])[:-4]
                    
                # else:
                x = np.diag(result[key][:-1,1:])
                

                
                x_max = np.amax(x)
                x_min = np.amin(x)
                x = (x - x_min) / (x_max - x_min)
                signals[key] = x.copy()
                
                if False: #normalize by grad norm if you want
                    x = x /  norm_
                plt.plot(x - 1.2 * i_key)
                leg.append(key.split('_')[0])
                
                #print(key, x_min, x_max)
                
        plt.legend(leg)
        plt.yticks([])
    except FileNotFoundError:
        pass
    
    #fig1.savefig(os.path.join(figs_path, 'fig22.pdf'), dpi=300, format='pdf')
    data_path = os.path.join(local_results, analysis_job_name)
    indices, checkpoints = unpack_analysis_results(data_path)
    #checkpoints = {k:result[k] for k in result.keys() if 'checkpoint' in k}
    
    ### come up with graph metric that uses the alignment
    # do output gradient norms spike after topological changes
    # cross correlograms of different statistics
    # correct by magnitude of error signal
    # come up with statistical summary tests that actually test hypothesises
    # might need more interesting task if we can only test rudimentary hypohessis with it
    # do different algorithms have diff
    # just before a phase transition, i have an expectated gradient direction and variance
    # how often do you actually get to this state transition? is it mean or flunctuations
    # maybe phase transitions are actually smooth, geometry isn't acutally affected that much
    # for a gradient of the same size, do we get bigger fluctuations in network dynamics than i would expec
    # during a phse transition?
    # how do we make this conversation scale-invariant, both in terms of gradient norm
    # and norm of network dynamics geomoetrical changes
    
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
    node_distances = np.array(node_distances)
    #grad_norms = np.array(grad_norms)
    
    
    data = task.gen_data(100, 10000)
    # sparse_inputs_task = Flip_Flop_Task(task.n_bit, 0.001)
    
    i_checkpoint = 9980
    checkpoint = checkpoints['checkpoint_{}'.format(i_checkpoint)]
    transform = Vanilla_PCA(checkpoint, data)
    ssa_2 = State_Space_Analysis(checkpoint, data, transform=transform)
    ssa_2 = plot_checkpoint_results(checkpoint, data, ssa=ssa_2,
                                    plot_cluster_means=True,
                                    eig_norm_color=False,
                                    plot_test_points=True,
                                    plot_fixed_points=False,
                                    plot_graph_structure=True,
                                    n_test_samples=None,
                                    graph_key='adjmat_input_3')
    
        
    # fig2 = plot_output_from_checkpoint(checkpoint, data, n_PCs=3)

    plot_input_dependent_topology(checkpoint, i_input=None)

    # indices = list(range(10000, 50000, 1000)) + [80000, 200000, 500000, 999000]
    # indices = list(range(0, 10000, 10))
    
    # for i_checkpoint in range(9000, 9100, 10):
    #     checkpoint = checkpoints['checkpoint_{}'.format(i_checkpoint)]
    #     ref_checkpoint = checkpoints['checkpoint_{}'.format(i_checkpoint - 100)]
    #     #plot_input_dependent_topology(checkpoint, i_input=None)
    #     fig, ax = plt.subplots(2, 2)
    #     ax[0, 0].imshow(ref_checkpoint['adjmat_input_0'])
    #     ax[0, 1].imshow(checkpoint['adjmat_input_0'])
    #     ax[1, 0].imshow(ref_checkpoint['forwardshared_adjmat_input_0'])
    #     ax[1, 1].imshow(checkpoint['backshared_adjmat_input_0'])

    # for i_checkpoint in indices:
    #     checkpoint = checkpoints['checkpoint_{}'.format(i_checkpoint)]
    #     fig = plot_input_dependent_topology(checkpoint, i_input=None, return_fig=True)
    #     fig.savefig(os.path.join(figs_path, 'Fig_{}.pdf'.format(i_checkpoint)), dpi=300, format='pdf')
        
        
    
    # with open(os.path.join(figs_path, 'tex_tex.txt'), 'w') as f:
    #     for i_index in range(0, len(indices), 4):
    #         i_checkpoints = [indices[i_index],
    #                           indices[i_index + 1],
    #                           indices[i_index + 2],
    #                           indices[i_index + 3]]
    #         f.write('\\begin{figure}[h]\n' +
    #                 '\\center\includegraphics[width=2.8cm]{{figs/Fig_{}.pdf}}'.format(i_checkpoints[0]) + 
    #                 '\\includegraphics[width=2.8cm]{{figs/Fig_{}.pdf}}'.format(i_checkpoints[1]) + 
    #                 '\\includegraphics[width=2.8cm]{{figs/Fig_{}.pdf}}'.format(i_checkpoints[2]) + 
    #                 '\\includegraphics[width=2.8cm]{{figs/Fig_{}.pdf}}\n'.format(i_checkpoints[3]) + 
    #                 '\\end{figure}\n')
        

    # meeting notes
    # kep track of node diff penalty as separate metric
    # do cross corr properly
    # comparison of learning rates
    # vary magnitude of inputs
        
    # only really one solution to flip flop task
    # problem with two different solutions? will different learning algs
    # find diff solutions? which stages of learning matter most if you switch
    # algs?
    # compute hessian exactly to assess local minima
    
        # assess robustness to slightly different task configs
    # study REINFORCE or simlar things even as it fails
    # what are computational limitations of different bio plausible learning algorithms
    # if stochastic gradients not  learning, what do metrics or kinetic energy
    # or whatever reveal about their shortcomings
    # can topological space show what makes one algorithm better or worse than
    # another?
    # at asymptoptic performance, do gradient directions matter for preserving
    # topology or is it just about their norms being small?
    # can otherwise identical topologies be compared w.r.t. robustness to perturbations?
    # perturbations both to weight confirugartions and also input magnitudes.
    # lesioning neurons?
    # shallow minima, find basins of attraction that are robust. Are topologoical
    # strucutres signifiers of robustness?

    #figs_path = '/Users/omarschall/weekly-reports/report_08-19-2020/figs'
    #ssa_2.fig.savefig(os.path.join(figs_path, 'fig16.pdf'), dpi=300, format='pdf')

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












