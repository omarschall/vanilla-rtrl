#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 13:48:33 2020

@author: omarschall
"""

import numpy as np
from simulation import *
from utils import *
import multiprocessing as mp
from functools import partial
from itertools import product
from pdb import set_trace
from network import *
from copy import copy, deepcopy
import time
import os
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
from sklearn.cross_decomposition import CCA
try:
    import umap
except ModuleNotFoundError:
    pass
from vae import Encoder, Decoder, VAE
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch

### --- WRAPPER METHODS --- ###

def get_test_sim_data(checkpoint, test_data, sigma=0):
    """Get hidden states from a test run for a given checkpoint"""

    rnn = deepcopy(checkpoint['rnn'])
    test_sim = Simulation(rnn)
    test_sim.run(test_data, mode='test',
                 monitors=['rnn.a'],
                 verbose=False,
                 a_initial=rnn.a.copy(),
                 sigma=sigma)

    return test_sim.mons['rnn.a']

def analyze_all_checkpoints(checkpoints, func, test_data, **kwargs):
    """For a given analysis function and a list of checkpoints, applies
    the function to each checkpoint in the list and returns all results.
    Uses multiprocessing to apply the analysis independently."""

    func_ = partial(func, test_data=test_data, **kwargs)
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(func_, checkpoints)

    return results

def analyze_checkpoint(checkpoint, data, N_iters=8000,
                       same_LR_criterion=5000, N=200,
                       n_PCs=3, context=None,
                       reference_checkpoint=None,
                       sigma=0,
                       **kwargs):

    print('Analyzing checkpoint {}...'.format(checkpoint['i_t']))

    rnn = checkpoint['rnn']
    test_sim = Simulation(rnn)
    test_sim.run(data,
                  mode='test',
                  monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
                  verbose=False,
                  sigma=sigma)

    transform = Vanilla_PCA(checkpoint, data, n_PCs=n_PCs, sigma=sigma)
    V = transform(np.eye(rnn.n_h))

    fixed_points, initial_states = find_KE_minima(checkpoint, data, N=N,
                                                  N_iters=N_iters, LR=10,
                                                  same_LR_criterion=same_LR_criterion,
                                                  context=context,
                                                  **kwargs)

    A = np.array([d['a_final'] for d in fixed_points])
    A_init = np.array(initial_states)
    KE = np.array([d['KE_final'] for d in fixed_points])

    dbscan = DBSCAN(eps=0.5)
    dbscan.fit(A)
    dbscan.labels_

    cluster_idx = np.unique(dbscan.labels_)
    n_clusters = len(cluster_idx) - (-1 in cluster_idx)
    cluster_means = np.zeros((n_clusters, rnn.n_h))
    for i in cluster_idx:

        if i == -1:
            continue
        else:
            cluster_means[i] = A[dbscan.labels_ == i].mean(0)

    cluster_eigs = []
    cluster_KEs = []
    for cluster_mean in cluster_means:
        checkpoint['rnn'].reset_network(a=cluster_mean)
        a_J = checkpoint['rnn'].get_a_jacobian(update=False)
        cluster_eigs.append(np.abs(np.linalg.eig(a_J)[0][0]))
        KE = checkpoint['rnn'].get_network_speed()
        cluster_KEs.append(KE)
    cluster_eigs = np.array(cluster_eigs)
    cluster_KEs = np.array(cluster_KEs)

    #Save results
    checkpoint['fixed_points'] = A
    checkpoint['KE'] = KE
    checkpoint['cluster_means'] = cluster_means
    checkpoint['cluster_labels'] = dbscan.labels_
    checkpoint['V'] = V
    checkpoint['A_init'] = A_init
    checkpoint['cluster_eigs'] = cluster_eigs
    checkpoint['cluster_KEs'] = cluster_KEs
    checkpoint['test_loss'] = test_sim.mons['rnn.loss_'].mean()
    
    if reference_checkpoint is not None:
        align_checkpoints(reference_checkpoint, checkpoint)

### --- ANALYSIS METHODS --- ###

def Vanilla_PCA(checkpoint, test_data, n_PCs=3, sigma=0):
    """Return first n_PCs PC axes of the test """

    test_a = get_test_sim_data(checkpoint, test_data, sigma=sigma)
    U, S, V = np.linalg.svd(test_a)

    checkpoint['participation_coef'] = np.square(S.sum()) / np.square(S).sum()

    transform = partial(np.dot, b=V[:,:n_PCs])

    return transform

def UMAP_(checkpoint, test_data, n_components=3, **kwargs):
    """Performs  UMAP with default parameters and returns component axes."""

    test_a = get_test_sim_data(checkpoint, test_data)
    fit = umap.UMAP(n_components=n_components, **kwargs)
    u = fit.fit_transform(test_a)

    return fit.transform

def find_KE_minima(checkpoint, test_data, N=1000, verbose_=False,
                   parallelize=False, sigma_pert=0, PCs=None, weak_input=None,
                   context=None, **kwargs):
    """Find many KE minima for a given checkpoint of training. Includes option
    to parallelize or not."""

    test_a = get_test_sim_data(checkpoint, test_data)
    results = []
    initial_states = []

    RNNs = []
    for i in range(N):

        #Set up initial conditions
        rnn = deepcopy(checkpoint['rnn'])
        if context is not None:
            rnn.b_rec += rnn.W_in.dot(context)
        i_a = np.random.randint(test_a.shape[0])
        u_pert = np.random.normal(0, sigma_pert, rnn.n_h)
        if PCs is not None:
            PC_pert = np.random.binomial(0, 0.2, PCs.shape[1]) * sigma_pert
            u_pert = PCs.dot(PC_pert)
        a_init = test_a[i_a] + u_pert
        rnn.reset_network(a=a_init)
        if weak_input is not None:
            x = np.random.binomial(0, 0.2, rnn.n_in) * weak_input
            rnn.next_state(x)
        RNNs.append(rnn)
        initial_states.append(rnn.a.copy())

    if parallelize:

        func_ = partial(find_KE_minimum, **kwargs)
        with mp.Pool(mp.cpu_count()) as pool:
                results = pool.map(func_, RNNs)

    if not parallelize:

        for i in range(N):

            #Report progress
            if i % (N // 10) == 0 and verbose_:
                print('{}% done'.format(i * 10 / N))

            #Select RNN (and starting point)
            rnn = RNNs[i]

            #Calculate final result
            result = find_KE_minimum(rnn, **kwargs)
            results.append(result)

    return results, initial_states

def find_KE_minimum(rnn, LR=1e-2, N_iters=1000000,
                    return_whole_optimization=False,
                    return_period=100,
                    N_KE_increase=3,
                    LR_drop_factor=5,
                    LR_drop_criterion=10,
                    same_LR_criterion=100000,
                    verbose=False,
                    calculate_linearization=False):
    """For a given RNN, performs gradient descent with adaptive learning rate
    to find a kinetic energy  minimum of the network. The seed state is just
    the state of the rnn, rnn.a.


    Returns either ust the final rnn state and KE, or if
    return_whole_optimization is True, then it also returns trajectory of a
    values, norms, and LR_drop times at a frequency specified by
    return_period."""

    #Initialize counters
    i_LR_drop = 0
    i_KE_increase = 0
    i_same_LR = 0

    #Initialize return lists
    a_values = []
    KEs = [rnn.get_network_speed()]
    norms = [norm(rnn.a)]
    LR_drop_times = []

    #Loop once for each iteration
    for i_iter in range(N_iters):

        #Report progress
        if i_iter % (N_iters//10) == 0:
            pct_complete = np.round(i_iter/N_iters*100, 2)
            if verbose:
                print('{}% done'.format(pct_complete))

        rnn.a -= LR * rnn.get_network_speed_gradient()
        a_values.append(rnn.a.copy())

        KEs.append(rnn.get_network_speed())
        norms.append(norm(rnn.a))

        #Stop optimization if KE increases too many steps in a row
        if KEs[i_iter] > KEs[i_iter - 1]:
            i_KE_increase += 1
            if i_KE_increase >= N_KE_increase:
                LR /= LR_drop_factor
                if verbose:
                    print('LR drop #{} at iter {}'.format(i_LR_drop, i_iter))
                LR_drop_times.append(i_iter)
                i_LR_drop += 1
                i_KE_increase = 0
                if i_LR_drop >= LR_drop_criterion:
                    if verbose:
                        print('Reached criterion at {} iter'.format(i_iter))
                    break
        else:
            i_KE_increase = 0
            i_same_LR += 1

        if i_same_LR >= same_LR_criterion:
            print('Reached same LR criterion at {} iter'.format(i_iter))
            break


    results = {'a_final': a_values[-1],
               'KE_final': KEs[-1]}

    if calculate_linearization:

        rnn.a = results['a_final']
        a_J = rnn.get_a_jacobian(update=False)
        eigs, _ = np.linalg.eig(a_J)
        results['jacobian_eigs'] = eigs

    if return_whole_optimization:
        results['a_trajectory'] = np.array(a_values[::return_period])
        results['norms'] = np.array(norms[::return_period])
        results['KEs'] = np.array(KEs[::return_period])
        results['LR_drop_times'] = LR_drop_times

    return results

def run_autonomous_sim(a_initial, rnn, N, monitors=[],
                       return_final_state=False, input_pulse=None,
                       background_input=0, sigma=0):
    """Creates and runs a test simulation with no inputs and a specified
    initial state of the network."""

    #Create empty data array
    data = {'test': {'X': np.zeros((N, rnn.n_in)) + background_input,
                     'Y': np.zeros((N, rnn.n_out))}}
    
    rnn.reset_network(a=a_initial)
    
    if input_pulse is not None:
        x = input_pulse + background_input
        rnn.next_state(x=x)
    
    sim = Simulation(rnn)
    sim.run(data, mode='test', monitors=monitors,
            a_initial=rnn.a,
            verbose=False,
            sigma=sigma)

    if return_final_state:
        return sim.rnn.a.copy()
    else:
        return sim

def get_graph_structure(checkpoint, N=100, time_steps=50, epsilon=0.01,
                        parallelize=True, key='adjacency_matrix',
                        input_pulse=None, background_input=0, nodes=None,
                        sigma=0):
    """For each fixed point cluster, runs an autonomous simulation with
    initial condition in small small neighborhood of a point and evaluates
    where it ends up."""

    if nodes is None:
        cluster_means = checkpoint['cluster_means']
    else:
        cluster_means = nodes
    n_clusters = cluster_means.shape[0]
    adjacency_matrix = np.zeros((n_clusters, n_clusters))
    rnn = checkpoint['rnn']


    if parallelize:
        for i in range(n_clusters):
            a_init = [(cluster_means[i] +
                      np.random.normal(0, epsilon, rnn.n_h))
                      for _ in range(N)]
            func_ = partial(run_autonomous_sim, rnn=rnn, N=time_steps,
                            monitors=[], return_final_state=True,
                            input_pulse=input_pulse,
                            background_input=background_input,
                            sigma=sigma)
            #set_trace()
            with mp.Pool(mp.cpu_count()) as pool:
                final_states = pool.map(func_, a_init)

            final_states = np.array(final_states)

            distances = distance.cdist(cluster_means, final_states)
            i_clusters = np.argmin(distances, axis=0)
            bins = list(np.arange(-0.5, n_clusters, 1))
            transition_probs, _ = np.histogram(i_clusters,
                                               bins=bins,
                                               density=True)
            #set_trace()
            adjacency_matrix[i] = transition_probs

    if not parallelize:
        for i in range(n_clusters):
            a_init = [(cluster_means[i] +
                      np.random.normal(0, epsilon, rnn.n_h))
                      for _ in range(N)]
            func_ = partial(run_autonomous_sim, rnn=rnn, N=time_steps,
                            monitors=[], return_final_state=True,
                            input_pulse=input_pulse,
                            background_input=background_input,
                            sigma=sigma)
            #set_trace()
            final_states = []
            for a_init_ in a_init:
                final_states.append(func_(a_init_))

            final_states = np.array(final_states)

            distances = distance.cdist(cluster_means, final_states)
            i_clusters = np.argmin(distances, axis=0)
            bins = list(np.arange(-0.5, n_clusters, 1))
            transition_probs, _ = np.histogram(i_clusters,
                                               bins=bins,
                                               density=True)
            #set_trace()
            adjacency_matrix[i] = transition_probs

    checkpoint[key] = adjacency_matrix
    
def get_input_dependent_graph_structure(checkpoint, inputs, contexts=None,
                                        parallelize=False,
                                        nodes_matrix=None,
                                        node_thresh=0.05,
                                        sigma=0):
    """After running get_graph_structure, this can be used to find input-
    dependent transition probabilities between stable nodes.
    
    Args:
        inputs: A list of input vectors to be sampled iid
    """
    
    M = checkpoint['adjacency_matrix']
    centroids = checkpoint['cluster_means']
    nodes = centroids[np.where(M.sum(0) > node_thresh)]
    checkpoint['nodes'] = nodes
    
    if contexts is not None:        
        conds = product(inputs, contexts)
    else:
        conds = inputs
        
    for i_x, x in enumerate(conds):
        
        if contexts is None:
            key = 'adjmat_input_{}'.format(i_x)
            
            adjacency_matrix = get_graph_structure(checkpoint,
                                                   parallelize=parallelize,
                                                   key=key,
                                                   input_pulse=x,
                                                   nodes=nodes,
                                                   sigma=sigma)
        else:
            key = 'adjmat_input_{}_context_{}'.format(i_x // len(inputs),
                                                      i_x % len(inputs))
            
            adjacency_matrix = get_graph_structure(checkpoint,
                                                   parallelize=parallelize,
                                                   key=key,
                                                   input_pulse=x[0],
                                                   background_input=x[1],
                                                   nodes=nodes,
                                                   sigma=sigma)


def SVCCA_distance(checkpoint_1, checkpoint_2, data, R=3):
    """Compute the singular-value canonical correlation analysis distance
    between two different networks."""

    A_1 = get_test_sim_data(checkpoint_1, data)
    A_2 = get_test_sim_data(checkpoint_1, data)

    cca = CCA(n_components=R)
    cca.fit(A_1, A_2)

    return cca.score(A_1, A_2)

def train_VAE(checkpoint, data, T=20, n_epochs=5, lr=0.01,
              latent_dim=10, encoder_dims=[256, 256],
              decoder_dims=[256]):

    #Generate data
    A = get_test_sim_data(checkpoint, data)
    train_data = torch.tensor(A.reshape((-1, T * checkpoint['rnn'].n_h)))

    input_dim = train_data.shape[1]
    #hidden_dim = 128

    #set_trace()

    #Define objects
    encoder = Encoder(input_dim, encoder_dims, latent_dim)
    decoder = Decoder(latent_dim, decoder_dims, input_dim)
    model = VAE(encoder, decoder)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #set_trace()

    # set the train mode
    model.train()

    losses = []
    recon_losses = []
    kl_losses = []

    for i_epoch in range(n_epochs):

        # loss of the epoch
        train_loss = 0
        print('starting epoch {}'.format(i_epoch))
        train_iterator = DataLoader(TensorDataset(train_data),
                                    batch_size=20,
                                    shuffle=True)

        #set_trace()

        for i, x in enumerate(train_iterator):
            # reshape the data into [batch_size, 784]
            x = x[0].view(-1, input_dim).type(torch.FloatTensor)

            #set_trace()

            # update the gradients to zero
            optimizer.zero_grad()

            # forward pass
            x_sample, z_mu, z_var = model(x)

            # reconstruction loss
            recon_loss = F.mse_loss(x_sample, x, size_average=False)

            # kl divergence loss
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) +
                                      z_mu ** 2 - 1.0 - z_var)

            # total loss
            loss = recon_loss + kl_loss

            recon_losses.append(recon_loss.detach().numpy())
            kl_losses.append(kl_loss.detach().numpy())
            losses.append(loss.detach().numpy())

            # backward pass
            loss.backward()
            train_loss += loss.item()

            # update the weights
            optimizer.step()

    checkpoint['VAE'] = model
    checkpoint['VAE_T'] = T
    checkpoint['VAE_train_losses'] = losses
    checkpoint['VAE_kl_losses'] = kl_losses
    checkpoint['VAE_recon_losses'] = recon_losses

def sample_from_VAE(checkpoint):

    model = checkpoint['VAE']
    T = checkpoint['VAE_T']
    # sample and generate a image

    z = torch.randn(1, model.enc.mu.out_features)

    # run only the decoder
    reconstructed_traj = model.dec(z)
    traj = reconstructed_traj.view(T, -1).data

    return traj

def test_vae(model_checkpoint, data, test_checkpoint=None):
    
    model = model_checkpoint['VAE']
    n_h = model_checkpoint['rnn'].n_h
    T = model_checkpoint['VAE_T']
    
    if test_checkpoint is None:
        A = get_test_sim_data(model_checkpoint, data)
        
    else:
        A = get_test_sim_data(test_checkpoint, data)
        
    test_data = torch.tensor(A.reshape((-1, T * n_h)))
    
    test_iterator = DataLoader(TensorDataset(test_data),
                               batch_size=A.shape[0],
                               shuffle=False)
    input_dim = test_data.shape[1]
    # set the evaluation mode
    model.eval()

    # test loss for the data
    test_loss = 0
    
    with torch.no_grad():
        for i, x in enumerate(test_iterator):
            # reshape the data
            x = x[0].view(-1, input_dim).type(torch.FloatTensor)

            # forward pass
            x_sample, z_mu, z_var = model(x)

            # reconstruction loss
            recon_loss = F.mse_loss(x_sample, x, size_average=False)
            
            # kl divergence loss
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)
            
            # total loss
            loss = recon_loss + kl_loss
            test_loss += loss.item()

    return test_loss


def get_multitask_loss_from_checkpoints(sim, multitask, N_test):
    
    data = multitask.gen_data(0, N_test)
    
    #set_trace()
    
    losses = {'task_{}_loss'.format(i): [] for i in range(multitask.n_tasks)}
    
    for i in range(multitask.n_tasks):
        for j in sorted(sim.checkpoints.keys()):
            
            rnn = sim.checkpoints[j]['rnn']
            test_sim = Simulation(rnn)
            test_sim.run(data,
                         mode='test_{}'.format(i),
                         monitors=['rnn.loss_'],
                         verbose=False)
            
            test_loss = test_sim.mons['rnn.loss_'].mean()
            
            losses['task_{}_loss'.format(i)].append(test_loss)
            
        losses['task_{}_loss'.format(i)] = np.array(losses['task_{}_loss'.format(i)])
        
    return losses

def get_loss_from_checkpoints(sim, task, N_test):
    
    data = task.gen_data(0, N_test)
    
    #set_trace()
    
    losses = []
    
    for j in sorted(sim.checkpoints.keys()):
        
        rnn = sim.checkpoints[j]['rnn']
        test_sim = Simulation(rnn)
        test_sim.run(data,
                     mode='test',
                     monitors=['rnn.loss_'],
                     verbose=False)
        
        test_loss = test_sim.mons['rnn.loss_'].mean()
        
        losses.append(test_loss)
        
    losses = np.array(losses)
        
    return losses
        
        
        

