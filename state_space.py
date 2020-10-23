#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 17:18:46 2019
@author: omarschall
"""

import numpy as np
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    pass
from utils import *
from dynamics import *
from mpl_toolkits.mplot3d import Axes3D


class State_Space_Analysis:

    def __init__(self, checkpoint, test_data, dim_reduction_method=Vanilla_PCA,
                 transform=None, **kwargs):
        """The array trajectories must have a shape of (sample, unit)"""

        if transform is None:
            self.transform = dim_reduction_method(checkpoint, test_data,
                                                  **kwargs)
        else:
            self.transform = transform

        dummy_data = np.zeros((10, checkpoint['rnn'].n_h))
        self.dim = self.transform(dummy_data).shape[1]

        self.fig = plt.figure()
        if self.dim == 2:
            self.ax = self.fig.add_subplot(111)
        if self.dim == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')

    def plot_in_state_space(self, trajectories, mark_start_and_end=False,
                            color='C0', *args, **kwargs):
        """Plots given trajectories' projection onto axes as defined in
        __init__ by training data."""

        projs = self.transform(trajectories)

        if self.dim == 2:
            self.ax.plot(projs[:, 0], projs[:, 1], *args, **kwargs,
                         color=color)
            if mark_start_and_end:
                self.ax.plot([projs[0, 0]], [projs[0, 1]], 'x', color=color)
                self.ax.plot([projs[-1, 0]], [projs[-1, 1]], 'o', color=color)
        if self.dim == 3:
            self.ax.plot(projs[:, 0], projs[:, 1], projs[:, 2],
                         *args, **kwargs, color=color)
            if mark_start_and_end:
                self.ax.plot([projs[0, 0]], [projs[0, 1]], [projs[0, 2]],
                             'x', color=color)
                self.ax.plot([projs[-1, 0]], [projs[-1, 1]], [projs[-1, 2]],
                             'o', color=color)

    def clear_plot(self):
        """Clears all plots from figure"""

        self.fig.axes[0].clear()

def plot_checkpoint_results(checkpoint, data, ssa=None, plot_test_points=False,
                            plot_fixed_points=False, plot_cluster_means=False,
                            plot_uncategorized_points=False,
                            plot_init_points=False, eig_norm_color=False,
                            plot_graph_structure=False,
                            n_vae_samples=None,
                            n_test_samples=None,
                            graph_key='adjacency_matrix'):

    rnn = checkpoint['rnn']
    test_sim = Simulation(rnn)
    test_sim.run(data,
                  mode='test',
                  monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
                  verbose=False)

    A_init = checkpoint['A_init']
    fixed_points = checkpoint['fixed_points']
    labels = checkpoint['cluster_labels']
    cluster_means = checkpoint['cluster_means']
    cluster_eigs = checkpoint['cluster_eigs']
    
    M = checkpoint['adjacency_matrix']
    nodes = cluster_means[np.where(M.sum(0) > 0)]

    if ssa is None:
        transform = partial(np.dot, b=checkpoint['V'])
        ssa = State_Space_Analysis(checkpoint, data, transform=transform)
    ssa.clear_plot()
    if plot_test_points:
        ssa.plot_in_state_space(test_sim.mons['rnn.a'][1000:], False, 'C0',
                                '.', alpha=0.009)
    if n_test_samples is not None:
        n = int(np.ceil(np.sqrt(n_test_samples)))
        fig, ax = plt.subplots(n, n)
        for i_ax in range(n_test_samples):
            n1 = i_ax // n
            n2 = i_ax % n
            try:
                T = checkpoint['VAE_T']
            except KeyError:
                T = 10
            T_total = test_sim.mons['rnn.a'].shape[0]
            t_start = np.random.randint(0, T_total - T)
            ssa.plot_in_state_space(test_sim.mons['rnn.a'][t_start:t_start + T],
                                    True, 'C0', alpha=0.7)
            for i_out in range(test_sim.rnn.n_out):
                ax[n1, n2].plot(data['test']['X'][t_start:t_start + T, i_out] + 2.5 * i_out,
                                (str(0.6)), linestyle='--')
                ax[n1, n2].plot(data['test']['Y'][t_start:t_start + T, i_out] + 2.5 * i_out, 'C0')
                ax[n1, n2].plot(test_sim.mons['rnn.y_hat'][t_start:t_start + T, i_out] + 2.5 * i_out, 'C2')
                # ax[n1, n2].plot(data['test']['X'][t_start:t_start + T, i_out],
                #                 (str(0.6)), linestyle='--')
                # ax[n1, n2].plot(data['test']['Y'][t_start:t_start + T, 1], 'C0')
                # ax[n1, n2].plot(test_sim.mons['rnn.y_hat'][t_start:t_start + T, 1], 'C2')
                # ax[n1, n2].plot(data['test']['X'][t_start:t_start + T, 2] - 2.5,
                #                 (str(0.6)), linestyle='--')
                # ax[n1, n2].plot(data['test']['Y'][t_start:t_start + T, 2] - 2.5, 'C0')
                # ax[n1, n2].plot(test_sim.mons['rnn.y_hat'][t_start:t_start + T, 2] - 2.5, 'C2')
            
            ax[n1, n2].set_yticks([])

    if plot_init_points:
        ssa.plot_in_state_space(A_init, False, 'C9', 'x', alpha=1)

    cluster_idx = np.unique(labels)
    n_clusters = len(cluster_idx) - (-1 in cluster_idx)
    for i in cluster_idx:

        if i == -1:
            color = 'k'
            if not plot_uncategorized_points:
                continue
        else:
            color = 'C{}'.format(i+1)
        if plot_fixed_points:
            ssa.plot_in_state_space(fixed_points[labels == i], False, color, '*', alpha=0.5)

    if plot_cluster_means:
        if eig_norm_color:
            ssa.plot_in_state_space(cluster_means[cluster_eigs<1], False, 'k', 'X', alpha=0.3)
            ssa.plot_in_state_space(cluster_means[cluster_eigs>1], False, 'k', 'o', alpha=0.3)
        else:
            ssa.plot_in_state_space(cluster_means, False, 'k', 'X', alpha=0.3)

    if plot_graph_structure:

        graph = checkpoint[graph_key]
        
        if graph_key == 'adjacency_matrix':
            nodes = cluster_means
        
        for i, j in zip(*np.where(graph != 0)):

            if i == j:
                continue

            weight = graph[i, j]
            line = np.array([nodes[i], nodes[j]])
            ssa.plot_in_state_space(line, True, color='k', alpha=weight)

    if n_vae_samples is not None:

        for _ in range(n_vae_samples):
            traj = sample_from_VAE(checkpoint)
            ssa.plot_in_state_space(traj, True, 'C3', alpha=0.7)

    return ssa

def plot_output_from_checkpoint(checkpoint, data, n_PCs=3):
    
    rnn = checkpoint['rnn']
    test_sim = Simulation(rnn)
    test_sim.run(data,
                  mode='test',
                  monitors=['rnn.loss_', 'rnn.y_hat'],
                  verbose=False)

    fig = plt.figure()
    plt.plot(data['test']['X'][:, 0] + 3.5, (str(0.6)), linestyle='--')
    plt.plot(data['test']['Y'][:, 0] + 3.5, 'C0')
    plt.plot(test_sim.mons['rnn.y_hat'][:, 0] + 3.5, 'C3')
    plt.plot(data['test']['X'][:, 1], (str(0.6)), linestyle='--')
    plt.plot(data['test']['Y'][:, 1], 'C0')
    plt.plot(test_sim.mons['rnn.y_hat'][:, 1], 'C3')
    if n_PCs == 3:
        plt.plot(data['test']['X'][:, 2] - 3.5, (str(0.6)), linestyle='--')
        plt.plot(data['test']['Y'][:, 2] - 3.5, 'C0')
        plt.plot(test_sim.mons['rnn.y_hat'][:, 2] - 3.5, 'C3')
    plt.xlim([0, 500])
    plt.yticks([])
    plt.xlabel('time steps')
    
    return fig

def plot_input_dependent_topology(checkpoint, reference_checkpoint=None,
                                  i_input=None, plotting_noise=0.03,
                                  return_fig=False):
    
    n_nodes = checkpoint['nodes'].shape[0]
    
    if reference_checkpoint is not None:
        total_nodes = max(reference_checkpoint['nodes'].shape[0], n_nodes)
    else:
        total_nodes = n_nodes
    
    fig = plt.figure(figsize=(4, 4))
    plt.title('Checkpoint {}'.format(checkpoint['i_t']))
    
    t_range = np.arange(np.pi, -np.pi, -2 * np.pi / total_nodes)
    circle_nodes = np.array([[np.cos(t), np.sin(t)] for t in t_range[:n_nodes]])
    
    #plot little circles
    theta = np.arange(0, 2 * np.pi, 0.01)
    for node in circle_nodes:
        plt.plot(node[0] + 0.15 * np.cos(theta), node[1] + 0.15 * np.sin(theta),
                 color=('0.6'))

    
    #keys = [k for k in checkpoint.keys() if 'adjmat' in k]
    
    if i_input is not None:
        keys = ['adjmat_input_{}'.format(i_input)]
    else:
        keys = ['adjmat_input_{}'.format(i) for i in range(6)]
    np.random.seed(0)
    leg = []
    for key in keys:
        
        graph = checkpoint[key]
        i_input_ = int(key.split('_')[2])
        i_color = i_input_ % 3
        linestyle = ['-', '--'][i_input_ // 3]
        leg_ = 'Input {}'.format(i_color)
        if linestyle == '-':
            leg_ += ', +'
        else:
            leg_ += ', -'
        leg.append(leg_)
        
        for i, j in zip(*np.where(graph != 0)):
    
            if i == j:
                continue
    
            weight = graph[i, j]
            
            if weight < 0.05:
                continue
            
            line = np.array([circle_nodes[i], circle_nodes[j]])
            
            start = circle_nodes[i] + np.random.normal(0, plotting_noise, 2)
            end = circle_nodes[j] + np.random.normal(0, plotting_noise, 2)
            
            plt.plot([start[0], end[0]],
                     [start[1], end[1]],
                     color='C{}'.format(i_color), alpha=weight,
                     linestyle=linestyle)
            
            plt.plot([start[0]],
                     [start[1]], 'x',
                     color='C{}'.format(i_color), alpha=weight)
            
            plt.plot([end[0]],
                     [end[1]], '.',
                     color='C{}'.format(i_color), alpha=weight)
        
    plt.xlim([-1.4, 1.4])
    plt.ylim([-1.4, 1.4])
    #plt.legend(leg)
    #plt.axis('equal')
    #set_trace()
    
    if return_fig:
        return fig
    
    
    
    
    
    
    
    
    
    
    
    
    
    