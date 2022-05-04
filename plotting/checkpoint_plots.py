import sys, os
sys.path.append(os.path.abspath('..'))
import numpy as np
import matplotlib.pyplot as plt
from core import Simulation
from dynamics import *
from plotting.State_Space_Analysis import State_Space_Analysis

def plot_output_from_checkpoint(checkpoint, data, plot_title=None,
                                figsize=(3, 2), xlim=500,
                                time_steps_per_trial=None,
                                trial_mask=None,
                                reset_sigma=None,
                                y_spacing=2,
                                **kwargs):
    """For a given checkpoint in a simulation and data dict, runs a fresh
    test simulation and plots the results in output spcae."""

    rnn = checkpoint['rnn']
    test_sim = Simulation(rnn,
                          time_steps_per_trial=time_steps_per_trial,
                          trial_mask=trial_mask,
                          reset_sigma=reset_sigma)
    test_sim.run(data, mode='test', monitors=['rnn.loss_', 'rnn.y_hat'],
                 verbose=False, **kwargs)

    fig = plt.figure(figsize=figsize)
    for i in range(rnn.n_out):

        plt.plot(data['test']['X'][:, i] - i * y_spacing, (str(0.6)))
        plt.plot(data['test']['Y'][:, i] - i * y_spacing, 'C0')
        plt.plot(test_sim.mons['rnn.y_hat'][:, i] - i * y_spacing, 'C3')
    if time_steps_per_trial is not None:
        for i in range(0, data['test']['X'].shape[0], time_steps_per_trial):
            plt.axvline(x=i, color='k', linestyle='--')
    plt.xlim([0, xlim])
    plt.yticks([])
    plt.xlabel('time steps')
    if plot_title is not None:
        plt.title(plot_title)

    return fig

def plot_input_dependent_topology(checkpoint, data,
                                  reference_checkpoint=None,
                                  graph_key='adjmat',
                                  i_input=None, plotting_noise=0.05,
                                  return_fig=False, n_inputs=None,
                                  color_scheme='dotted',
                                  plot_thresh=0.05,
                                  colors_=None,
                                  space_range=1,
                                  circle_color=('0.8'),
                                  circle_radius=0.18,
                                  output_ylim=[-9, 3],
                                  figsize=(8, 4)):

    if n_inputs is None:
        n_in = 6
    else:
        n_in = n_inputs

    n_half = n_in // 2

    if i_input is not None:
        keys = ['{}_input_{}'.format(graph_key, i_input)]
    else:
        keys = ['{}_input_{}'.format(graph_key, i) for i in range(n_in)]

    n_nodes = checkpoint[keys[0]].shape[0]

    if reference_checkpoint is not None:
        total_nodes = max(reference_checkpoint['nodes'].shape[0], n_nodes)
    else:
        total_nodes = n_nodes

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Checkpoint {}'.format(checkpoint['i_t']))

    t_range = np.arange(np.pi, -np.pi, -2 * np.pi / total_nodes)
    circle_nodes = np.array([[np.cos(t), np.sin(t)] for t in t_range[:n_nodes]])

    #plot little circles
    # theta = np.arange(0, 2 * np.pi, 0.01)
    # for i_cn, node in enumerate(circle_nodes):
    #     plt.plot(node[0] + circle_radius * np.cos(theta),
    #              node[1] + circle_radius * np.sin(theta),
    #              color=('0.8'))
    #     plt.fill_between(node[0] + circle_radius * np.cos(theta)[2:-2],
    #                      node[1] + circle_radius * np.sin(theta)[2:-2],
    #                      color=('0.8'))
    for node in circle_nodes:
        circ = plt.Circle(node, radius=circle_radius, color=circle_color)
        ax[0].add_artist(circ)

    #Create multigraph tensor
    graph = np.stack([checkpoint[key] for key in keys], axis=-1)
    leg = []
    colors = []
    linestyles = []
    for key in keys:
        i_input_ = int(key.split('_')[-1])
        if color_scheme == 'dotted':
            i_color = i_input_ % n_half
            linestyle = ['-', '--'][i_input_ // n_half]
        if color_scheme == 'different':
            i_color = i_input_
            linestyle = '-'
        linestyles.append(linestyle)
        if colors_ is None:
            colors.append('C{}'.format(i_color))
        else:
            colors.append(colors_[i_color])

        leg_ = 'Input {}'.format(i_color)
        if linestyle == '-':
            leg_ += ', +'
        else:
            leg_ += ', -'
        leg.append(leg_)

    for i in range(graph.shape[0]):
        for j in range(i + 1, graph.shape[1]):

            ij_connects = np.where(graph[i, j] > plot_thresh)[0]
            ji_connects = np.where(graph[j, i] > plot_thresh)[0]
            n_connects = len(ij_connects) + len(ji_connects)

            node_i = circle_nodes[i]
            node_j = circle_nodes[j]
            diff = node_j - node_i
            u_diff = diff / norm(diff)
            np.random.seed(0)
            w = np.random.normal(0, 1, 2)
            diff_perp = w - u_diff.dot(w) * u_diff
            diff_perp = circle_radius * diff_perp / norm(diff_perp)

            spacings = np.linspace(-space_range, space_range, n_connects)
            start_points = [node_i + s * diff_perp for s in spacings]
            end_points = [node_j + s * diff_perp for s in spacings]

            for i_flip, connects in enumerate([ij_connects, ji_connects]):
                for i_connect, i_input in enumerate(connects):

                    i_shift = i_flip * len(ij_connects)
                    to_circle = np.sqrt(1 - spacings[i_connect + i_shift] ** 2) * circle_radius * u_diff
                    if i_flip == 0:
                        start = start_points[i_connect + i_shift] + to_circle
                        end = end_points[i_connect + i_shift] - to_circle
                    else:
                        start = end_points[i_connect + i_shift] - to_circle
                        end = start_points[i_connect + i_shift] + to_circle

                    col = colors[i_input]
                    linestyle = linestyles[i_input]

                    ax[0].plot([start[0], end[0]],
                            [start[1], end[1]], color=col, linestyle=linestyle)

                    ax[0].plot([start[0]], [start[1]], 'x', color=col)

                    ax[0].plot([end[0]], [end[1]], '.', color=col)

    ax[0].set_xlim([-1.4, 1.4])
    ax[0].set_ylim([-1.4, 1.4])
    ax[0].axis('off')

    #Add output plots
    rnn = checkpoint['rnn']
    test_sim = Simulation(rnn)
    test_sim.run(data, mode='test', monitors=['rnn.y_hat'], verbose=False)

    for i in range(rnn.n_in):
        ax[1].plot(data['test']['X'][40:, i] - i * 2.5, color=('0.7'), linestyle='--')
        ax[1].plot(data['test']['Y'][40:, i] - i * 2.5, color=('0.2'))
        ax[1].plot(test_sim.mons['rnn.y_hat'][40:, i] - i * 2.5, color=colors[i])

    ax[1].axis('off')
    ax[1].set_ylim(output_ylim)
    if return_fig:
        return fig

def plot_input_dependent_task_topology(task, reference_checkpoint=None,
                                       i_input=None, plotting_noise=0.05,
                                       return_fig=False, color_scheme='dotted'):

    n_nodes = task.n_states
    total_nodes = n_nodes

    fig = plt.figure(figsize=(4, 4))
    plt.title('Task structure')

    t_range = np.arange(np.pi, -np.pi, -2 * np.pi / total_nodes)
    circle_nodes = np.array([[np.cos(t), np.sin(t)] for t in t_range[:n_nodes]])

    #plot little circles
    theta = np.arange(0, 2 * np.pi, 0.01)
    for node in circle_nodes:
        plt.plot(node[0] + 0.18 * np.cos(theta), node[1] + 0.18 * np.sin(theta),
                 color=('0.6'))

    n_in = len(task.T_dict.keys())
    n_half = n_in // 2

    if i_input is not None:
        keys = ['input_{}'.format(i_input)]
    else:
        keys = ['input_{}'.format(i) for i in range(n_in)]
    np.random.seed(0)
    leg = []
    for key in keys:

        graph = task.T_dict[key]
        i_input_ = int(key.split('_')[1])

        if color_scheme == 'dotted':
            i_color = i_input_ % 3
            linestyle = ['-', '--'][i_input_ // n_half]
        if color_scheme == 'different':
            i_color = i_input_
            linestyle = '-'

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

            # plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
            #           color='C{}'.format(i_color), alpha=weight,
            #           head_width=0.025, head_length=0.05,
            #           linestyle=linestyle)

    plt.xlim([-1.4, 1.4])
    plt.ylim([-1.4, 1.4])
    #plt.legend(leg)
    #plt.axis('equal')
    #set_trace()

    if return_fig:
        return fig

def plot_projection_of_rec_weights(checkpoint_lists, return_fig=False):


    n_params = checkpoint_lists[0][0]['rnn'].n_h_params

    fig = plt.figure()
    U = np.linalg.qr(np.random.normal(0, 1, (n_params, n_params)))[0][:2]

    for i_list, checkpoints_list in enumerate(checkpoint_lists):
        rec_params = []
        for checkpoint in checkpoints_list:
            rnn = checkpoint['rnn']
            params = np.concatenate([rnn.W_rec.flatten(),
                                     rnn.W_in.flatten(),
                                     rnn.b_rec])
            rec_params.append(params)
        rec_params = np.array(rec_params)



        proj = rec_params.dot(U.T)


        col = 'C{}'.format(i_list)
        plt.plot(proj[:,0], proj[:,1], color=col)
        plt.plot([proj[0,0]], [proj[0,1]], 'x', color=col)
        plt.plot([proj[-1,0]], [proj[-1,1]], '.', color=col)

    if return_fig:
        return fig

def plot_checkpoint_results(checkpoint, data, ssa=None, plot_test_points=False,
                            plot_fixed_points=False, plot_cluster_means=False,
                            plot_uncategorized_points=False,
                            plot_init_points=False, eig_norm_color=False,
                            plot_graph_structure=False,
                            n_vae_samples=None,
                            n_test_samples=None,
                            graph_key='adjacency_matrix'):
    """For a fresh or already given State_Space_Analysis object, plots many
    relevant data from an analyzed checkpoint in the 3D State Space."""

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
            color = 'C{}'.format(i + 1)
        if plot_fixed_points:
            ssa.plot_in_state_space(fixed_points[labels == i], False, color, '*', alpha=0.5)

    if plot_cluster_means:
        if eig_norm_color:
            ssa.plot_in_state_space(cluster_means[cluster_eigs < 1], False, 'k', 'X', alpha=0.3)
            ssa.plot_in_state_space(cluster_means[cluster_eigs > 1], False, 'k', 'o', alpha=0.3)
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