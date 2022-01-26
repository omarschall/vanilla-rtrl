from math import ceil
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import decimate
from scipy.ndimage.filters import uniform_filter1d
from sklearn.manifold import MDS, TSNE
from utils import get_param_values_from_list_of_config_strings

def plot_eigenvalues(*matrices, fig=None, return_fig=False):
    """Plots eigenvalues of a given matrix in the complex plane, as well
    as the unit circle for reference."""

    if fig is None:
        fig = plt.figure()
    theta = np.arange(0, 2*np.pi, 0.01)
    plt.plot(np.cos(theta), np.sin(theta), 'k', linestyle='--', linewidth=1.5)
    plt.axis('equal')

    for M in matrices:
        eigs, _ = np.linalg.eig(M)
        plt.plot(np.real(eigs), np.imag(eigs), '.', markersize=10)

    if return_fig:
        return fig

def plot_array_of_histograms(counts, weights, ticks=None, return_fig=True,
                             plot_zero_line=True, fig_size=(6,3), **kwargs):
    """Plots count data in the shape (n_samples, n_row, n_col) as an array
    of histograms with n_row rows and n_col cols."""

    fig, ax = plt.subplots(counts.shape[1], counts.shape[2],
                           figsize=fig_size)

    n_bins = 100
    if 'n_bins' in kwargs.keys():
        n_bins = kwargs['n_bins']

    weighted_medians = np.zeros((counts.shape[1], counts.shape[2]))

    for i in range(counts.shape[1]):
        for j in range(counts.shape[2]):
            if i <= j:
                fig.delaxes(ax[i, j])
                continue
            w = weights[:, i, j]
            hist, bins = np.histogram(counts[:, i, j],
                                      weights=w/w.sum(),
                                      bins=np.linspace(-1, 1, n_bins),
                                      density=True)
            bin_centers = bins[:-1] + (bins[1] - bins[0])/2
            #weighted_medians[i, j] = weighted_median(hist, bin_centers)
            weighted_medians[i, j] = np.mean(counts[:, i, j])
            hist = uniform_filter1d(hist, 10)
            #ax[i, j].hist(counts[:, i, j], bins=np.linspace(-1, 1, n_bins))
            ax[i, j].plot(bin_centers, hist, linewidth=0.7)
            ax[i, j].fill_between(bin_centers,
                                  np.zeros_like(bin_centers),
                                  hist,
                                  color='C0',
                                  alpha=0.3)
            ax[i, j].axvline(x=weighted_medians[i, j], color='C0',
                             linestyle='--', dashes=[1.5,0.75], linewidth=0.7)
            #ax[i, j].set_ylim([0, 2])
            if plot_zero_line:
                ax[i, j].axvline(x=0, color='k', linestyle='--',
                                 dashes=[1.5,0.75], linewidth=0.7)
            ax[i, j].set_yticks([])
            if ticks is not None:
                if i == counts.shape[1] - 1:
                    pass
                    ax[i, j].set_xticks([])
                    #ax[i, j].set_xlabel(ticks[j])
                else:
                    ax[i, j].set_xticks([])
                if j == 0:
                    pass
                    #ax[i, j].set_ylabel(ticks[i])

    if return_fig:
        return (fig, weighted_medians)

def plot_array_of_downsampled_signals(signals, ticks=None, return_fig=False,
                                      plot_zero_line=True, **kwargs):
    """Plots time series data in the shape (T, n_row, n_col) as an array
    of filtered signals with n_row rows and n_col cols."""

    fig, ax = plt.subplots(signals.shape[1], signals.shape[2],
                           figsize=(30, 10))

    n_bins = 100
    if 'n_bins' in kwargs.keys():
        n_bins = kwargs['n_bins']

    for i in range(signals.shape[1]):
        for j in range(signals.shape[2]):
            if i < j:
                fig.delaxes(ax[i, j])
                continue
            signal = decimate(decimate(decimate(signals[:, i, j], 10), 10), 10)
            #time_ = np.arange(0, signals.shape[0], 1000)
            #i_time_ticks = [len(time_)//5*i for i in range(5)]
            #time_ticks = time_[np.array(i_time_ticks)]
            ax[i, j].plot(signal)
            ax[i, j].set_ylim([-1, 1])
            if plot_zero_line:
                ax[i, j].axhline(y=0, color='k', linestyle='--')
            if ticks is not None:
                if i == signals.shape[1] - 1:
                    ax[i, j].set_xlabel(ticks[j])
                    #labels = ['{}k'.format(int(time_tick/1000)) for time_tick in time_ticks]
                    #ax[i, j].set_xticks(time_ticks, labels)
                    #set_trace()
                else:
                    ax[i, j].set_xticks([])
                if j == 0:
                    ax[i, j].set_ylabel(ticks[i])
                    ax[i, j].set_yticks([-1, 0, 1])
                else:
                    ax[i, j].set_yticks([])

    if return_fig:
        return fig

def plot_signal_xcorr(s1, s2, return_fig=False, finite_diff=True):
    """IN PROGRESS -- need principled method for computing xcorr for non-
    stationary signals."""

    assert s1.shape == s2.shape

    #finite_difference

    if finite_diff:
        s1 = (s1[1:] - s1[:-1])# / (np.abs(s1[1:]) + np.abs(s1[:-1]) + 1)
        s2 = (s2[1:] - s2[:-1])# / (np.abs(s2[1:]) + np.abs(s2[:-1]) + 1)

    n_pts = s1.shape[0]
    lags = np.arange(1 - n_pts, n_pts)
    #s1 = (s1 - s1.mean()) / np.std(s1)
    #s2 = (s2 - s2.mean()) / np.std(s2)

    xcorr = np.correlate(s1, s2, 'full') / max(len(s1), len(s2))

    fig = plt.figure()
    plt.plot(lags, xcorr)
    plt.axvline(x=0, color='k', linestyle='--')

    if return_fig:
        return fig

def plot_2d_MDS_from_distance_matrix(distances, point_classes, return_fig=False,
                                     alpha=1, markersize=10):
    """For a given set of pariwise distances, plots the elements in a 2-D space
    that maximally preserves specified distances."""

    mds = MDS(n_components=2, dissimilarity='precomputed')
    proj = mds.fit_transform(distances)

    fig = plt.figure()

    for i_class in range(len(np.unique(point_classes))):

        col = 'C{}'.format(i_class)
        class_idx = np.where(point_classes == i_class)[0]
        start_idx = np.amin(class_idx)
        stop_idx = np.amax(class_idx)
        plt.plot(proj[class_idx, 0], proj[class_idx, 1], color=col, alpha=alpha)
        plt.plot([proj[start_idx, 0]], [proj[start_idx, 1]], 'x', color=col,
                 markersize=markersize)
        plt.plot([proj[stop_idx, 0]], [proj[stop_idx, 1]], '.', color=col,
                 markersize=markersize)

    if return_fig:
        return fig

def plot_3d_MDS_from_distance_matrix(distances, point_classes, return_fig=False,
                                     alpha=1, markersize=10, colors=None):
    """For a given set of pariwise distances, plots the elements in a 3-D space
    that maximally preserves specified distances."""

    mds = MDS(n_components=3, dissimilarity='precomputed')
    proj = mds.fit_transform(distances)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if colors is not None:
        unique_colors = list(set(colors))
        color_avgs = {uc: [] for uc in unique_colors}

    for i_class in range(len(np.unique(point_classes))):

        if colors is None:
            col = 'C{}'.format(i_class)
        else:
            col = colors[i_class]

        class_idx = np.where(point_classes == i_class)[0]
        start_idx = np.amin(class_idx)
        stop_idx = np.amax(class_idx)
        ax.plot(proj[class_idx, 0],
                proj[class_idx, 1],
                proj[class_idx, 2], color=col, alpha=alpha)
        ax.plot([proj[start_idx, 0]],
                [proj[start_idx, 1]],
                [proj[start_idx, 2]], 'x', color=col, markersize=markersize)
        ax.plot([proj[stop_idx, 0]],
                [proj[stop_idx, 1]],
                [proj[stop_idx, 2]], '.', color=col, markersize=markersize)

        color_avgs[col].append(proj[class_idx])

    #Plot average within color
    for key in color_avgs.keys():

        avg = sum(color_avgs[key]) / len(color_avgs[key])
        ax.plot(avg[:, 0], avg[:, 1], avg[:, 2], color=key)

    if return_fig:
        return fig

def plot_3d_tSNE_from_distance_matrix(distances, point_classes, return_fig=False,
                                      alpha=1, markersize=10, colors=None,
                                      **tsne_args):
    """For a given set of pariwise distances, plots the elements in a 3-D space
    that maximally preserves specified distances."""

    tsne = TSNE(n_components=3, metric='precomputed', **tsne_args)
    proj = tsne.fit_transform(distances)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if colors is not None:
        unique_colors = list(set(colors))
        color_avgs = {uc: [] for uc in unique_colors}

    for i_class in range(len(np.unique(point_classes))):

        if colors is None:
            col = 'C{}'.format(i_class)
        else:
            col = colors[i_class]

        class_idx = np.where(point_classes == i_class)[0]
        start_idx = np.amin(class_idx)
        stop_idx = np.amax(class_idx)
        ax.plot(proj[class_idx, 0],
                proj[class_idx, 1],
                proj[class_idx, 2], color=col, alpha=alpha)
        ax.plot([proj[start_idx, 0]],
                [proj[start_idx, 1]],
                [proj[start_idx, 2]], 'x', color=col, markersize=markersize)
        ax.plot([proj[stop_idx, 0]],
                [proj[stop_idx, 1]],
                [proj[stop_idx, 2]], '.', color=col, markersize=markersize)

        color_avgs[col].append(proj[class_idx])

    #Plot average within color
    for key in color_avgs.keys():

        avg = sum(color_avgs[key]) / len(color_avgs[key])
        ax.plot(avg[:, 0], avg[:, 1], avg[:, 2], color=key)

    if return_fig:
        return fig


def plot_signals(signals, key_restriction=None, title=None, x_values=None,
                 signal_clips={}, colors=None, legend=True,
                 stage_assignments=None):
    """For a dictionary of 1D time series signals, plots each vertically
    in a min-max range from 0 to 1.

    Includes option to clip any signal from above for easier visualization,
    via the signal_clips dictionary."""

    keys = signals.keys()
    if key_restriction is not None:
        keys = key_restriction

    if colors is None:
        colors = ['C{}'.format(i) for i in range(len(keys))]

    fig = plt.figure(figsize=(10, 2 * len(keys)))
    leg = []
    for i_key, key in enumerate(keys):

        y = signals[key].copy()

        if key in signal_clips.keys():
            y = np.clip(y, 0, signal_clips[key])

        y_max = np.amax(y)
        y_min = np.amin(y)

        if y_max != y_min:
            y = (y - y_min) / (y_max - y_min)
        else:
            y = y - y_min

        if x_values is not None:
            x = x_values[:len(y)]
        else:
            x = np.array(range(len(y)))

        plt.plot(x, y - 1.2 * i_key, color=colors[i_key])
        leg.append(key)

    if stage_assignments is not None:
        if x_values is not None:
            x = x_values[:len(stage_assignments)]
        else:
            x = np.array(range(len(stage_assignments)))
        ylim = -1.2 * (len(keys) -1)
        for i in range(4):
            #x_stage = x[stage_assignments == i + 1]
            where_ = stage_assignments == i + 1
            plt.fill_between(x=x, y1=1, y2=ylim, color='C{}'.format(i),
                             alpha=0.3, where=where_)

    if legend:
        plt.legend(leg)
    plt.yticks([])
    if title is not None:
        plt.title(title)

    return fig

def plot_multiple_signals(signal_dicts, key_restriction=None, title=None,
                          x_values=None, signal_clips={}, alpha=1):
    """For a list of dictionaries of 1D time series signals, plots each
    vertically in a min-max range from 0 to 1.

    Includes option to clip any signal from above for easier visualization,
    via the signal_clips dictionary."""

    fig = plt.figure(figsize=(10, 2 * len(signal_dicts[0].keys())))
    n_signals = len(signal_dicts)

    signals_for_avg = {k: [] for k in signal_dicts[0].keys()}
    for i_signals, signals in enumerate(signal_dicts):

        keys = signals.keys()
        if key_restriction is not None:
            keys = key_restriction

        leg = []
        for i_key, key in enumerate(keys):

            y = signals[key].copy()

            if key in signal_clips.keys():
                y = np.clip(y, 0, signal_clips[key])

            y_max = np.amax(y)
            y_min = np.amin(y)

            if y_max != y_min:
                y = (y - y_min) / (y_max - y_min)
            else:
                y = y - y_min

            if x_values is not None:
                x = x_values[:len(y)]
            else:
                x = list(range(len(y)))

            signals_for_avg[key].append(y)
            plt.plot(x, y - 1.2 * i_key, color='C{}'.format(i_key), alpha=alpha)
            leg.append(key)

        if i_signals == 0:
            plt.legend(leg)
            plt.yticks([])
            if title is not None:
                plt.title(title)

    #Plot averages
    for i_key, key in enumerate(signals_for_avg):
        y_avg = np.array(signals_for_avg[key]).mean(0)

        if x_values is not None:
            x = x_values[:len(y_avg)]
        else:
            x = list(range(len(y_avg)))
        plt.plot(x, y_avg - 1.2 * i_key, color='C{}'.format(i_key), alpha=1)

    return fig


def plot_2d_array_of_config_results(configs_array, results_array, key_order,
                                    log_scale=False, tick_rounding=3, **imshow_kwargs):
    """Given an array of configs (must be 2D) and corresponding results as
    floats, plots the result in a 2D grid averaging over random seeds."""

    fig = plt.figure()

    plt.imshow(results_array.mean(-1), **imshow_kwargs)

    if log_scale:
        plt.yticks(range(results_array.shape[0]),
                   np.round(np.log10(configs_array[key_order[0]]),
                            tick_rounding))
        plt.xticks(range(results_array.shape[1]),
                   np.round(np.log10(configs_array[key_order[1]]),
                            tick_rounding))
    else:
        plt.yticks(range(results_array.shape[0]),
                   np.round(configs_array[key_order[0]],
                            tick_rounding))
        plt.xticks(range(results_array.shape[1]),
                   np.round(configs_array[key_order[1]],
                            tick_rounding))

    plt.ylabel(key_order[0])
    plt.xlabel(key_order[1])
    plt.colorbar()

    return fig

def plot_3d_or_4d_array_of_config_results(configs_array, results_array, key_order,
                                          tick_rounding=3, **imshow_kwargs):
    """Given an array of configs (must be 3-4D) and corresponding results as
    floats, plots the result in a grid averaging over random seeds."""

    d_grid = len(results_array.shape) - 3

    if d_grid == 1:
        n_x = 1
        n_y = results_array.shape[2]
    elif d_grid == 2:
        n_x, n_y = results_array.shape[2:4]
    else:
        #from pdb import set_trace
        #set_trace()
        raise ValueError('Configs must be 3 or 4 dimensional')

    fig, axes = plt.subplots(n_x, n_y, figsize=(n_y * 5, n_x * 5))

    for i_x in range(n_x):
        for i_y in range(n_y):

            if d_grid == 1:
                results_slice = results_array[:,:,i_y,:].mean(-1)
                ax = axes[i_y]
            if d_grid == 2:
                results_slice = results_array[:,:,i_x,i_y,:].mean(-1)
                ax = axes[i_x, i_y]

            ax.imshow(results_slice, **imshow_kwargs)

            ax.set_yticks(list(range(len(configs_array[key_order[0]]))))
            ax.set_xticks(list(range(len(configs_array[key_order[1]]))))
            if type(configs_array[key_order[0]][0]) != str:
                ax.set_yticklabels(np.round(configs_array[key_order[0]],
                                            tick_rounding))
            else:
                ax.set_yticklabels(configs_array[key_order[0]])
            if type(configs_array[key_order[1]][0]) != str:
                ax.set_xticklabels(np.round(configs_array[key_order[1]],
                                            tick_rounding))
            else:
                ax.set_xticklabels(configs_array[key_order[1]])

            ax.set_ylabel(key_order[0])
            ax.set_xlabel(key_order[1])

            if d_grid == 1:
                y_param = key_order[2]
                title = y_param + '= {}'.format(configs_array[y_param][i_y])
            if d_grid == 2:
                x_param = key_order[2]
                y_param = key_order[3]
                title = '{} = {}, {} = {}'.format(x_param,
                                                  configs_array[x_param][i_x],
                                                  y_param,
                                                  configs_array[y_param][i_y])

            ax.set_title(title)

    #plt.colorbar()

    return fig

def plot_1d_or_2d_array_of_config_examples(configs_array, results_array,
                                           key_order, sim_dict, data,
                                           task_dict=None, N_task_data=None,
                                           xlim=500, trace_spacing=2):
    """Given an array of configs (must be 2D) and corresponding results as
    floats, plots the result in a 2D grid averaging over random seeds."""

    d_grid = len(results_array.shape) - 1

    if d_grid == 1:
        n_x = 1
        n_y = results_array.shape[0]
    elif d_grid == 2:
        n_x, n_y = results_array.shape[:2]
    else:
        raise ValueError('Configs must be 1 or 2 dimensional')

    fig, axes = plt.subplots(n_x, n_y, figsize=(n_y * 5, n_x * 5))

    for i_x in range(n_x):
        for i_y in range(n_y):

            for i_seed in range(len(configs_array['i_seed'])):

                if d_grid == 1:
                    ax = axes[i_y]
                    sim_dict_key = (str(configs_array[key_order[0]][i_y])
                                    + '_{}'.format(i_seed))
                if d_grid == 2:
                    ax = axes[i_x, i_y]
                    sim_dict_key = (str(configs_array[key_order[0]][i_x])
                                    + '_'
                                    + str(configs_array[key_order[1]][i_y])
                                    + '_{}'.format(i_seed))

                if task_dict is not None:
                    task = task_dict[sim_dict_key]
                    np.random.seed(0)
                    data = task.gen_data(0, N_task_data)

                sim = sim_dict[sim_dict_key]
                rnn = sim.rnn
                test_sim = sim.get_test_sim()
                test_sim.run(data, mode='test', monitors=['rnn.loss_', 'rnn.y_hat'],
                             verbose=False)

                for i in range(rnn.n_out):
                    ax.plot(data['test']['X'][:, i] - i * trace_spacing, (str(0.6)))
                    ax.plot(data['test']['Y'][:, i] - i * trace_spacing, 'C0')
                    ax.plot(test_sim.mons['rnn.y_hat'][:, i] - i * trace_spacing, 'C3', alpha=0.7)
                if sim.time_steps_per_trial is not None:
                    for i in range(0, data['test']['X'].shape[0], sim.time_steps_per_trial):
                        ax.axvline(x=i, color='k', linestyle='--')
                ax.set_xlim([0, xlim])
                ax.set_yticks([])
                ax.set_xlabel('time steps')

                if d_grid == 1:
                    y_param = key_order[0]
                    title = y_param + '= {}'.format(configs_array[y_param][i_y])
                if d_grid == 2:
                    x_param = key_order[0]
                    y_param = key_order[1]
                    title = '{} = {}, {} = {}'.format(x_param,
                                                      configs_array[x_param][i_x],
                                                      y_param,
                                                      configs_array[y_param][i_y])

                ax.set_title(title)

    return fig

def plot_kinetic_energy_histograms(indices, checkpoints, return_fig=False,
                                   red_line=-4):
    """For a list of ordered indices and corresponding dict of checkpoints,
    plots the histogram of log kinetic energy for each checkpoint in array."""

    all_KEs = []
    for i_checkpoint in indices:
        checkpoint = checkpoints['checkpoint_{}'.format(i_checkpoint)]
        all_KEs.append(np.log10(checkpoint['KE']))

    all_KEs = np.concatenate(all_KEs)

    bins = np.linspace(np.amin(all_KEs), np.amax(all_KEs), 30)

    n_rows = ceil(np.sqrt(len(indices)))

    fig, ax = plt.subplots(n_rows, n_rows, figsize=(2 * n_rows,
                                                    2 * n_rows))

    for i in range(n_rows ** 2):

        i_x = i // n_rows
        i_y = i % n_rows

        ax[i_x, i_y].set_xticks([])
        ax[i_x, i_y].set_yticks([])

        try:
            i_checkpoint = indices[i]
            checkpoint = checkpoints['checkpoint_{}'.format(i_checkpoint)]
            hist, edges = np.histogram(np.log10(checkpoint['KE']), bins=bins)
            bin_width = edges[1] - edges[0]

            ax[i_x, i_y].plot(edges[:-1] + bin_width / 2, hist)
            ax[i_x, i_y].axvline(x=red_line, color='C3', linestyle='--')
            ax[i_x, i_y].set_title(str(i_checkpoint))
        except IndexError:
            continue

    if return_fig:
        return fig

def plot_task_data(data, mode='train', time_points=100, curve_spacing=1.2):
    """Plot the X and Y values of a data matrix from a given task"""

    x_dim = data[mode]['X'].shape[1]
    y_dim = data[mode]['Y'].shape[1]

    for i_x in range(x_dim):
        X = data[mode]['X'][:time_points, i_x]
        plt.plot(X - curve_spacing * i_x, color=(str(0.6)), linestyle='--')

    for i_y in range(y_dim):
        Y = data[mode]['Y'][:time_points, i_y]
        plt.plot(Y - curve_spacing * (i_y + x_dim), color='C3')

def color_fader(color_1, color_2, mix=0):
    color_1 = np.array(mpl.colors.to_rgb(color_1))
    color_2 = np.array(mpl.colors.to_rgb(color_2))
    return mpl.colors.to_hex((1 - mix) * color_1 + mix * color_2)

def plot_array_of_signals(signal_dicts, root_name,
                          signal_keys=[], x_values=None, return_fig=False,
                          alpha=1, fig_width=3.4252, fig_length=4,
                          swap_order=False):

    param_values, key_order = get_param_values_from_list_of_config_strings(signal_dicts,
                                                                           root_name=root_name)

    value_keys = [k for k in key_order if k != 'seed']
    if swap_order:
        value_keys = value_keys[::-1]

    if len(value_keys) > 2:
        raise ValueError('Must be no more than 2 parameter variations')
    if len(value_keys) == 1:
        #Add dummy parameter
        param_values['dummy'] = [0]
        value_keys.append('dummy')

    n_x = len(param_values[value_keys[0]])
    n_y = len(param_values[value_keys[1]])

    fig, ax = plt.subplots(n_x, n_y, figsize=(fig_width, fig_length))

    for i_x in range(n_x):
        for i_y in range(n_y):
            for i_seed in param_values['seed']:

                file_name = 'analyze_' + root_name + '_seed={}'.format(i_seed)
                if not swap_order:
                    file_name += '_{}={}'.format(value_keys[0], str(param_values[value_keys[0]][i_x]).replace('.', ','))
                    file_name += '_{}={}'.format(value_keys[1], str(param_values[value_keys[1]][i_y]).replace('.', ','))
                if swap_order:
                    file_name += '_{}={}'.format(value_keys[1], str(param_values[value_keys[1]][i_y]).replace('.', ','))
                    file_name += '_{}={}'.format(value_keys[0], str(param_values[value_keys[0]][i_x]).replace('.', ','))

                for i_key, key in enumerate(signal_keys):

                    y = signal_dicts[file_name][key]

                    if x_values is not None:
                        x = x_values[:len(y)]
                    else:
                        x = list(range(len(y)))

                    ax[i_x, i_y].plot(x, y, color='C{}'.format(i_key), alpha=alpha)

            if i_x == 0:
                ax[i_x, i_y].set_title('{} = {}'.format(value_keys[1],
                                                        param_values[value_keys[1]][i_y]))
            if i_y == 0:
                ax[i_x, i_y].set_ylabel('{} = {}'.format(value_keys[0],
                                                         param_values[value_keys[0]][i_x]))

    if return_fig:
        return fig

def plot_time_spent_in_stages(list_of_stage_assignments, colors=None,
                              fig_width=3.4252, fig_length=4,
                              return_fig=False):
    """Plots histograms of stage assignment time points for different RNN
    training runs.

    Optional argument for colors, a list which must have same length as
    list_of_stage_assignments with a color for each run."""

    fig = plt.figure(figsize=(fig_width, fig_length))

    for i_sa, stage_assignments in enumerate(list_of_stage_assignments):
        counts, bins = np.histogram(stage_assignments,
                                    bins=[0.5, 1.5, 2.5, 3.5, 4.5])

        if colors is not None:
            col = colors[i_sa]
        else:
            col = 'C0'

        plt.plot(list(range(4)), counts, alpha=0.6, color=col)

    if return_fig:
        return fig