import os, pickle
import numpy as np
from copy import deepcopy

def unpack_analysis_results(data_path):
    """For a path to results, unpacks the data into a dict of checkpoints
    and sorted corresponding indices."""

    done = False
    checkpoints = {}
    i = 0
    i_missing = 0
    while not done:

        file_path = os.path.join(data_path, 'result_{}'.format(i))

        try:
            with open(file_path, 'rb') as f:
                result = pickle.load(f)
            checkpoints.update(result)
        except FileNotFoundError:
            i_missing += 1

        i += 1
        if i_missing > 5:
            done = True
    indices = sorted([int(k.split('_')[-1]) for k in
                      checkpoints.keys() if 'checkpoint' in k])

    indices = np.array(indices)

    return indices, checkpoints


def unpack_compare_result(saved_run_name, checkpoint_stats={},
                          project_name='learning-dynamics',
                          results_subdir='misc',
                          username='oem214'):
    """Unpack the results of a full analysis -> compare run. Returns
    a dict of 'signals', i.e. numpy arrays with shape (n_checkpoints).

    Args:
        saved_run_name (str): Original file name of analyzed simulation
        checkpoint_stats (dict): Dictionary whose entries are
            functions that take in a checkpoint and return some float."""

    analysis_job_name = 'analyze_{}'.format(saved_run_name)
    compare_job_name = 'compare_{}'.format(saved_run_name)

    project_dir = os.path.join('/scratch/', username, project_name)
    results_dir = os.path.join(project_dir, 'results', results_subdir)
    analysis_result_path = os.path.join(results_dir, analysis_job_name)
    compare_result_path = os.path.join(results_dir, compare_job_name)

    ### --- Unpack neighbor comparison results --- ###

    with open(os.path.join(compare_result_path, 'result_0'), 'rb') as f:
        result = pickle.load(f)

    signals = {}

    for key in result.keys():

        if 'distance' in key:
            x = np.diag(result[key][:-1, 1:])
            signals[key] = x.copy()

    ### --- Unpack individual checkpoint results --- ###

    indices, checkpoints = unpack_analysis_results(analysis_result_path)

    for key in checkpoint_stats.keys():

        stat_ = []

        for i_index, index in enumerate(indices):
            checkpoint = checkpoints['checkpoint_{}'.format(index)]
            stat_.append(checkpoint_stats[key](checkpoint))

        signals[key] = np.array(stat_)

    return signals

def unpack_cross_compare_result(saved_run_root_name, checkpoint_stats={},
                                relative_weight_change=True,
                                multi_job_comp=False,
                                project_name='learning-dynamics',
                                results_subdir='misc',
                                username='oem214'):
    """Unpack the results of a full analysis -> compare run. Returns
    a dict of 'signals', i.e. numpy arrays with shape (n_checkpoints).

    Args:
        saved_run_name (str): Original file name of analyzed simulation
        checkpoint_stats (dict): Dictionary whose entries are
            functions that take in a checkpoint and return some float."""


    ### --- Get paths, extract and unpack compare data --- ###

    project_dir = os.path.join('/scratch/', username, project_name)
    results_subdir_abs = os.path.join(project_dir, 'results', results_subdir)
    saved_runs_dir = 'saved_runs'

    compare_job_name = 'cross_compare_{}'.format(saved_run_root_name)
    compare_result_path = os.path.join(results_subdir_abs, compare_job_name)
    if multi_job_comp:
        result = unpack_sparse_cross_compare_results(saved_run_root_name,
                                                     project_name=project_name,
                                                     results_subdir=results_subdir,
                                                     username=username)
    else:
        with open(os.path.join(compare_result_path, 'result_0'), 'rb') as f:
            result = pickle.load(f)
    result['job_indices'] = np.array(result['job_indices'])

    all_indices = []
    checkpoints_lists = []
    job_indices = []

    signal_dicts = {}
    for i_job, analysis_job_name in enumerate(result['analysis_job_names']):
        analysis_dir = os.path.join(results_subdir_abs, analysis_job_name)

        # Unpack data
        indices, checkpoints = unpack_analysis_results(analysis_dir)
        all_indices = np.concatenate([all_indices, indices]).astype(np.int)
        checkpoints_lists.append(checkpoints)
        job_indices += [i_job] * len(indices)

        if relative_weight_change:
            W_init = checkpoints['checkpoint_0']['rnn'].W_rec
        else:
            W_init = None

        signals_ = {}
        for key in checkpoint_stats.keys():

            stats_ = []

            for i_index, index in enumerate(indices):
                checkpoint = checkpoints['checkpoint_{}'.format(index)]
                try:
                    stat_ = checkpoint_stats[key](checkpoint, W_init=W_init)
                except TypeError:
                    stat_ = checkpoint_stats[key](checkpoint)
                stats_.append(stat_)

            signals_[key] = np.array(stats_)

        for key in result.keys():

            if 'distance' in key:
                idx = np.where(result['job_indices'] == i_job)[0]
                sub_distance_mat = result[key][idx, :][:, idx]
                x = np.diag(sub_distance_mat[:-1, 1:])
                signals_[key] = x.copy()

        signal_dicts[analysis_job_name] = signals_

    return signal_dicts, result

def unpack_sparse_cross_compare_results(saved_run_root_name,
                                        project_name='learning-dynamics',
                                        results_subdir='misc',
                                        username='oem214'):
    """Unpacks the results of a cross comparison where discrete chunks are
    computed separately."""

    ### --- Get paths, extract and unpack compare data --- ###

    project_dir = os.path.join('/scratch/', username, project_name)
    results_subdir_abs = os.path.join(project_dir, 'results', results_subdir)
    saved_runs_dir = 'saved_runs'

    compare_job_name = 'cross_compare_{}'.format(saved_run_root_name)
    compare_result_path = os.path.join(results_subdir_abs, compare_job_name)

    args_path = os.path.join(project_dir, 'args', saved_run_root_name)

    combined_result_path = os.path.join(compare_result_path, 'result_combined')
    if os.path.exists(combined_result_path):
        with open(combined_result_path, 'rb') as f:
            result = pickle.load(f)
        return result
    else:
        with open(args_path, 'rb') as f:
            all_args = pickle.load(f)

        result = {}

        for i_comp_job in range(all_args['compare_n_comp_jobs']):
            job_path = os.path.join(compare_result_path, 'result_{}'.format(i_comp_job))
            with open(job_path, 'rb') as f:
                subresult = pickle.load(f)

            if i_comp_job == 0:
                for key in subresult.keys():
                    result[key] = deepcopy(subresult[key])
            else:
                for key in result.keys():
                    if 'distances' in key or 'check' in key:
                        result[key] += subresult[key]

        for key in result.keys():
            if 'distances' in key or 'check' in key:
                result[key] = np.array(result[key].todense())

        with open(combined_result_path, 'wb') as f:
            pickle.dump(result, f)

        return result