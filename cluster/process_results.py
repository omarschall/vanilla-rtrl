import os, pickle
import numpy as np

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

    return indices, checkpoints


def unpack_compare_result(saved_run_name, checkpoint_stats={}):
    """Unpack the results of a full analysis -> compare run. Returns
    a dict of 'signals', i.e. numpy arrays with shape (n_checkpoints).

    Args:
        saved_run_name (str): Original file name of analyzed simulation
        checkpoint_stats (dict): Dictionary whose entries are
            functions that take in a checkpoint and return some float."""

    analysis_job_name = 'analyze_{}'.format(saved_run_name)
    compare_job_name = 'compare_{}'.format(saved_run_name)

    results_dir = '/scratch/oem214/learning-dynamics/results/'

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

def unpack_cross_compare_result(saved_run_root_name, checkpoint_stats={}):
    """Unpack the results of a full analysis -> compare run. Returns
    a dict of 'signals', i.e. numpy arrays with shape (n_checkpoints).

    Args:
        saved_run_name (str): Original file name of analyzed simulation
        checkpoint_stats (dict): Dictionary whose entries are
            functions that take in a checkpoint and return some float."""

    analysis_job_name = 'analyze_{}'.format(saved_run_name)
    compare_job_name = 'compare_{}'.format(saved_run_name)

    results_dir = '/scratch/oem214/learning-dynamics/results/'

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