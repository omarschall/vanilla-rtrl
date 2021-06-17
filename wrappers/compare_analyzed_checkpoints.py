import os, pickle
from cluster import unpack_analysis_results
from dynamics import *

default_compare_args = {'wasserstein': False,
                        'VAE_': False,
                        'PC1': False,
                        'PC2': False,
                        'PC3': True,
                        'SVCCA': False,
                        'aligned_graph': True,
                        'node_diff': True,
                        'rec_weight': True}

def compare_analyzed_checkpoints(analysis_job_name,
                                 compare_args=default_compare_args,
                                 username='oem214',
                                 project_name='learning-dynamics'):
    """For a given analysis job name, takes the analyzed checkpoints and
    computes neighboring distances in a matrix along the off-diagonal for
    specificied distance functions."""

    ### --- Unpack distance flags --- ###


    wasserstein = compare_args['wasserstein']
    VAE_ = compare_args['VAE_']
    PC1 = compare_args['PC1']
    PC2 = compare_args['PC2']
    PC3 = compare_args['PC3']
    SVCCA = compare_args['SVCCA']
    aligned_graph = compare_args['aligned_graph']
    node_diff = compare_args['node_diff']
    rec_weight = compare_args['rec_weight']

    ### --- Get paths, extract and unpack data --- ###

    project_dir = os.path.join('scratch/{}/'.format(username), project_name)
    results_dir = os.path.join(project_dir, 'results/')
    analysis_dir = os.path.join(results_dir, analysis_job_name)

    saved_run_name = analysis_job_name.split('analyze_')[-1]
    saved_run_path = os.path.join(project_dir, 'notebooks', 'saved_runs',
                                  saved_run_name)
    with open(saved_run_path, 'rb') as f:
        saved_run = pickle.load(f)
    task = saved_run['task']

    compare_job_name = 'compare_{}'.format(saved_run_name)
    log_path = os.path.join(project_dir, 'logs/' + compare_job_name) + '.o.log'

    # Unpack data
    indices, checkpoints = unpack_analysis_results(analysis_dir)

    if VAE_:
        big_data = task.gen_data(100, 20000)
    if SVCCA:
        data = task.gen_data(100, 10000)

    ### --- Initialize dissimilarity matrices --- ###

    n_checkpoints = len(indices)

    if wasserstein:
        wasserstein_distances = np.zeros((n_checkpoints, n_checkpoints))
    if VAE_:
        VAE_distances = np.zeros((n_checkpoints, n_checkpoints))
    if PC1:
        PC1_distances = np.zeros((n_checkpoints, n_checkpoints))
    if PC2:
        PC2_distances = np.zeros((n_checkpoints, n_checkpoints))
    if PC3:
        PC3_distances = np.zeros((n_checkpoints, n_checkpoints))
    if SVCCA:
        SVCCA_distances = np.zeros((n_checkpoints, n_checkpoints))
    if aligned_graph:
        aligned_graph_distances = np.zeros((n_checkpoints, n_checkpoints))
    if node_diff:
        node_diff_distances = np.zeros((n_checkpoints, n_checkpoints))
    if rec_weight:
        rec_weight_distances = np.zeros((n_checkpoints, n_checkpoints))

    for i in range(len(indices) - 1):

        if i % 10 == 0:
            with open(log_path, 'a') as f:
                f.write('Calculating distance row {}\n'.format(i))

        for j in range(i + 1, i + 2):

            i_index = indices[i]
            j_index = indices[j]

            try:
                checkpoint_1 = checkpoints['checkpoint_{}'.format(i_index)]
                checkpoint_2 = checkpoints['checkpoint_{}'.format(j_index)]
            except KeyError:
                continue

            try:
                align_checkpoints(checkpoint_2, checkpoint_1, n_inputs=4)
                align_checkpoints(checkpoint_2, checkpoint_1, n_inputs=4)
            except ValueError:
                break

            if wasserstein:
                wasserstein_distances[i, j] = wasserstein_distance(checkpoint_1,
                                                                   checkpoint_2)

            if VAE_:
                VAE_distances[i, j] = VAE_distance(checkpoint_1, checkpoint_2,
                                                   big_data=big_data)

            if PC1:
                PC1_distances[i, j] = PC_distance_1(checkpoint_1,
                                                    checkpoint_2)
            if PC2:
                PC2_distances[i, j] = PC_distance_2(checkpoint_1,
                                                    checkpoint_2)
            if PC3:
                PC3_distances[i, j] = PC_distance_3(checkpoint_1,
                                                    checkpoint_2,
                                                    N_avg=25, N_test=1000,
                                                    task=task)
            if SVCCA:
                SVCCA_distances[i, j] = SVCCA_distance(checkpoint_1,
                                                       checkpoint_2,
                                                       data=data, R=3)
            if aligned_graph:
                aligned_graph_distances[i, j] = aligned_graph_distance(checkpoint_1,
                                                                       checkpoint_2,
                                                                       node_diff_penalty=0,
                                                                       n_inputs=4)
            if node_diff:
                node_diff_distances[i, j] = node_diff_distance(checkpoint_1,
                                                               checkpoint_2)
            if rec_weight:
                rec_weight_distances[i, j] = rec_weight_distance(checkpoint_1,
                                                                 checkpoint_2)

    result = {}

    if wasserstein:
        result['wasserstein_distances'] = wasserstein_distances
    if VAE_:
        result['VAE_distances'] = VAE_distances
    if PC1:
        result['PC1_distances'] = PC1_distances
    if PC2:
        result['PC2_distances'] = PC2_distances
    if PC3:
        result['PC3_distances'] = PC3_distances
    if SVCCA:
        result['SVCCA_distances'] = SVCCA_distances
    if aligned_graph:
        result['aligned_graph_distances'] = aligned_graph_distances
    if node_diff:
        result['node_diff_distances'] = node_diff_distances
    if rec_weight:
        result['rec_weight_distances'] = rec_weight_distances

    i_job = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
    result['i_job'] = i_job
    save_dir = os.environ['SAVEDIR']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'result_{}'.format(i_job))

    with open(save_path, 'wb') as f:
        pickle.dump(result, f)