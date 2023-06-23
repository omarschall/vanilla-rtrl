import os, pickle
from cluster import unpack_analysis_results
from dynamics import *
from scipy import sparse
from math import ceil

def cross_compare_analyzed_checkpoints(saved_run_root_name,
                                       compare_args,
                                       username='om2382',
                                       notebook_dir=None,
                                       project_name='learning-dynamics'):
    """For a set of analysis results by root name, takes the analyzed
    checkpoints and computes neighboring distances in a matrix."""

    ### --- Unpack distance flags --- ###

    wasserstein = compare_args['wasserstein']
    VAE_ = compare_args['VAE_']
    PC1 = compare_args['PC1']
    PC2 = compare_args['PC2']
    PC3 = compare_args['PC3']
    SVCCA = compare_args['SVCCA']
    CKA = compare_args['CKA']
    aligned_graph = compare_args['aligned_graph']
    node_diff = compare_args['node_diff']
    node_drift = compare_args['node_drift']
    rec_weight = compare_args['rec_weight']
    output_weight = compare_args['output_weight']
    weight_change_alignment = compare_args['weight_change_alignment']

    ### --- Get paths, extract and unpack data --- ###

    project_dir = os.path.join('/home/{}/'.format(username), project_name)
    results_dir = os.path.split(os.environ['SAVEDIR'])[0]
    if notebook_dir is None:
        saved_runs_dir = os.path.join(project_dir, 'notebooks', 'saved_runs')
    else:
        saved_runs_dir = os.path.join(notebook_dir, 'saved_runs')


    ### --- Loop through each individual analysis job --- ###

    analysis_job_names = ['analyze_' + sr for sr in os.listdir(saved_runs_dir)
                          if saved_run_root_name in sr]
    if compare_args['cross_param_ordering'] is None:
        analysis_job_names = sorted(analysis_job_names)
    else:
        hashes = [''] * len(analysis_job_names)
        for key in compare_args['cross_param_ordering']:
            for i_ajn, ajn in enumerate(analysis_job_names):
                value = ajn.split(key+'=')[-1].split('_')[0]
                hashes[i_ajn] += value
        analysis_job_names = [analysis_job_names[i] for i in np.argsort(hashes)]

    all_indices = []
    checkpoints_lists = []
    job_indices = []
    for i_job, analysis_job_name in enumerate(analysis_job_names):
        analysis_dir = os.path.join(results_dir, analysis_job_name)

        saved_run_name = analysis_job_name.split('analyze_')[-1]
        saved_run_path = os.path.join(saved_runs_dir, saved_run_name)
        with open(saved_run_path, 'rb') as f:
            saved_run = pickle.load(f)
        task = saved_run['task']

        compare_job_name = 'compare_{}'.format(saved_run_name)
        log_path = os.path.join(project_dir, 'logs/' + compare_job_name) + '.log'

        # Unpack data
        indices, checkpoints = unpack_analysis_results(analysis_dir)
        all_indices = np.concatenate([all_indices, indices]).astype(np.int)
        checkpoints_lists.append(checkpoints)
        job_indices += [i_job] * len(indices)

    if VAE_:
        big_data = task.gen_data(1000, 20000)
    if SVCCA:
        data = task.gen_data(1000, 10000)

    ### --- Initialize dissimilarity matrices --- ###

    n_checkpoints = len(all_indices)

    calculation_check = np.zeros((n_checkpoints, n_checkpoints))

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
    if CKA:
        CKA_distances = np.zeros((n_checkpoints, n_checkpoints))
    if aligned_graph:
        aligned_graph_distances = np.zeros((n_checkpoints, n_checkpoints))
    if node_diff:
        node_diff_distances = np.zeros((n_checkpoints, n_checkpoints))
    if node_drift:
        node_drift_distances = np.zeros((n_checkpoints, n_checkpoints))
    if rec_weight:
        rec_weight_distances = np.zeros((n_checkpoints, n_checkpoints))
    if output_weight:
        output_weight_distances = np.zeros((n_checkpoints, n_checkpoints))
    if weight_change_alignment:
        weight_change_alignment_distances = np.zeros((n_checkpoints, n_checkpoints))

    #Compare window
    if compare_args['n_comp_window'] == 'full':
        n_comp_window = len(all_indices) - 1
    else:
        n_comp_window = compare_args['n_comp_window']

    #Job distribution arithmetic
    comp_overflow = n_comp_window * (n_comp_window + 1) / 2
    total_comps = n_comp_window * n_checkpoints - comp_overflow
    comps_per_job = ceil(total_comps / compare_args['n_comp_jobs'])
    leftover_comps = total_comps % comps_per_job

    i_comp_job = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
    comp_start = comps_per_job * i_comp_job
    if i_comp_job == compare_args['n_comp_jobs'] - 1 and leftover_comps > 0:
        comp_end = comp_start + leftover_comps
        comp_end = int(comp_end)
    else:
        comp_end = comps_per_job * (i_comp_job + 1)
    comp_range = list(range(comp_start, comp_end))

    # #Determine splitting point
    N = len(all_indices)
    i_split = N - n_comp_window
    i_range = []
    for i_ep, idx_endpoint in enumerate([comp_start, comp_end - 1]):
        if idx_endpoint <= (i_split * n_comp_window):
            i_endpoint = idx_endpoint // n_comp_window
            if i_ep == 0:
                j_start = int(idx_endpoint % n_comp_window)
        else:
            spillover = idx_endpoint - i_split * n_comp_window
            n_triangle = N - i_split - 1
            triangular_indices = n_triangle * (n_triangle + 1) / 2
            k = triangular_indices - spillover
            n, r = triangular_integer_decomposition(k)
            i_spillover = n_triangle - n - int(r > 0)
            i_endpoint = i_split + i_spillover
            if i_ep == 0:
                j_start = int(n_comp_window - 1 - i_spillover - r) * int(r > 0)
        i_range.append(i_endpoint)

    idx_flat = comp_start
    hit_range_yet = False

    for i in range(i_range[0], i_range[1] + 1):

        if i % 10 == 0:
            with open(log_path, 'a') as f:
                f.write('Calculating distance row {}\n'.format(i))

        if not hit_range_yet:
            j_start_ = j_start
        else:
            j_start_ = 0

        for j in range(i + 1 + j_start_, i + 1 + n_comp_window):

            try:
                i_index = all_indices[i]
                j_index = all_indices[j]
            except IndexError:
                continue

            if idx_flat not in comp_range:
                continue

            hit_range_yet = True

            idx_flat += 1

            checkpoints_1 = checkpoints_lists[job_indices[i]]
            checkpoints_2 = checkpoints_lists[job_indices[j]]

            try:
                checkpoint_1 = checkpoints_1['checkpoint_{}'.format(i_index)]
                checkpoint_2 = checkpoints_2['checkpoint_{}'.format(j_index)]
            except KeyError:
                continue

            if compare_args['align_checkpoints']:
                try:
                    #align_checkpoints_based_on_output(checkpoint_2, checkpoint_1,
                    #                                  n_inputs=compare_args['n_inputs'])
                    #align_checkpoints_based_on_output(checkpoint_2, checkpoint_1,
                    #                                  n_inputs=compare_args['n_inputs'])
                    pass
                except ValueError:
                    continue

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
                                                    N_avg=None, N_test=1000,
                                                    task=task)
            if SVCCA:
                SVCCA_distances[i, j] = SVCCA_distance(checkpoint_1,
                                                       checkpoint_2,
                                                       R=8)

            if CKA:
                CKA_distances[i, j] = CKA_distance(checkpoint_1, checkpoint_2)

            if aligned_graph:
                aligned_graph_distances[i, j] = aligned_graph_distance(checkpoint_1,
                                                                       checkpoint_2,
                                                                       node_diff_penalty=0,
                                                                       n_inputs=compare_args['n_inputs'],
                                                                       minimize_over_permutations=compare_args['minimize_over_permutations'])
            if node_diff:
                node_diff_distances[i, j] = node_diff_distance(checkpoint_1,
                                                               checkpoint_2)

            if node_drift:
                node_drift_distances[i, j] = np.sum(checkpoint_2['corr_node_distances'])

            if rec_weight:
                rec_weight_distances[i, j] = rec_weight_distance(checkpoint_1,
                                                                 checkpoint_2)
            if output_weight:
                output_weight_distances[i, j] = output_weight_distance(checkpoint_1,
                                                                      checkpoint_2)
            if weight_change_alignment:
                weight_change_alignment_distances[i, j] = weight_change_alignment_distance(checkpoint_1,
                                                                                           checkpoint_2)

            calculation_check[i, j] = 1

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
    if CKA:
        result['CKA_distances'] = CKA_distances
    if aligned_graph:
        result['aligned_graph_distances'] = aligned_graph_distances
    if node_diff:
        result['node_diff_distances'] = node_diff_distances
    if node_drift:
        result['node_drift_distances'] = node_drift_distances
    if rec_weight:
        result['rec_weight_distances'] = rec_weight_distances
    if output_weight:
        result['output_weight_distances'] = output_weight_distances
    if weight_change_alignment:
        result['weight_change_alignment_distances'] = weight_change_alignment_distances

    result['calculation_check'] = calculation_check

    if compare_args['n_comp_jobs'] > 1:
        for key in result.keys():
            result[key] = sparse.csr_matrix(result[key])

    result['all_indices'] = all_indices
    result['job_indices'] = job_indices
    result['analysis_job_names'] = analysis_job_names

    result['i_comp_job'] = i_comp_job
    save_dir = os.environ['SAVEDIR']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'result_{}'.format(i_comp_job))

    with open(save_path, 'wb') as f:
        pickle.dump(result, f)