import os

def get_default_args():
    """Produce and return a dictionary with the default args for the topological
    analysis pipeline."""

    default_FP_args = {'find_FPs': True, 'sigma_pert': 0.5, 'N': 1200,
                       'KE_criterion': 0.001, 'N_iters': 10000,
                       'same_LR_criterion': 9000, 'sigma': 0,
                       'DB_eps': 0.5}
    default_test_args = {'N': 10000, 'n_PCs': 3, 'save_data': False}
    default_graph_args = {'N': 100, 'time_steps': 50, 'epsilon': 0.01,
                          'sigma': 0}
    default_compare_args = {'wasserstein': False,
                            'VAE_': False,
                            'PC1': False,
                            'PC2': False,
                            'PC3': False,
                            'SVCCA': False,
                            'CKA': False,
                            'aligned_graph': True,
                            'node_diff': True,
                            'node_drift': True,
                            'rec_weight': True,
                            'output_weight': True,
                            'weight_change_alignment': False,
                            'n_inputs': 6,
                            'align_checkpoints': True,
                            'n_comp_window': 'full'}

    all_args = {}

    meta_keys = ['FP', 'test', 'graph', 'compare']
    arg_dicts = [default_FP_args, default_test_args, default_graph_args,
                 default_compare_args]

    for meta_key, arg_dict in zip(meta_keys, arg_dicts):
        for key in arg_dict.keys():

            all_args['{}_{}'.format(meta_key, key)] = arg_dict[key]

    all_args['n_checkpoints_per_job_'] = None
    all_args['notebook_dir'] = os.getcwd()

    return all_args