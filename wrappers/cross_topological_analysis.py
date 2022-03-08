import os, pickle
from math import ceil
from cluster import write_job_file, submit_job
from wrappers.get_default_args import get_default_args

def cross_topological_analysis(saved_run_root_name,
                               project_name='learning-dynamics',
                               module_name='vanilla-rtrl',
                               results_subdir='misc',
                               username='oem214',
                               ppn=16,
                               compare_only=False,
                               n_checkpoints_per_job_=None,
                               **kwargs):
    """Wrapper script for taking a set of saved runs by root name, analyzing
    each checkpoint in isolation, and comparing checkpoints by distance."""

    analysis_job_ids = []

    ### --- Identify all saved runs fitting root name --- ###

    saved_run_names = [saved_run for saved_run in os.listdir('saved_runs')
                       if saved_run_root_name in saved_run]
    saved_run_names = sorted(saved_run_names)

    ### --- Define relevant paths --- ###

    project_dir = os.path.join('/scratch/{}/'.format(username), project_name)
    module_dir = os.path.join('/scratch/{}/'.format(username), module_name)
    cluster_main_dir = os.path.join(project_dir, 'cluster_main_scripts/')
    args_dir = os.path.join(project_dir, 'args')

    analyze_main_path = os.path.join(module_dir, 'analyze_main.py')
    args_path = os.path.join(args_dir, saved_run_root_name)

    ### --- Gather and save args for analysis and comparison --- ###

    all_args_dict = get_default_args()
    all_args_dict.update(kwargs)
    all_args_dict['n_checkpoints_per_job_'] = n_checkpoints_per_job_
    all_args_dict['results_subdir'] = results_subdir

    with open(args_path, 'wb') as f:
        pickle.dump(all_args_dict, f)

    if not compare_only:

        for saved_run_name in saved_run_names:

            ### --- Calculate number of total jobs needed for analysis --- ###

            with open(os.path.join('saved_runs', saved_run_name), 'rb') as f:
                saved_run = pickle.load(f)

            sim = saved_run['sim']
            if type(sim.checkpoint_interval) is int:
                indices = list(range(0, sim.total_time_steps, sim.checkpoint_interval))
            elif type(sim.checkpoint_interval) is list:
                indices = sim.checkpoint_interval
            n_checkpoints = len(indices)
            if n_checkpoints_per_job_ is None:
                n_checkpoints_per_job = ceil(n_checkpoints / 1000)
            else:
                n_checkpoints_per_job = n_checkpoints_per_job_
            n_jobs = ceil(n_checkpoints / n_checkpoints_per_job)

            ### --- Submit analysis job script --- ###

            analysis_job_name = 'analyze_{}'.format(saved_run_name)

            write_job_file(analysis_job_name, py_file_name='analyze_main.py',
                           results_subdir=results_subdir,
                           py_args='--name {}'.format(saved_run_name), ppn=ppn)
            get_ipython().system('cp {} {}'.format(analyze_main_path, cluster_main_dir))
            analysis_job_id = submit_job('../job_scripts/{}.s'.format(analysis_job_name),
                                         n_array=n_jobs,
                                         results_subdir=results_subdir,
                                         py_file_name='analyze_main.py')
            analysis_job_ids.append(analysis_job_id)

        analysis_id_dependency = ':'.join([str(id_) for id_ in analysis_job_ids])
    else:
        analysis_id_dependency = None

    ### -- Submit compare job script when done

    compare_job_name = 'cross_compare_{}'.format(saved_run_root_name)
    compare_main_path = os.path.join(module_dir, 'cross_compare_main.py')

    write_job_file(compare_job_name, py_file_name='cross_compare_main.py',
                   results_subdir=results_subdir,
                   py_args='--name {}'.format(saved_run_root_name),
                   n_hours=12)
    get_ipython().system('cp {} {}'.format(compare_main_path, cluster_main_dir))
    submit_job('../job_scripts/{}.s'.format(compare_job_name),
               n_array=1,
               results_subdir=results_subdir,
               py_file_name='cross_compare_main.py',
               id_dependency=analysis_id_dependency)
