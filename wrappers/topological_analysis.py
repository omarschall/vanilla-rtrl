import os, pickle
from math import ceil
from cluster import write_job_file, submit_job
from wrappers.get_default_args import get_default_args

def topological_analysis(saved_run_name,
                         project_name='learning-dynamics',
                         module_name='vanilla-rtrl',
                         username='oem214',
                         **kwargs):
    """Wrapper script for taking a saved run by its name, analyzing each
    checkpoint in isolation, and comparing checkpoints by distance."""

    ### --- Calculate number of total jobs needed for analysis --- ###

    with open(os.path.join('saved_runs', saved_run_name), 'rb') as f:
        saved_run = pickle.load(f)

    sim = saved_run['sim']
    indices = list(range(0, sim.total_time_steps, sim.checkpoint_interval))
    n_checkpoints = len(indices)
    n_checkpoints_per_job = ceil(n_checkpoints / 1000)
    n_jobs = ceil(n_checkpoints / n_checkpoints_per_job)

    ### --- Define relevant paths --- ###

    project_dir = os.path.join('/scratch/{}/'.format(username), project_name)
    module_dir = os.path.join('/scratch/{}/'.format(username), module_name)
    cluster_main_dir = os.path.join(project_dir, 'cluster_main_scripts/')
    args_dir = os.path.join(project_dir, 'args')

    analyze_main_path = os.path.join(module_dir, 'analyze_main.py')
    compare_main_path = os.path.join(module_dir, 'compare_main.py')
    args_path = os.path.join(args_dir, saved_run_name)

    ### --- Gather and save args for analysis and comparison --- ###
    all_args_dict = get_default_args()
    all_args_dict.update(kwargs)

    with open(args_path, 'wb') as f:
        pickle.dump(all_args_dict, f)

    ### --- Submit analysis job script --- ###

    analysis_job_name = 'analyze_{}'.format(saved_run_name)

    write_job_file(analysis_job_name, py_file_name='analyze_main.py',
                   py_args='--name {}'.format(saved_run_name), ppn=16)
    get_ipython().system('cp {} {}'.format(analyze_main_path, cluster_main_dir))
    analysis_job_id = submit_job('../job_scripts/{}.s'.format(analysis_job_name),
                                 n_array=n_jobs,
                                 py_file_name='analyze_main.py')

    ### -- Submit compare job script when done

    compare_job_name = 'compare_{}'.format(saved_run_name)

    write_job_file(compare_job_name, py_file_name='compare_main.py',
                   py_args='--name {}'.format(saved_run_name))
    get_ipython().system('cp {} {}'.format(compare_main_path, cluster_main_dir))
    submit_job('../job_scripts/{}.s'.format(compare_job_name),
               n_array=1,
               py_file_name='compare_main.py',
               id_dependency=analysis_job_id)