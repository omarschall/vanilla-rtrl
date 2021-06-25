import os
import pickle
from dynamics import *
from math import ceil

default_analysis_args = {'sigma_pert': 0.5, 'N': 1200,
                         'N_iters': 10000, 'same_LR_criterion': 9000,
                         'sigma': 0}
default_graph_args = {'N': 100, 'time_steps': 50, 'epsilon': 0.01,
                      'sigma': 0}
default_input_graph_args = {'N': 100, 'time_steps': 50, 'epsilon': 0.01,
                            'sigma': 0}

def analyze_training_run(saved_run_name,
                         analysis_args=default_analysis_args,
                         graph_args=default_graph_args,
                         input_graph_args=default_input_graph_args,
                         username='oem214',
                         project_name='learning-dynamics'):
    """For a given simulation (containing checkpoints), analyzes some subset
    of the checkpoints for a given job ID depending on how many total.

    Analysis includes finding the fixed points, clustering them, and extracting
    autonomous and input-driven transition probabilities."""

    ### --- Define relevant paths --- ###

    project_dir = os.path.join('/scratch/{}/'.format(username), project_name)
    analysis_job_name = 'analyze_{}'.format(saved_run_name)
    log_path = os.path.join(project_dir, 'logs/' + analysis_job_name) + '.o.log'

    ### --- Load sim and task --- ###

    saved_runs_dir = os.path.join(project_dir, 'notebooks', 'saved_runs')

    with open(os.path.join(saved_runs_dir, saved_run_name), 'rb') as f:
        saved_run = pickle.load(f)

    sim = saved_run['sim']
    task = saved_run['task']

    ### --- Determine which checkpoints to analyze for this job id --- ###

    indices = list(range(0, sim.total_time_steps, sim.checkpoint_interval))
    n_checkpoints = len(indices)
    n_checkpoints_per_job = ceil(n_checkpoints / 1000)
    i_job = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
    i_start = sim.checkpoint_interval * n_checkpoints_per_job * i_job
    i_end = sim.checkpoint_interval * n_checkpoints_per_job * (i_job + 1)

    ### --- Analyze each checkpoint --- ###

    result = {}
    data = task.gen_data(100, 30000)

    for i_checkpoint in range(i_start, i_end, sim.checkpoint_interval):
        with open(log_path, 'a') as f:
            f.write('Analyzing chekpoint {}\n'.format(i_checkpoint))

        try:
            checkpoint = sim.checkpoints[i_checkpoint]
        except KeyError:
            continue
        analyze_checkpoint(checkpoint, data, verbose=False, parallelize=False,
                           **analysis_args)

        get_graph_structure(checkpoint, parallelize=False, background_input=0,
                            **graph_args)
        get_input_dependent_graph_structure(checkpoint,
                                            inputs=task.probe_inputs,
                                            **input_graph_args)

        result['checkpoint_{}'.format(i_checkpoint)] = deepcopy(checkpoint)

    result['i_job'] = i_job
    save_dir = os.environ['SAVEDIR']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'result_{}'.format(i_job))

    with open(save_path, 'wb') as f:
        pickle.dump(result, f)