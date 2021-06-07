import os
import pickle
from dynamics import *
from math import ceil

def analyze_training_run(sim, task, job_name, sigma=0):

    indices = list(range(0, sim.total_time_steps, sim.checkpoint_interval))
    n_checkpoints = len(indices)
    n_checkpoints_per_job = ceil(n_checkpoints / 1000)

    n_jobs = n_checkpoints / n_checkpoints_per_job
    n_jobs_per_checkpoint = n_checkpoints / n_jobs

    i_job = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
    i_start = sim.checkpoint_interval * n_jobs_per_checkpoint * i_job
    i_end = sim.checkpoint_interval * n_jobs_per_checkpoint * (i_job + 1)

    print('Analyzing checkpoints...')

    # Progress logging
    scratch_path = '/scratch/oem214/vanilla-rtrl/'
    log_path = os.path.join(scratch_path, 'log/' + job_name) + '_{}.log'.format(i_job)

    inputs = task.probe_inputs
    result = {}

    data = task.gen_data(100, 30000)

    for i_checkpoint in range(i_start, i_end, sim.checkpoint_interval):
        with open(log_path, 'a') as f:
            f.write('Analyzing chekpoint {}\n'.format(i_checkpoint))

        checkpoint = sim.checkpoints[i_checkpoint]
        analyze_checkpoint(checkpoint, data, verbose=False,
                           sigma_pert=0.5, N=500, parallelize=False,
                           N_iters=6000, same_LR_criterion=5000,
                           context=contexts[0], sigma=sigma)

        get_graph_structure(checkpoint, parallelize=False,
                            epsilon=0.01, background_input=0)
        get_input_dependent_graph_structure(checkpoint,
                                            inputs=task.probe_inputs,
                                            contexts=None)

        result['checkpoint_{}'.format(i_checkpoint)] = deepcopy(checkpoint)

    result['i_job'] = i_job
    result['config'] = params
    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'result_'+str(i_job))

    with open(save_path, 'wb') as f:
        pickle.dump(result, f)