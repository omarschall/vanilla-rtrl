import os
import numpy as np
import pickle

def submit_job(job_file_path, n_array,
               py_file_name=None,
               id_dependency=None,
               project_name='learning-dynamics',
               results_subdir='misc',
               module_name='vanilla-rtrl'):
    """Submit an array job in reference to a particular job file, with a
    specified number of sub-jobs. Creates directories for storing results."""

    ### --- Make results directory -- ###

    username = get_ipython().getoutput('whoami')[0] #WHOOOO AMMMMM I?
    if username == 'oem214':
        project_base = '/scratch/'
    if username == 'om2382':
        project_base = '/home/'
    job_name = job_file_path.split('/')[-1].split('.')[0]
    project_dir = os.path.join(project_base, username, project_name)
    results_dir = os.path.join(project_dir, 'results', results_subdir, job_name)
    code_dir = os.path.join(results_dir, 'code')
    main_dir = os.path.join(project_dir, 'cluster_main_scripts')
    if py_file_name is None:
        py_file_name = job_name+'.py'
    main_path = os.path.join(main_dir, py_file_name)
    job_path = os.path.join(project_dir, 'job_scripts', job_name + '.s')

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    if not os.path.exists(code_dir):
        os.mkdir(code_dir)

    ### --- Clear old results --- ###

    get_ipython().system('rm {}/result_*'.format(results_dir))

    ### --- Copy state of module to code dir --- ###

    module_dir = os.path.join(project_base, username, module_name)
    get_ipython().system('rsync -aav --exclude __pycache__ {} {}'.format(module_dir, code_dir))
    get_ipython().system('scp {} {}'.format(main_path, code_dir))

    ### --- Include dependencies on previous jobs --- ###

    dependency_arg = ''
    if id_dependency is not None:
        dependency_arg = '--dependency=afterok:{}'.format(id_dependency)

    ### -- Submit job --- ###
    sbatch_command = 'sbatch {} --array=1-{} {}'.format(dependency_arg,
                                                        n_array,
                                                        job_path)
    job_stdout = get_ipython().getoutput(sbatch_command)
    job_id = int(job_stdout[0].split(' ')[-1])

    return job_id

def write_job_file(job_name, py_file_name='main.py',
                   py_args='',
                   project_name='learning-dynamics',
                   results_subdir='misc',
                   nodes=1, ppn=1, mem=16, n_hours=8):
    """Create a job file for running a standard single-main-script job.

    Args:
        job_name (str): String specifying the name of the job (without .s
            extension.)
        py_file_name (str): Name of the python file (including .py extension)
            to be run on the cluster.
        project_name (str): Name of the project directory
        nodes (int): Number of nodes requested (use default of 1 probably)
        ppn (int): Number of processes per node (again use default 1)
        mem (int): Memory requirements in GB
        n_hours (int): Number of hours before job automatically terminates."""

    ### --- Define key paths --- ###

    username = get_ipython().getoutput('whoami')[0] #WHOOOO AMMMMM I?
    if username == 'oem214':
        project_base = '/scratch/'
    if username == 'om2382':
        project_base = '/home/'

    project_dir = os.path.join(project_base, username, project_name)
    sbatch_dir = os.path.join(project_dir, 'job_scripts')
    main_dir = os.path.join(project_dir, 'cluster_main_scripts')
    save_dir = os.path.join(project_dir, 'results', results_subdir, job_name)

    job_path = os.path.join(sbatch_dir, job_name + '.s')
    log_path = os.path.join(project_dir, 'logs', job_name)

    ### --- Define key commands and singularity environments --- ###

    command = ('pwd > {}.log; '.format(log_path)
              + 'date >> {}.log; '.format(log_path)
              + 'which python >> {}.log; '.format(log_path)
              + 'python {} {}\n'.format(py_file_name, py_args))

    if username == 'oem214':
        overlay = '/home/{}/pytorch1.7.0-cuda11.0.ext3:ro'.format(username)
        singularity_dir = '/scratch/work/public/singularity/'
        singularity_name = 'cuda11.0-cudnn8-devel-ubuntu18.04.sif'
        singularity_path = os.path.join(singularity_dir, singularity_name)
        singularity_exe_path = '/share/apps/singularity/bin/singularity'
        execute_command = ('{} exec '.format(singularity_exe_path)
                           + '--overlay {} {} '.format(overlay, singularity_path)
                           + 'bash -c "source /ext3/env.sh; {}"'.format(command))
    if username == 'om2382':
        execute_command = ('ml load anaconda3-2019.03; '
                           + 'conda activate v-rtrl; '
                           + command)

    ### --- Write job file -- ###

    with open(job_path, 'w') as f:
        f.write(
            '#! /bin/bash\n'
            + '\n'
            + '#SBATCH --nodes={}\n'.format(nodes)
            + '#SBATCH --ntasks-per-node=1\n'
            + '#SBATCH --cpus-per-task={}\n'.format(ppn)
            + '#SBATCH --mem={}GB\n'.format(mem)
            + '#SBATCH --time={}:00:00\n'.format(n_hours)
            + '#SBATCH --job-name={}\n'.format(job_name[0:16])
            + '#SBATCH --output={}\n'.format(log_path)
            + '\n'
            + 'module purge\n'
            + 'SAVEDIR={}\n'.format(save_dir)
            + 'export SAVEDIR\n'
            + 'cd {}\n'.format(main_dir)
            + execute_command)

    return job_path

# def write_dependent_job_script(job_name, job_name_1, job_name_2,
#                                project_name='learning-dynamics',
#                                username='oem214',):
#
#     project_dir = os.path.join('/scratch/', username, project_name)
#     sbatch_dir = os.path.join(project_dir, 'job_scripts')
#     job_path = os.path.join(sbatch_dir, job_name)
#
#     ### --- Write job file -- ###
#
#     with open(job_path, 'w') as f:
#         f.write(
#             '#! /bin/bash\n'
#             + '\n'
#             + 'jid=$(sbatch --array=1-{} {}.s | cut -d " " -f 4)\n'.format(n_array, job_name_1))
#             + 'sbatch --dependency=afterok:${jid} {}'.format(job_name_2))
#
#     return job_path

def unpack_processed_data(job_file_path,
                          project_name='learning-dynamics',
                          results_subdir='misc',
                          username='oem214'):
    """Unpack processed data from an array job."""

    username = get_ipython().getoutput('whoami')[0] #WHOOOO AMMMMM I?
    if username == 'oem214':
        project_base = '/scratch/'
    if username == 'om2382':
        project_base = '/home/'

    job_name = job_file_path.split('/')[-1].split('.')[0]
    project_dir = os.path.join(project_base, username, project_name)
    data_dir = os.path.join(project_dir, 'results', results_subdir, job_name)
    dir_list = sorted([s for s in os.listdir(data_dir) if 'result' in s])

    max_seed = 0

    ### --- Find and organized unique macro configs --- ###

    for i_file, file in enumerate(dir_list):

        with open(os.path.join(data_dir, file), 'rb') as f:
            result = pickle.load(f)

        if i_file == 0:

            configs_array = {key: [] for key in result['config'].keys()}
            key_order = [key for key in result['config'].keys()]

        for key in result['config'].keys():
            if result['config'][key] not in configs_array[key]:
                configs_array[key].append(result['config'][key])

        max_seed = np.maximum(max_seed, result['i_seed'])

    configs_array['i_seed'] = list(range(max_seed + 1))
    key_order.append('i_seed')

    ### --- Sort individual config dimensions --- ####

    for key in configs_array.keys():

        configs_array[key] = sorted(configs_array[key])

    ### --- Determine shape of processed data --- ###

    array_dims = [len(configs_array[key]) for key in key_order]
    processed_data_example = result['processed_data']
    if type(processed_data_example) != np.float64:
        array_dims += [len(processed_data_example)]
    results_array = np.zeros(array_dims)

    ### --- Put data in array with shape matched to config arrays --- ###

    sim_dict = {}
    for i_file, file in enumerate(dir_list):

        with open(os.path.join(data_dir, file), 'rb') as f:
            result = pickle.load(f)

        sim_dict_key = ''
        index = []
        for key in key_order:
            try:
                index.append(configs_array[key].index(result['config'][key]))
                sim_dict_key += (str(result['config'][key]) + '_')
            except KeyError:
                index.append(result['i_seed'])
                sim_dict_key += (str(result['i_seed']))
        index = tuple(index)

        results_array[index] = result['processed_data']
        try:
            sim_dict[sim_dict_key] = result['sim']
        except AttributeError:
            pass

    return configs_array, results_array, key_order, sim_dict