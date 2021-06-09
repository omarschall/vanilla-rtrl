#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 19:19:46 2018

@author: omarschall
"""

import subprocess
import os
import numpy as np
from pdb import set_trace
import pickle

def clear_results(job_file, data_path='/Users/omarschall/cluster_results/vanilla-rtrl/'):

    job_name = job_file.split('/')[-1].split('.')[0]
    data_dir = os.path.join(data_path, job_name)

    subprocess.run(['rm', data_dir+'/*_*'])

def retrieve_results(job_file, scratch_path='/scratch/oem214/vanilla-rtrl/',
               username='oem214', domain='greene.hpc.nyu.edu'):

    #job_name = job_file.split('/')[-1].split('.')[0]
    job_name = '.'.join(job_file.split('/')[-1].split('.')[:-1])
    data_path = '/Users/omarschall/cluster_results/vanilla-rtrl/'
    data_dir = os.path.join(data_path, job_name)

    source_path = username+'@'+domain+':'+scratch_path+'library/'+job_name+'/'

    subprocess.run(['rsync', '-aav', source_path, data_dir])

def submit_job(job_file, n_array, scratch_path='/scratch/oem214/vanilla-rtrl/',
               username='oem214', domain='greene.hpc.nyu.edu'):

    job_name = job_file.split('/')[-1]#
    data_path = '/Users/omarschall/cluster_results/vanilla-rtrl/'
    data_dir = os.path.join(data_path, job_name.split('.')[0])

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        code_dir = os.path.join(data_dir, 'code')
        os.mkdir(code_dir)

    code_dir = os.path.join(data_dir, 'code')

    #copy main script to results dir
    subprocess.run(['rsync',
                    '-aav',
                    '--exclude', '.git',
                    '/Users/omarschall/vanilla-rtrl/',
                    code_dir])

    subprocess.run(['rsync',
                    '-aav',
                    '--exclude', '.git',
                    '/Users/omarschall/vanilla-rtrl/',
                    username+'@'+domain+':'+scratch_path])

    subprocess.run(['scp', job_file, username+'@'+domain+':/home/oem214/'])

    subprocess.run(['ssh', username+'@'+domain,
                    'sbatch', '--array=1-'+str(n_array), job_name])

def submit_job(job_file_path, n_array,
               project_name='learning-dynamics',
               module_name='vanilla-rtrl',
               username='oem214'):

    ### --- Make resutls directory -- ###

    job_name = job_file_path.split('/')[-1].split('.')[0]
    results_dir = os.path.join('/scratch/', username,
                               project_name, 'results', job_name)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
        code_dir = os.path.join(results_dir, 'code')
        os.mkdir(code_dir)

    ### --- Copy state of module to code dir --- ###

    module_dir = os.path.join('/scratch/', username, module_name)
    subprocess.run(['rsync', '-aav', module_dir, code_dir])

    ### -- Submit job --- ###
    subprocess.run(['sbatch', '--array=1-{}'.format(n_array), job_name + '.s'])




def write_job_file(job_name, py_file_name='main.py',
                   project_name='learning-dynamics',
                   username='oem214',
                   nodes=1, ppn=1, mem=16, n_hours=24):
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

    project_dir = os.path.join('/scratch/', username, project_name)
    sbatch_dir = os.path.join(project_dir, 'job_scripts')
    main_dir = os.path.join(project_dir, 'cluster_main_scripts')
    save_dir = os.path.join(project_dir, 'results', job_name)

    job_path = os.path.join(sbatch_dir, job_name + '.s')
    log_path = os.path.join(project_dir, 'logs', job_name + '.o')

    ### --- Define key commands and singularity environments --- ###

    command = ('pwd > {}.log; '.format(log_path)
              + 'date >> {}.log; '.format(log_path)
              + 'which python >> {}.log; '.format(log_path)
              + 'python {}\n'.format(py_file_name))

    overlay = '/home/{}/pytorch1.7.0-cuda11.0.ext3:ro'.format(username)
    singularity_dir = '/scratch/work/public/singularity/'
    singularity_name = 'cuda11.0-cudnn8-devel-ubuntu18.04.sif'
    singularity_path = os.path.join(singularity_dir, singularity_name)

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
            + 'singularity exec --nv '
            + '--overlay {} {} '.format(overlay, singularity_path)
            + 'bash -c "source /ext3/env.sh; {}"'.format(command))

    return job_path

def process_results(job_file):

    job_name = job_file.split('/')[-1].split('.')[0]
    data_path = os.path.join('/Users/omarschall/cluster_results/vanilla-rtrl/',
                             job_name)
    dir_list = os.listdir(data_path)
    dir_list.pop(dir_list.index('code'))
    # for file in dir_list:
    #     if 'rnn' not in file:
    #         del(dir_list[dir_list.index(file)])

    max_seed = 0

    for i_file, file in enumerate(dir_list):

        with open(os.path.join(data_path, file), 'rb') as f:
            data = pickle.load(f)

        if i_file == 0:

            configs_array = {key : [] for key in data['config'].keys()}
            key_order = [key for key in data['config'].keys()]

        for key in data['config'].keys():
            if data['config'][key] not in configs_array[key]:
                configs_array[key].append(data['config'][key])

        max_seed = np.maximum(max_seed, data['i_seed'])

    configs_array['i_seed'] = list(range(max_seed + 1))
    key_order.append('i_seed')

    for key in configs_array.keys():

        configs_array[key] = sorted(configs_array[key])

    array_dims = [len(configs_array[key]) for key in key_order]
    #processed_data_example = [d for d in data['processed_data'].values()][0]
    processed_data_example = np.array([d for d in data['processed_data'].values()])
    #processed_data_example = 0.6
    if type(processed_data_example) != np.float64:
        #set_trace()
        array_dims += [len(processed_data_example)]
    results_array = np.zeros(array_dims)

    #set_trace()

    #Put data in array
    sim_dict = {}
    for i_file, file in enumerate(dir_list):

        with open(os.path.join(data_path, file), 'rb') as f:
            data = pickle.load(f)

        sim_dict_key = ''
        index = []
        for key in key_order:
            try:
                index.append(configs_array[key].index(data['config'][key]))
                sim_dict_key += (str(data['config'][key]) + '_')
            except KeyError:
                index.append(data['i_seed'])
                sim_dict_key += (str(data['i_seed']))
        index = tuple(index)
        #set_trace()
        #processed_data = [d for d in data['processed_data'].values()][0]
        losses = np.array([data['processed_data']['task_{}'.format(i)] for i in range(1,4)] +
                          [data['processed_data']['combined_task']])
        #set_trace()
        results_array[index] = losses
        #results_array[index] = data['processed_data']['test_loss']
        #results_array[index] = data['processed_data']
        try:
            sim_dict[sim_dict_key] = data['sim']
        except AttributeError:
            pass

    return configs_array, results_array, key_order, sim_dict