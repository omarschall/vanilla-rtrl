import subprocess, os
from cluster.sync_cluster import sync_cluster
try:
    import webbrowser
    import appscript
except ModuleNotFoundError:
    pass
import time

def start_jupyter_notebook(local_module_path='/Users/omarschall/vanilla-rtrl/',
                           module_name='vanilla-rtrl/',
                           project_name='learning-dynamics',
                           username='oem214', domain='greene.hpc.nyu.edu'):
    """For given user data, opens a jupyter notebook on the cluster, accesses
    port/token info, and opens a web browser with jupyter notebook running.

    Requires:
    (1) Singularity files on cluster to use proper python environment
    (2) jupyter.s file that submits a job with name "jupyter", accesses the
        Singularity, and runs jupyter notebook to output file jupyter.o .

    Starts by updating cluster with local python code base. Then clears
    old jupyter.o files, runs jupyter notebook via sbatch, continuosly checks
    for relevant connection info (port and token) in jupyter.o file, and when
    available opens a browser connecting to this notebook job."""

    sync_cluster(local_module_path=local_module_path, module_name=module_name,
                 username=username, domain=domain)
    remote = '{}@{}'.format(username, domain)

    scratch_path = '/scratch/{}/'.format(username)
    project_path = os.path.join(scratch_path, project_name)
    jupyter_path = os.path.join(project_path, 'jupyter_notebook')
    jupyter_slurm_path = os.path.join(jupyter_path, 'jupyter.s')
    jupyter_output_path = os.path.join(jupyter_path, 'jupyter.o')

    ### --- Clear old notebook data files and cancel old notebooks --- ###

    subprocess.run(['ssh', remote,
                    'rm', jupyter_output_path])

    ### --- Run new jupyter notebook --- ###

    subprocess.run(['ssh', remote,
                    'sbatch', jupyter_slurm_path])


    ### --- Repeatedly check for connection info in output file --- ###

    done = False
    while not done:

        sp = subprocess.run(['ssh', remote,
                             'tail', '-2', jupyter_output_path,
                             '|', 'head', '-1'],
                            capture_output=True)

        line_info = str(sp.stdout, 'utf-8')

        if 'http' not in line_info:
            continue

        done = True

    ### -- Extract url and port info from output file --- ###

    url = line_info.split(' ')[-1]
    port = url.split('/?token')[0].split('localhost:')[-1]

    ### --- Open terminal window and SSH into cluster --- ###

    cluster_login = 'ssh -L {}:localhost:{} {}'.format(port, port, remote)
    appscript.app('Terminal').do_script(cluster_login)
    time.sleep(5) #Give terminal a few seconds to run SSH

    ### --- Open jupyter notebook in browser --- ###

    webbrowser.open(url)