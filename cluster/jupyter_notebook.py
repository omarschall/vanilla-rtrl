import subprocess
from sync import sync_cluster
import webbrowser
import appscript
import time

def start_jupyter_notebook(home_path='/Users/omarschall/vanilla-rtrl/',
                           scratch_path='/scratch/oem214/vanilla-rtrl/',
                           username='oem214', domain='greene.hpc.nyu.edu'):

    sync_cluster(home_path=home_path, scratch_path=scratch_path,
                 username=username, domain=domain)
    remote = '{}@{}'.format(username, domain)

    ### --- Clear old notebook data files and cancel old notebooks --- ###

    subprocess.run(['ssh', remote,
                    'rm', '~/jupyter_notebook/jupyter.o'])

    ### --- Run new jupyter notebook --- ###

    subprocess.run(['ssh', remote,
                    'sbatch', 'jupyter_notebook/jupyter.s'])


    ### --- Repeatedly check for connection info in output file --- ###

    done = False
    while not done:

        sp = subprocess.run(['ssh', remote,
                             'tail', '-2', 'jupyter_notebook/jupyter.o',
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

if __name__ == '__main__':
    start_jupyter_notebook()