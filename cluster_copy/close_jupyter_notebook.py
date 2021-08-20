import subprocess

def close_jupyter_notebook(username='oem214', domain='greene.hpc.nyu.edu'):
    """Closes down jupyter notebook job on cluster."""

    remote = '{}@{}'.format(username, domain)
    #Cancel all jobs for your user with name "jupyter"
    subprocess.run(['ssh', remote, 'scancel', '-n', 'jupyter', '-u', username])