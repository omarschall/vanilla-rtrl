import subprocess

def sync_cluster(local_path='/Users/omarschall/vanilla-rtrl/',
                 scratch_path='/scratch/oem214/vanilla-rtrl/',
                 username='oem214', domain='greene.hpc.nyu.edu'):
    """Sync local code with scratch path on cluster."""

    remote_path = '{}@{}:{}'.format(username, domain, scratch_path)
    subprocess.run(['rsync', '-aav',
                    '--exclude', '.git',
                    '--exclude', 'files',
                    local_path, remote_path])