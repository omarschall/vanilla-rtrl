import subprocess, os

def sync_cluster(local_module_path='/Users/omarschall/vanilla-rtrl/',
                 module_name='vanilla-rtrl',
                 username='oem214', domain='greene.hpc.nyu.edu'):
    """Sync local code with module path on cluster."""

    scratch_path = '/scratch/{}/'.format(username)

    module_path = os.path.join(scratch_path, module_name)


    remote_path = '{}@{}:{}'.format(username, domain, module_path)
    subprocess.run(['rsync', '-aav',
                    '--exclude', '.git',
                    '--exclude', 'files',
                    local_module_path, remote_path])

def sync_columbia_cluster(local_module_path='/Users/omarschall/vanilla-rtrl/',
                          module_name='vanilla-rtrl',
                          username='om2382', domain='axon.rc.zi.columbia.edu'):
    """Sync local code with module path on cluster."""

    scratch_path = '/home/{}/'.format(username)

    module_path = os.path.join(scratch_path, module_name)


    remote_path = '{}@{}:{}'.format(username, domain, module_path)
    subprocess.run(['rsync', '-aav',
                    '--exclude', '.git',
                    '--exclude', 'files',
                    local_module_path, remote_path])