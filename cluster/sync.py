import os
import subprocess

def sync_cluster(home_path='/Users/omarschall/vanilla-rtrl/',
                 scratch_path='/scratch/oem214/vanilla-rtrl/',
                 username='oem214', domain='greene.hpc.nyu.edu'):

    remote_path = '{}@{}:{}'.format(username, domain, scratch_path)
    subprocess.run(['rsync', '-aav',
                    '--exclude', '.git',
                    '--exclude', 'files',
                    home_path, remote_path])