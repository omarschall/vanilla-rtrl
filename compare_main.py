import argparse, os, pickle, sys
sys.path.append('/home/om2382/vanilla-rtrl/')
from wrappers import compare_analyzed_checkpoints

parser = argparse.ArgumentParser()
parser.add_argument('--name', dest='name')

args = parser.parse_args()

analysis_job_name = 'analyze_{}'.format(args.name)

args_path = os.path.join('/home/om2382/learning-dynamics/args/', args.name)

with open(args_path, 'rb') as f:
    all_args = pickle.load(f)

compare_args = {k.split('compare_')[1]: all_args[k]
                for k in all_args.keys() if 'compare_' in k}

compare_analyzed_checkpoints(analysis_job_name,
                             compare_args=compare_args,
                             notebook_dir=all_args['notebook_dir'])