import argparse, sys
sys.path.append('/scratch/oem214/vanilla-rtrl/')
from wrappers import cross_compare_analyzed_checkpoints

parser = argparse.ArgumentParser()
parser.add_argument('--name', dest='name')

args = parser.parse_args()

saved_run_root_name = 'analyze_{}'.format(args.name)

cross_compare_analyzed_checkpoints(saved_run_root_name)