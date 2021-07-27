import argparse, sys
sys.path.append('/scratch/oem214/vanilla-rtrl/')
from wrappers import cross_compare_analyzed_checkpoints

parser = argparse.ArgumentParser()
parser.add_argument('--name', dest='name')

args = parser.parse_args()

cross_compare_analyzed_checkpoints(args.name)