import argparse, sys
sys.path.append('/scratch/oem214/vanilla-rtrl/')
from wrappers import analyze_training_run

parser = argparse.ArgumentParser()
parser.add_argument('--name', dest='name')

args = parser.parse_args()

saved_run_name = args.name

analysis_args =

analyze_training_run(saved_run_name)