import argparse, os, pickle, sys
sys.path.append('/scratch/oem214/vanilla-rtrl/')
from wrappers import analyze_training_run

parser = argparse.ArgumentParser()
parser.add_argument('--name', dest='name')

args = parser.parse_args()

saved_run_name = args.name
args_path = os.path.join('/scratch/oem214/learning-dynamics/args/', args.name)

with open(args_path, 'rb') as f:
    all_args = pickle.load(f)

FP_args = {k.split('FP_')[1]: all_args[k]
           for k in all_args.keys() if 'FP_' in k}
test_args = {k.split('test_')[1]: all_args[k]
             for k in all_args.keys() if 'test_' in k}
graph_args = {k.split('graph_')[1]: all_args[k]
              for k in all_args.keys() if 'graph_' in k}

analyze_training_run(saved_run_name,
                     FP_args=FP_args,
                     test_args=test_args,
                     graph_args=graph_args)