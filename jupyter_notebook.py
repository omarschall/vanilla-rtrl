from cluster import start_jupyter_notebook, close_jupyter_notebook

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-open', dest='open_', action='store_true')
parser.add_argument('-close', dest='close_', action='store_true')
parser.set_defaults(open_=False, close_=False)

args = parser.parse_args()

if args.open_:

    start_jupyter_notebook()

if args.close_:

    close_jupyter_notebook()