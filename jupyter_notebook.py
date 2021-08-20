from cluster import start_jupyter_notebook, close_jupyter_notebook
import argparse

### Open or close a jupyter notebook depending on arguments passed when
### calling script

parser = argparse.ArgumentParser()
parser.add_argument('-open', dest='open_', action='store_true')
parser.add_argument('-close', dest='close_', action='store_true')
parser.set_defaults(open_=True, close_=False)

args = parser.parse_args()

if args.open_ and not args.close_:

    start_jupyter_notebook()

if args.close_:

    close_jupyter_notebook()