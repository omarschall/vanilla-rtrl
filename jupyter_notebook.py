from cluster import start_jupyter_notebook, close_jupyter_notebook, start_axon_jupyter_notebook
import argparse

### Open or close a jupyter notebook depending on arguments passed when
### calling script

parser = argparse.ArgumentParser()
parser.add_argument('-open', dest='open_', action='store_true')
parser.add_argument('-close', dest='close_', action='store_true')
parser.add_argument('-axon', dest='axon_', action='store_true')
parser.set_defaults(open_=True, close_=False, axon_=False)

args = parser.parse_args()

if args.open_ and not args.close_:

    if args.axon_:
        start_axon_jupyter_notebook()
    else:
        start_jupyter_notebook()

if args.close_:

    if args.axon_:
        close_jupyter_notebook(username='om2382', domain='axon.rc.zi.columbia.edu')
    else:
        close_jupyter_notebook()