from dynamics import *

def analyze_individual_checkpoint(checkpoint, task, data, all_args):
    """For a given checkpoint and set of analysis args, analyses that one checkpoint.

    Analysis includes finding the fixed points, clustering them, and extracting
    autonomous and input-driven transition probabilities."""

    FP_args = {k.split('FP_')[1]: all_args[k]
           for k in all_args.keys() if 'FP_' in k}
    graph_args = {k.split('graph_')[1]: all_args[k]
                  for k in all_args.keys() if 'graph_' in k}

    analysis_args = {k: FP_args[k] for k in FP_args if k != 'find_FPs'}
    analyze_checkpoint(checkpoint, data, verbose=False,
                       parallelize=True, **analysis_args)

    get_graph_structure(checkpoint, parallelize=True,
                        background_input=0,  **graph_args)
    get_input_dependent_graph_structure(checkpoint,
                                        inputs=task.probe_inputs,
                                        **graph_args)

    return checkpoint