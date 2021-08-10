import numpy as np
from core import Simulation
from scipy.spatial import distance
from utils import norm
from copy import deepcopy

def get_test_sim_data(checkpoint, test_data, sigma=0):
    """Get hidden states from a test run for a given checkpoint"""

    rnn = deepcopy(checkpoint['rnn'])
    test_sim = Simulation(rnn)
    test_sim.run(test_data, mode='test',
                 monitors=['rnn.a'],
                 verbose=False,
                 a_initial=rnn.a.copy(),
                 sigma=sigma)

    return test_sim.mons['rnn.a']

def align_checkpoints(checkpoint, reference_checkpoint, n_inputs=6):
    """Finds the best possible alignment between fixed point clusters (as
    found by DBSCAN) between an reference checkpoint and the second. Only
    the second is changed.

    First, we define a few key integers, based on n_centroids_ref and
    n_centroids, the numbers of centroids in each checkpoint, respectively.

    n_shared_max (int): This is the maximum number of shared

    We then take an 'outer product' of distances between each pair of
    centroids. Then we take the n_shared_max smallest distance pairs and
    provisionally align these centroids."""

    ref_nodes = reference_checkpoint['nodes']
    nodes = checkpoint['nodes']

    n_nodes = nodes.shape[0]
    I_x = []
    I_y = []
    D = distance.cdist(ref_nodes, nodes)
    corr_node_distances = []
    while len(I_x) < n_nodes:

        d = np.argmin(D)
        d_min = np.min(D)
        if d_min == np.inf:
            break
        else:
            corr_node_distances.append(d_min)

        x, y = (d // n_nodes), (d % n_nodes)

        I_x.append(x)
        I_y.append(y)

        D[x,:] = np.inf
        D[:,y] = np.inf

    I_ = [I_y[i_x] for i_x in np.argsort(I_x)]
    I_f = sorted(I_x)
    I_b = sorted(I_)
    extra_indices = list(range(n_nodes))
    [extra_indices.remove(i) for i in I_]
    I = I_ + extra_indices

    keys = ['adjmat_input_{}'.format(i) for i in range(n_inputs)] + ['nodes']

    for key in keys:

        if 'adjmat' in key:
            checkpoint[key] = checkpoint[key][I][:,I]
            checkpoint['backshared_' + key] = checkpoint[key][I_b][:,I_b]
            reference_checkpoint['forwardshared_' + key] = reference_checkpoint[key][I_f][:,I_f]

        if key == 'nodes':
            checkpoint[key] = checkpoint[key][I]

    checkpoint['corr_node_distances'] = [corr_node_distances[i_x] for i_x in np.argsort(I_x)]

def linearly_interpolate_checkpoints(sim, start_checkpoint, end_checkpoint,
                                     density):
    """Create an otherwise empty simulation object containing checkpoints
    with parameters that linearly interpolate between a start and end point."""

    sim.checkpoints = {}

    total_time = end_checkpoint['i_t']
    timeline = np.arange(0, total_time + density, density)

    rnn_start = start_checkpoint['rnn']
    rnn_end = end_checkpoint['rnn']

    slope = 1 / total_time
    W_in_diff = rnn_end.W_in - rnn_start.W_in
    W_rec_diff = rnn_end.W_rec - rnn_start.W_rec
    W_out_diff = rnn_end.W_out - rnn_start.W_out
    b_rec_diff = rnn_end.b_rec - rnn_start.b_rec
    b_out_diff = rnn_end.b_out - rnn_start.b_out

    for i_t in timeline:

        checkpoint = {'i_t': i_t}

        scale = i_t * slope

        W_in = rnn_start.W_in + scale * W_in_diff
        W_rec = rnn_start.W_rec + scale * W_rec_diff
        W_out = rnn_start.W_out + scale * W_out_diff
        b_rec = rnn_start.b_rec + scale * b_rec_diff
        b_out = rnn_start.b_out + scale * b_out_diff

        rnn = deepcopy(rnn_start)
        rnn.W_in = W_in
        rnn.W_rec = W_rec
        rnn.W_out = W_out
        rnn.b_rec = b_rec
        rnn.b_out = b_out
        checkpoint['rnn'] = rnn

        sim.checkpoints[i_t] = checkpoint

    return sim


def concatenate_simulation_checkpoints(simulations):

    dicts = [s.checkpoints for s in simulations]

    ret = dicts.pop(0)

    for d in dicts:
        i_t_shift = max(ret.keys()) + 1
        del ret[i_t_shift - 1]
        for key in d.keys():
            ret[key + i_t_shift] = d[key]

    return ret

def align_checkpoints_based_on_output(checkpoint, reference_checkpoint,
                                      n_inputs=6):
    """Align a pair of checkpoints, i.e. re-index the arbitrary order of fixed
    points implied by the transition probability matrices.

    The alignment is based on the distances between the fixed points in output
    space, i.e. when each fixed point is mapped to its output by its own RNN."""

    #Get local access to the stable fixed points from each checkpoint.
    ref_nodes = reference_checkpoint['nodes']
    nodes = checkpoint['nodes']

    #Get RNNs for each checkpoint
    rnn = deepcopy(checkpoint['rnn'])
    ref_rnn = deepcopy((reference_checkpoint['rnn']))

    #Keep track of the *original* indices for each corresponding node pairs
    #in order
    n_nodes = nodes.shape[0]
    n_ref_nodes = ref_nodes.shape[0]
    I_x = []
    I_y = []

    #Get associated output for each set of nodes, align based on this
    ref_outputs = [ref_rnn.output.f(ref_rnn.W_out.dot(ref_node) + ref_rnn.b_out)
                   for ref_node in ref_nodes]
    ref_outputs = np.vstack(ref_outputs)
    outputs = [rnn.output.f(rnn.W_out.dot(node) + rnn.b_out) for node in nodes]
    outputs = np.vstack(outputs)

    #Collect all pairwise distances between ref and target checkpoint outputs
    #at each pair of fixed points.
    D = distance.cdist(ref_outputs, outputs)

    #Keep track of the distances for the corresponding fixed point pairs, both
    #in output and hidden space.
    corr_node_output_distances = []
    corr_node_distances = []

    #Loop through to identify corresponding points until the number of ref
    #nodes reaches the number of target nodes
    while len(I_x) < n_nodes:

        #Identify the next smallest distance between nodes
        d = np.argmin(D)
        d_min = np.min(D)
        x, y = (d // n_nodes), (d % n_nodes)

        #If all nodes have been matched, break
        if d_min == np.inf:
            break
        else:
            corr_node_output_distances.append(d_min)
            corr_node_distances.append(norm(ref_nodes[x] - nodes[y]))

        #Track original indices of ref and target
        I_x.append(x)
        I_y.append(y)

        #Make all distances for both nodes in identified pair infinite so as to
        #not double-count.
        D[x,:] = np.inf
        D[:,y] = np.inf

    #For each reference index (in original order), list the corresponding
    #target index in that same order
    I_ = [I_y[i_x] for i_x in np.argsort(I_x)]

    #Get the "forward" indices, i.e. the same order for ref checkpoint (since
    #order doesn't change for ref checkpoint) but
    #only those that were matched to a target index
    I_f = sorted(I_x)

    #Get the "back" indices, i.e. the indices for target checkpoint ordered
    #according to the ref index order (not sorted). These are identically I_.
    I_b = [i for i in I_]
    extra_indices = list(range(n_nodes))
    [extra_indices.remove(i) for i in I_]
    I = I_ + extra_indices

    keys = ['adjmat_input_{}'.format(i) for i in range(n_inputs)] + ['nodes']

    for key in keys:

        if 'adjmat' in key:
            checkpoint['backshared_' + key] = checkpoint[key][I_b][:,I_b]
            reference_checkpoint['forwardshared_' + key] = reference_checkpoint[key][I_f][:,I_f]

            if n_nodes < n_ref_nodes:
                checkpoint['backembed_' + key] = np.zeros_like(reference_checkpoint[key])
                for i in I_f:
                    checkpoint['backemded_' + key][i] = reference_checkpoint[key][i]
            if n_nodes > n_ref_nodes:
                reference_checkpoint['forwardembed_' + key] = np.zeros_like(checkpoint[key])
                for i in I_b:
                    reference_checkpoint['forwardembed_' + key][i] = checkpoint[key][i]
            checkpoint[key] = checkpoint[key][I][:, I]

        if key == 'nodes':
            checkpoint[key] = checkpoint[key][I]

    checkpoint['corr_node_distances'] = [corr_node_distances[i_x]
                                         for i_x in np.argsort(I_x)]
    checkpoint['corr_node_output_distances'] = [corr_node_output_distances[i_x]
                                                for i_x in np.argsort(I_x)]