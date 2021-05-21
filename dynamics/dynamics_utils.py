import numpy as np
from core import Simulation
from scipy.spatial import distance
import copy
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

    shape_1 = copy.copy(nodes.shape)

    n_nodes = nodes.shape[0]
    n_ref_nodes = ref_nodes.shape[0]
    n_shared_max = min(n_nodes, n_ref_nodes)

    # 'cluster means'
    # 'cluster_eigs'
    # 'cluster_KEs'

    I_x = []
    I_y = []
    D = distance.cdist(ref_nodes, nodes)
    corr_node_distances = []
    while len(I_x) < n_nodes:

        #set_trace()

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
    # if checkpoint['nodes'].shape != shape_1:
    #     print('hey now', checkpoint['i_t'])
    #     set_trace()

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
