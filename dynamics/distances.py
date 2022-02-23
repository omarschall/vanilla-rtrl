from dynamics.dynamics_utils import get_test_sim_data, align_checkpoints_based_on_output
from dynamics.Dynamics import Vanilla_PCA
from itertools import permutations
import numpy as np
from netcomp.distance import netsimile
from pyemd import emd
from sklearn.cross_decomposition import CCA
from utils import norm, normalized_dot_product
#from dynamics.VAE import test_vae

def wasserstein_distance(checkpoint_1, checkpoint_2):
    """Calculates the Wassterstein ("Earth Mover's") distance between the
    fixed points of 2 different checkpoints.

    Checkpoints must be separately analyzed to have fixed points and cluster
    means computed."""

    cluster_means_1 = checkpoint_1['cluster_means']
    cluster_labels_1 = checkpoint_1['cluster_labels']
    cluster_weights_1 = []
    for j in range(cluster_means_1.shape[0]):
        cluster_weights_1.append(len(np.where(cluster_labels_1 == j)[0]))
    cluster_weights_1 = np.array(cluster_weights_1)

    cluster_means_2 = checkpoint_2['cluster_means']
    cluster_labels_2 = checkpoint_2['cluster_labels']
    cluster_weights_2 = []
    for j in range(cluster_means_2.shape[0]):
        cluster_weights_2.append(len(np.where(cluster_labels_2 == j)[0]))
    cluster_weights_2 = np.array(cluster_weights_2)

    hist1 = np.concatenate([cluster_weights_1,
                            np.zeros_like(cluster_weights_2)], axis=0).astype(np.float64)
    hist2 = np.concatenate([np.zeros_like(cluster_weights_1),
                            cluster_weights_2], axis=0).astype(np.float64)
    N = len(cluster_weights_1) + len(cluster_weights_2)

    combined_means = np.concatenate([cluster_means_1, cluster_means_2], axis=0)

    distances = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            distances[i, j] = norm(combined_means[i] - combined_means[j])

    return emd(hist1, hist2, distances)

# def SVCCA_distance(checkpoint_1, checkpoint_2, data, R=3):
#     """Compute the singular-value canonical correlation analysis distance
#     between two different networks."""
#
#     A_1 = get_test_sim_data(checkpoint_1, data)
#     A_2 = get_test_sim_data(checkpoint_2, data)
#
#     #U_1, S_1, V_1 = np.linalg.svd(A_1)
#     #U_2, S_2, V_2 = np.linalg.svd(A_2)
#
#     cca = CCA(n_components=R, max_iter=1000)
#     #cca.fit(V_1, V_2)
#     #cca.fit(A_1.dot(V_1), A_2.dot(V_2))
#     cca.fit(A_1, A_2)
#
#     #return 1 - cca.score(A_1.dot(V_1), A_2.dot(V_2))
#     #return 1 - cca.score(V_1, V_2)
#    return 1 - cca.score(A_1, A_2)

def SVCCA_distance(checkpoint_1, checkpoint_2, R=32):
    """Compute the singular-value canonical correlation analysis distance
    between two different networks."""

    A_1 = checkpoint_1['test_data']
    A_2 = checkpoint_2['test_data']

    #U_1, S_1, V_1 = np.linalg.svd(A_1)
    #U_2, S_2, V_2 = np.linalg.svd(A_2)

    cca = CCA(n_components=R, max_iter=1000)
    #cca.fit(V_1, V_2)
    #cca.fit(A_1.dot(V_1), A_2.dot(V_2))
    cca.fit(A_1, A_2)

    #return 1 - cca.score(A_1.dot(V_1), A_2.dot(V_2))
    #return 1 - cca.score(V_1, V_2)
    return 1 - cca.score(A_1, A_2)

def CKA_distance(checkpoint_1, checkpoint_2, data=None, centered=True):
    """Compute CKA distance between two checkpoints"""

    try:
        A_1 = checkpoint_1['test_data']
        A_2 = checkpoint_2['test_data']
    except KeyError:
        A_1 = get_test_sim_data(checkpoint_1, data)
        A_2 = get_test_sim_data(checkpoint_2, data)

    N = A_1.shape[0]

    if centered:
        A_1 = A_1 - np.mean(A_1, axis=0)
        A_2 = A_2 - np.mean(A_2, axis=0)


    return 1 - (norm(A_1.T.dot(A_2))**2 / (norm(A_1.T.dot(A_1)) * norm(A_2.T.dot(A_2))))


def rec_weight_distance(checkpoint_1, checkpoint_2):
    """Computes the norm of the difference in the recurrent weight matrix
    between two checkpoints."""

    rnn_1 = checkpoint_1['rnn']
    rnn_2 = checkpoint_2['rnn']

    W_rec_1 = np.hstack([rnn_1.W_rec, rnn_1.W_in, rnn_1.b_rec.reshape(-1, 1)])
    W_rec_2 = np.hstack([rnn_2.W_rec, rnn_2.W_in, rnn_2.b_rec.reshape(-1, 1)])

    return norm(W_rec_1 - W_rec_2)

def output_weight_distance(checkpoint_1, checkpoint_2):
    """Computes the norm of the difference in the output weight matrix
    between two checkpoints."""

    rnn_1 = checkpoint_1['rnn']
    rnn_2 = checkpoint_2['rnn']

    W_out_1 = np.hstack([rnn_1.W_out, rnn_1.b_out.reshape(-1, 1)])
    W_out_2 = np.hstack([rnn_2.W_out, rnn_2.b_out.reshape(-1, 1)])

    return norm(W_out_1 - W_out_2)

def graph_distance(checkpoint_1, checkpoint_2):
    """Calculates the netsimile distance between the adjacency matrix for
    the fixed-point transition graphs."""

    return netsimile(checkpoint_1['adjacency_matrix'],
                     checkpoint_2['adjacency_matrix'])

def input_dependent_graph_distance(checkpoint_1, checkpoint_2):
    """Calculates the average netsimile distance between the adjacency
    matrices for each input pattern."""

    diffs = []
    keys = ['adjmat_input_{}'.format(i) for i in range(6)]
    for key in keys:
        diffs.append(netsimile(checkpoint_1[key],
                               checkpoint_2[key]))
    return sum(diffs)

    # return sum([netsimile(checkpoint_1['adjmat_input_{}'.format(i_x)],
    #                checkpoint_2['adjmat_input_{}'.format(i_x)])
    #             for i_x in range(6)]) / 6


def PC_distance_1(checkpoint_1, checkpoint_2):
    """Returns the inverted, normalized alignment between the first n_dim PC
    axes as calculated during analysis pipeline.

    The result is in the interval [0, 2] with 2 being maximally dissimilar."""

    n_dim = checkpoint_1['V'].shape[1]

    V1 = checkpoint_1['V']
    V2 = checkpoint_2['V']

    return 1 - np.abs(np.sum(V1 * V2) / n_dim)

def PC_distance_2(checkpoint_1, checkpoint_2):
    """Returns the angular alignment between the first n_dim PC
    axes as calculated during analysis pipeline, computed using determinants.

    The result is in the interval [0, pi]."""

    V1 = checkpoint_1['V']
    V2 = checkpoint_2['V']

    M = V1.T.dot(V2)

    return np.arccos(np.sqrt(np.linalg.det(M.dot(M.T))))

def PC_distance_3(checkpoint_1, checkpoint_2, N_avg=None, N_test=2000,
                  task=None):
    """Measure distance between first 3 PCs of two checkpoints by searching
    for the best pairwise alignments and averaging them."""

    n_dim = checkpoint_1['V'].shape[1]

    if N_avg is None:

        V1 = checkpoint_1['V']
        V2 = checkpoint_2['V']

        return 1 - np.sort(np.abs(V1.T.dot(V2)).flatten())[-n_dim:].sum() / n_dim

    else:

        n_1 = checkpoint_1['rnn'].n_h
        n_2 = checkpoint_2['rnn'].n_h


        Ds = []
        for i in range(N_avg):
            np.random.seed(i)
            test_data = task.gen_data(0, N_test)
            transform = Vanilla_PCA(checkpoint_1, test_data)
            V1 = transform(np.eye(n_1))
            transform = Vanilla_PCA(checkpoint_2, test_data)
            V2 = transform(np.eye(n_2))
            avg = np.sort(np.abs(V1.T.dot(V2)).flatten())[-n_dim:].sum() / n_dim
            Ds.append(1 - avg)

        return np.mean(Ds)



# def VAE_distance(checkpoint_1, checkpoint_2, big_data):
#     """Returns the reconstruction error of trajectories sampled from the RNN
#     of checkpoint_2 by the VAE trained on the RNN of checkpoint_1."""
#
#     return test_vae(model_checkpoint=checkpoint_1, data=big_data,
#                     test_checkpoint=checkpoint_2)

def aligned_graph_distance(checkpoint_1, checkpoint_2, node_diff_penalty=1,
                           n_inputs=6, minimize_over_permutations=False):
    """Returns the custom graph similarity metric for aligned nodes.

    We assume checkpoint_2 *follows* checkpoint_1, i.e. checkpoint_1 was used
    to align the nodes of checkpoint_2.
    """

    if not minimize_over_permutations:
        align_checkpoints_based_on_output(checkpoint_2, checkpoint_1,
                                          n_inputs=n_inputs)

        n_1 = checkpoint_1['nodes'].shape[0]
        n_2 = checkpoint_2['nodes'].shape[0]

        if n_1 == n_2:
            base_key_1 = 'forwardshared_adjmat_input'
            base_key_2 = 'backshared_adjmat_input'
        if n_1 > n_2:
            base_key_1 = 'adjmat_input'
            base_key_2 = 'backembed_adjmat_input'
        if n_1 < n_2:
            base_key_1 = 'forwardembed_adjmat_input'
            base_key_2 = 'adjmat_input'


        adjmats_1 = [checkpoint_1[base_key_1 + '_{}'.format(i)] for i in range(n_inputs)]
        adjmats_2 = [checkpoint_2[base_key_2 + '_{}'.format(i)] for i in range(n_inputs)]

        ret = 1 - sum([normalized_dot_product(M1, M2) for M1, M2 in zip(adjmats_1, adjmats_2)]) / n_inputs

        ret = ret + node_diff_penalty * np.abs(n_1 - n_2) / max(n_1, n_2)

        return ret
    else:
        n_out = checkpoint_1['rnn'].n_out
        distances = []
        perms = list(permutations(range(n_out)))
        for perm in perms:
            align_checkpoints_based_on_output(checkpoint_2, checkpoint_1,
                                              n_inputs=n_inputs,
                                              output_perm=perm)

            n_1 = checkpoint_1['nodes'].shape[0]
            n_2 = checkpoint_2['nodes'].shape[0]

            if n_1 == n_2:
                base_key_1 = 'forwardshared_adjmat_input'
                base_key_2 = 'backshared_adjmat_input'
            if n_1 > n_2:
                base_key_1 = 'adjmat_input'
                base_key_2 = 'backembed_adjmat_input'
            if n_1 < n_2:
                base_key_1 = 'forwardembed_adjmat_input'
                base_key_2 = 'adjmat_input'

            adjmats_1 = [checkpoint_1[base_key_1 + '_{}'.format(i)] for i in range(n_inputs)]
            adjmats_2 = [checkpoint_2[base_key_2 + '_{}'.format(i)] for i in range(n_inputs)]

            distance = 1 - sum([normalized_dot_product(M1, M2)
                                for M1, M2 in zip(adjmats_1, adjmats_2)]) / n_inputs

            distance = distance + node_diff_penalty * np.abs(n_1 - n_2) / max(n_1, n_2)
            distances.append(distance)

        i_min_perm = np.argmin(distances)
        min_perm = perms[i_min_perm]

        checkpoint_2['backalign_output_perm'] = min_perm
        align_checkpoints_based_on_output(checkpoint_2, checkpoint_1,
                                          n_inputs=n_inputs,
                                          output_perm=min_perm)

        return distances[i_min_perm]


def node_diff_distance(checkpoint_1, checkpoint_2):
    """Returns the node_diff

    We assume checkpoint_2 *follows* checkpoint_1, i.e. checkpoint_1 was used
    to align the nodes of checkpoint_2.
    """

    n_1 = checkpoint_1['nodes'].shape[0]
    n_2 = checkpoint_2['nodes'].shape[0]

    return np.abs(n_1 - n_2) / max(n_1, n_2)

def weight_change_alignment_distance(checkpoint_1, checkpoint_2):
    """Returns the angular alignment between weight updates relative to each
    checkpoint's most recent weight update.

    NOTE: To work, you must have "checkpoint_optimizer" flag set to True when
    running simulation object."""

    opt_1 = checkpoint_1['optimizer']
    opt_2 = checkpoint_2['optimizer']
    dW_1 = np.concatenate([v.flatten() for v in opt_1.vel])
    dW_2 = np.concatenate([v.flatten() for v in opt_2.vel])

    return normalized_dot_product(dW_1, dW_2)