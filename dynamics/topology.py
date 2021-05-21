from utils import *
from dynamics.dynamics_utils import *
from core import Simulation
import multiprocessing as mp
from functools import partial
from itertools import product
from copy import deepcopy
from scipy.spatial import distance

def find_KE_minima(checkpoint, test_data, N=1000, verbose_=False,
                   parallelize=False, sigma_pert=0, PCs=None, weak_input=None,
                   context=None, **kwargs):
    """Find many KE minima for a given checkpoint of training. Includes option
    to parallelize or not."""

    test_a = get_test_sim_data(checkpoint, test_data)
    results = []
    initial_states = []

    RNNs = []
    for i in range(N):

        #Set up initial conditions
        rnn = deepcopy(checkpoint['rnn'])
        if context is not None:
            rnn.b_rec += rnn.W_in.dot(context)
        i_a = np.random.randint(test_a.shape[0])
        u_pert = np.random.normal(0, sigma_pert, rnn.n_h)
        if PCs is not None:
            PC_pert = np.random.binomial(0, 0.2, PCs.shape[1]) * sigma_pert
            u_pert = PCs.dot(PC_pert)
        a_init = test_a[i_a] + u_pert
        rnn.reset_network(a=a_init)
        if weak_input is not None:
            x = np.random.binomial(0, 0.2, rnn.n_in) * weak_input
            rnn.next_state(x)
        RNNs.append(rnn)
        initial_states.append(rnn.a.copy())

    if parallelize:

        func_ = partial(find_KE_minimum, **kwargs)
        with mp.Pool(mp.cpu_count()) as pool:
                results = pool.map(func_, RNNs)

    if not parallelize:

        for i in range(N):

            #Report progress
            if i % (N // 10) == 0 and verbose_:
                print('{}% done'.format(i * 10 / N))

            #Select RNN (and starting point)
            rnn = RNNs[i]

            #Calculate final result
            result = find_KE_minimum(rnn, **kwargs)
            results.append(result)

    return results, initial_states

def find_KE_minimum(rnn, LR=1e-2, N_iters=1000000,
                    return_whole_optimization=False,
                    return_period=100,
                    N_KE_increase=3,
                    LR_drop_factor=5,
                    LR_drop_criterion=10,
                    same_LR_criterion=100000,
                    verbose=False,
                    calculate_linearization=False):
    """For a given RNN, performs gradient descent with adaptive learning rate
    to find a kinetic energy  minimum of the network. The seed state is just
    the state of the rnn, rnn.a.


    Returns either ust the final rnn state and KE, or if
    return_whole_optimization is True, then it also returns trajectory of a
    values, norms, and LR_drop times at a frequency specified by
    return_period."""

    #Initialize counters
    i_LR_drop = 0
    i_KE_increase = 0
    i_same_LR = 0

    #Initialize return lists
    a_values = []
    KEs = [rnn.get_network_speed()]
    norms = [norm(rnn.a)]
    LR_drop_times = []

    #Loop once for each iteration
    for i_iter in range(N_iters):

        #Report progress
        if i_iter % (N_iters//10) == 0:
            pct_complete = np.round(i_iter/N_iters*100, 2)
            if verbose:
                print('{}% done'.format(pct_complete))

        rnn.a -= LR * rnn.get_network_speed_gradient()
        a_values.append(rnn.a.copy())

        KEs.append(rnn.get_network_speed())
        norms.append(norm(rnn.a))

        #Stop optimization if KE increases too many steps in a row
        if KEs[i_iter] > KEs[i_iter - 1]:
            i_KE_increase += 1
            if i_KE_increase >= N_KE_increase:
                LR /= LR_drop_factor
                if verbose:
                    print('LR drop #{} at iter {}'.format(i_LR_drop, i_iter))
                LR_drop_times.append(i_iter)
                i_LR_drop += 1
                i_KE_increase = 0
                if i_LR_drop >= LR_drop_criterion:
                    if verbose:
                        print('Reached criterion at {} iter'.format(i_iter))
                    break
        else:
            i_KE_increase = 0
            i_same_LR += 1

        if i_same_LR >= same_LR_criterion:
            print('Reached same LR criterion at {} iter'.format(i_iter))
            break


    results = {'a_final': a_values[-1],
               'KE_final': KEs[-1]}

    if calculate_linearization:

        rnn.a = results['a_final']
        a_J = rnn.get_a_jacobian(update=False)
        eigs, _ = np.linalg.eig(a_J)
        results['jacobian_eigs'] = eigs

    if return_whole_optimization:
        results['a_trajectory'] = np.array(a_values[::return_period])
        results['norms'] = np.array(norms[::return_period])
        results['KEs'] = np.array(KEs[::return_period])
        results['LR_drop_times'] = LR_drop_times

    return results

def run_autonomous_sim(a_initial, rnn, N, monitors=[],
                       return_final_state=False, input_pulse=None,
                       background_input=0, sigma=0):
    """Creates and runs a test simulation with no inputs and a specified
    initial state of the network."""

    #Create empty data array
    data = {'test': {'X': np.zeros((N, rnn.n_in)) + background_input,
                     'Y': np.zeros((N, rnn.n_out))}}

    rnn.reset_network(a=a_initial)

    if input_pulse is not None:
        x = input_pulse + background_input
        rnn.next_state(x=x)

    sim = Simulation(rnn)
    sim.run(data, mode='test', monitors=monitors,
            a_initial=rnn.a,
            verbose=False,
            sigma=sigma)

    if return_final_state:
        return sim.rnn.a.copy()
    else:
        return sim

def get_graph_structure(checkpoint, N=100, time_steps=50, epsilon=0.01,
                        parallelize=True, key='adjacency_matrix',
                        input_pulse=None, background_input=0, nodes=None,
                        sigma=0):
    """For each fixed point cluster, runs an autonomous simulation with
    initial condition in small small neighborhood of a point and evaluates
    where it ends up."""

    if nodes is None:
        cluster_means = checkpoint['cluster_means']
    else:
        cluster_means = nodes
    n_clusters = cluster_means.shape[0]
    adjacency_matrix = np.zeros((n_clusters, n_clusters))
    rnn = checkpoint['rnn']


    if parallelize:
        for i in range(n_clusters):
            a_init = [(cluster_means[i] +
                      np.random.normal(0, epsilon, rnn.n_h))
                      for _ in range(N)]
            func_ = partial(run_autonomous_sim, rnn=rnn, N=time_steps,
                            monitors=[], return_final_state=True,
                            input_pulse=input_pulse,
                            background_input=background_input,
                            sigma=sigma)
            #set_trace()
            with mp.Pool(mp.cpu_count()) as pool:
                final_states = pool.map(func_, a_init)

            final_states = np.array(final_states)

            distances = distance.cdist(cluster_means, final_states)
            i_clusters = np.argmin(distances, axis=0)
            bins = list(np.arange(-0.5, n_clusters, 1))
            transition_probs, _ = np.histogram(i_clusters,
                                               bins=bins,
                                               density=True)
            #set_trace()
            adjacency_matrix[i] = transition_probs

    if not parallelize:
        for i in range(n_clusters):
            a_init = [(cluster_means[i] +
                      np.random.normal(0, epsilon, rnn.n_h))
                      for _ in range(N)]
            func_ = partial(run_autonomous_sim, rnn=rnn, N=time_steps,
                            monitors=[], return_final_state=True,
                            input_pulse=input_pulse,
                            background_input=background_input,
                            sigma=sigma)
            #set_trace()
            final_states = []
            for a_init_ in a_init:
                final_states.append(func_(a_init_))

            final_states = np.array(final_states)

            distances = distance.cdist(cluster_means, final_states)
            i_clusters = np.argmin(distances, axis=0)
            bins = list(np.arange(-0.5, n_clusters, 1))
            transition_probs, _ = np.histogram(i_clusters,
                                               bins=bins,
                                               density=True)
            #set_trace()
            adjacency_matrix[i] = transition_probs

    checkpoint[key] = adjacency_matrix

def get_input_dependent_graph_structure(checkpoint, inputs, contexts=None,
                                        parallelize=False,
                                        nodes_matrix=None,
                                        node_thresh=0.05,
                                        sigma=0):
    """After running get_graph_structure, this can be used to find input-
    dependent transition probabilities between stable nodes.

    Args:
        inputs: A list of input vectors to be sampled iid
    """

    M = checkpoint['adjacency_matrix']
    centroids = checkpoint['cluster_means']
    nodes = centroids[np.where(M.sum(0) > node_thresh)]
    checkpoint['nodes'] = nodes

    if contexts is not None:
        conds = product(inputs, contexts)
    else:
        conds = inputs

    for i_x, x in enumerate(conds):

        if contexts is None:
            key = 'adjmat_input_{}'.format(i_x)

            adjacency_matrix = get_graph_structure(checkpoint,
                                                   parallelize=parallelize,
                                                   key=key,
                                                   input_pulse=x,
                                                   nodes=nodes,
                                                   sigma=sigma)
        else:
            key = 'adjmat_input_{}_context_{}'.format(i_x // len(inputs),
                                                      i_x % len(inputs))

            adjacency_matrix = get_graph_structure(checkpoint,
                                                   parallelize=parallelize,
                                                   key=key,
                                                   input_pulse=x[0],
                                                   background_input=x[1],
                                                   nodes=nodes,
                                                   sigma=sigma)
