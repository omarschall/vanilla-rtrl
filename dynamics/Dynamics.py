from sklearn.cluster import DBSCAN
from dynamics.topology import *

def analyze_all_checkpoints(checkpoints, func, test_data, **kwargs):
    """For a given analysis function and a list of checkpoints, applies
    the function to each checkpoint in the list and returns all results.
    Uses multiprocessing to apply the analysis independently."""

    func_ = partial(func, test_data=test_data, **kwargs)
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(func_, checkpoints)

    return results

def analyze_checkpoint(checkpoint, data, N_iters=8000,
                       same_LR_criterion=5000, N=200,
                       n_PCs=3, context=None, KE_criterion=None,
                       reference_checkpoint=None,
                       sigma=0,
                       DB_eps=0.5,
                       **kwargs):

    print('Analyzing checkpoint {}...'.format(checkpoint['i_t']))

    rnn = checkpoint['rnn']
    test_sim = Simulation(rnn)
    test_sim.run(data,
                  mode='test',
                  monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
                  verbose=False,
                  sigma=sigma)

    transform = Vanilla_PCA(checkpoint, data, n_PCs=n_PCs, sigma=sigma)
    V = transform(np.eye(rnn.n_h))

    fixed_points, initial_states = find_KE_minima(checkpoint, data, N=N,
                                                  N_iters=N_iters, LR=10,
                                                  same_LR_criterion=same_LR_criterion,
                                                  context=context,
                                                  **kwargs)

    A = np.array([d['a_final'] for d in fixed_points])
    A_init = np.array(initial_states)
    KE = np.array([d['KE_final'] for d in fixed_points])

    if KE_criterion is not None:
        idx = np.where(KE < KE_criterion)
        A_KE_crit_pass = A[idx]
        A_init_KE_crit_pass = A_init[idx]
        KE_crit_pass = KE[idx]

        while A_KE_crit_pass.size == 0:
            KE_criterion *= 3
            idx = np.where(KE < KE_criterion)
            A_KE_crit_pass = A[idx]
            A_init_KE_crit_pass = A_init[idx]
            KE_crit_pass = KE[idx]

        A = A_KE_crit_pass
        A_init = A_init_KE_crit_pass
        KE = KE_crit_pass

    dbscan = DBSCAN(eps=DB_eps)
    dbscan.fit(A)
    dbscan.labels_

    cluster_idx = np.unique(dbscan.labels_)
    n_clusters = len(cluster_idx) - (-1 in cluster_idx)
    if n_clusters == 0:
        n_clusters_ = 1
    else:
        n_clusters_ = n_clusters
    cluster_means = np.zeros((n_clusters_, rnn.n_h))
    for i in cluster_idx:

        if i == -1 and n_clusters > 0:
            continue
        else:
            cluster_means[i] = A[dbscan.labels_ == i].mean(0)

    cluster_eigs = []
    cluster_KEs = []
    for cluster_mean in cluster_means:
        checkpoint['rnn'].reset_network(a=cluster_mean)
        a_J = checkpoint['rnn'].get_a_jacobian(update=False)
        cluster_eigs.append(np.abs(np.linalg.eig(a_J)[0][0]))
        cluster_KE = checkpoint['rnn'].get_network_speed()
        cluster_KEs.append(cluster_KE)
    cluster_eigs = np.array(cluster_eigs)
    cluster_KEs = np.array(cluster_KEs)

    #Save results
    checkpoint['fixed_points'] = A
    checkpoint['KE'] = KE
    checkpoint['KE_criterion'] = KE_criterion
    checkpoint['cluster_means'] = cluster_means
    checkpoint['cluster_labels'] = dbscan.labels_
    checkpoint['V'] = V
    checkpoint['A_init'] = A_init
    checkpoint['cluster_eigs'] = cluster_eigs
    checkpoint['cluster_KEs'] = cluster_KEs
    checkpoint['test_loss'] = test_sim.mons['rnn.loss_'].mean()

    if reference_checkpoint is not None:
        align_checkpoints(reference_checkpoint, checkpoint)

### --- ANALYSIS METHODS --- ###

def Vanilla_PCA(checkpoint, test_data, n_PCs=3, sigma=0):
    """Return first n_PCs PC axes of the test """

    test_a = get_test_sim_data(checkpoint, test_data, sigma=sigma)
    U, S, VT = np.linalg.svd(test_a)

    checkpoint['participation_coef'] = np.square(S.sum()) / np.square(S).sum()

    transform = partial(np.dot, b=VT.T[:,:n_PCs])

    return transform

def UMAP_(checkpoint, test_data, n_components=3, **kwargs):
    """Performs  UMAP with default parameters and returns component axes."""

    test_a = get_test_sim_data(checkpoint, test_data)
    fit = umap.UMAP(n_components=n_components, **kwargs)
    u = fit.fit_transform(test_a)

    return fit.transform


