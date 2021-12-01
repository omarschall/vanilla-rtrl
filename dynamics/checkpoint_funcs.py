import numpy as np

def get_checkpoint_loss(checkpoint):
    """Gets checkpoint loss"""

    return checkpoint['test_loss']

def get_checkpoint_spectral_radius(checkpoint):
    """Gets checkpoint spectral radius"""

    eigs, _ = np.linalg.eig(checkpoint['rnn'].W_rec)

    return np.amax(np.absolute(eigs))

def get_checkpoint_2nd_eigenvalue(checkpoint):
    """Gets checkpoint spectral radius"""

    eigs, _ = np.linalg.eig(checkpoint['rnn'].W_rec)
    eigs_sorted = sorted(np.absolute(eigs))[::-1]

    return eigs_sorted[1]

def get_checkpoint_3nd_eigenvalue(checkpoint):
    """Gets checkpoint spectral radius"""

    eigs, _ = np.linalg.eig(checkpoint['rnn'].W_rec)
    eigs_sorted = sorted(np.absolute(eigs))[::-1]

    return eigs_sorted[2]

def get_checkpoint_weight_std(checkpoint, W_init=None):
    """Gets checkpoint spectral radius"""

    if W_init is not None:
        W_0 = W_init
    else:
        W_0 = np.zeros_like(checkpoint['rnn'].W_rec)

    W = checkpoint['rnn'].W_rec - W_0

    return W.std()

def get_checkpoint_weight_mean(checkpoint, W_init=None):
    """Gets checkpoint spectral radius"""

    if W_init is not None:
        W_0 = W_init
    else:
        W_0 = np.zeros_like(checkpoint['rnn'].W_rec)

    W = checkpoint['rnn'].W_rec - W_0

    return W.mean()

def get_checkpoint_n_stable_FPs(checkpoint):
    """Gets the number of stable checkpoint nodes after topological analysis."""

    return checkpoint['nodes'].shape[0]

def get_checkpoint_n_unstable_FPs(checkpoint):
    """Gets the number of unstable fixed points."""

    return checkpoint['cluster_means'].shape[0] - checkpoint['nodes'].shape[0]

def get_checkpoint_W_rec_dim(checkpoint, W_init=None):
    """Gets the latent dimensionality of the network as measured by
    participation coefficient of the recurrent weights themselves."""

    if W_init is not None:
        W_0 = W_init
    else:
        W_0 = np.zeros_like(checkpoint['rnn'].W_rec)

    W = checkpoint['rnn'].W_rec - W_0

    eigs, _ = np.linalg.eig(W)

    abs_eigs = np.abs(eigs)

    if abs_eigs.sum() == 0:
        participation_coef = 0
    else:
        participation_coef = np.square(np.sqrt(abs_eigs).sum()) / abs_eigs.sum()

    return participation_coef

def get_checkpoint_participation_coefficient(checkpoint):
    """Gets the participation coefficient of the network test activity."""

    return checkpoint['participation_coef']

def get_checkpoint_performance_based_dimensionality(checkpoint):

    pass