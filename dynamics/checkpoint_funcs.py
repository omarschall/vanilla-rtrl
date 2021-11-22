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

def get_checkpoint_weight_std(checkpoint):
    """Gets checkpoint spectral radius"""

    return checkpoint['rnn'].W_rec.std()

def get_checkpoint_weight_mean(checkpoint):
    """Gets checkpoint spectral radius"""

    return checkpoint['rnn'].W_rec.mean()

def get_checkpoint_n_stable_FPs(checkpoint):
    """Gets the number of stable checkpoint nodes after topological analysis."""

    return checkpoint['nodes'].shape[0]

def get_checkpoint_n_unstable_FPs(checkpoint):
    """Gets the number of unstable fixed points."""

    return checkpoint['cluster_means'].shape[0] - checkpoint['nodes'].shape[0]

def get_checkpoint_W_rec_dim(checkpoint):
    """Gets the latent dimensionality of the network as measured by
    participation coefficient of the recurrent weights themselves."""

    W_rec = checkpoint['rnn'].W_rec

    eigs, _ = np.linalg.eig(W_rec)

    abs_eigs = np.abs(eigs)

    return np.square(np.sqrt(abs_eigs).sum()) / abs_eigs.sum()

def get_checkpoint_participation_coefficient(checkpoint):
    """Gets the participation coefficient of the network test activity."""

    return checkpoint['participation_coef']