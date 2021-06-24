import numpy as np

def get_checkpoint_loss(checkpoint):
    """Gets checkpoint loss"""

    return checkpoint['test_loss']

def get_checkpoint_spectral_radius(checkpoint):
    """Gets checkpoint spectral radius"""

    eigs, _ = np.linalg.eig(checkpoint['rnn'].W_rec)

    return np.amax(np.absolute(eigs))

def get_checkpoint_weight_std(checkpoint):
    """Gets checkpoint spectral radius"""

    return checkpoint['W_rec'].std()

def get_checkpoint_weight_mean(checkpoint):
    """Gets checkpoint spectral radius"""

    return checkpoint['W_rec'].mean()

