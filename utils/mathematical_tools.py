import numpy as np
from scipy.stats import unitary_group
from math import floor, ceil
from scipy.sparse import csc_matrix

def norm(z):
    """Computes the L2 norm of a numpy array."""

    return np.sqrt(np.sum(np.square(z)))

def clip_norm(z, max_norm=1.0):
    """Clips the norm of an array"""

    if norm(z) > max_norm:
        z = z * (max_norm / norm(z))

    return z


def rectangular_filter(signal, filter_size=100):
    """Convolves a given signal with a rectangular filter in 'valid' mode

    Args:
        signal (numpy array): An 1-dimensional array specifying the signal.
        filter_size (int): An integer specifcying the width of the rectangular
            filter used for the convolution."""

    return np.convolve(signal, np.ones(filter_size)/filter_size, mode='valid')

def classification_accuracy(data, y_hat):
    """Calculates the fraction of test data whose argmax matches that of
    the prediction."""

    y_hat = np.array(y_hat)

    i_label = np.argmax(data['test']['Y'], axis=1)
    i_pred = np.argmax(y_hat, axis=1)

    acc = np.sum(i_label == i_pred) / len(i_label)

    return acc

def normalized_dot_product(a, b):
    """Calculates the normalized dot product between two numpy arrays, after
    flattening them."""

    a_norm = norm(a)
    b_norm = norm(b)

    if a_norm > 0 and b_norm > 0:
        return np.dot(a.flatten(), b.flatten())/(a_norm * b_norm)
    else:
        return 0

def half_normalized_dot_product(a, b):
    """Calculates the projection of b onto the unit vector defined by a,
    after flattening."""

    a_norm = norm(a)

    if a_norm >0:
        return np.dot(a.flatten(),b.flatten())/(a_norm)
    else:
        return 0

def get_spectral_radius(M):
    """Calculates the spectral radius of a matrix."""

    eigs, _ = np.linalg.eig(M)

    return np.amax(np.absolute(eigs))

def generate_real_matrix_with_given_eigenvalues(evals):
    """For a given set of complex eigenvalues, generates a real matrix with
    those eigenvalues.

    More precisely, the user should specify *half* of the eigenvalues (since
    the other half must be complex conjugates). Thus the dimension should be
    even for conveneince.

    Args:
        evals (numpy array): An array of shape (n_half), where n_half is half
            the dimensionality of the matrix to be generated, that specifies
            the desired eigenvalues.

    Returns:
        A real matrix, half of whose eigenvalues are evals.

    CAVEAT: DOESN'T WORK."""

    n_half = len(evals)
    evals = np.concatenate([evals, np.conjugate(evals)])

    evecs = unitary_group.rvs(2 * n_half)[:, :n_half]
    evecs = np.concatenate([evecs, np.conjugate(evecs)], axis=1)

    M = evecs.dot(np.diag(evals)).dot(evecs)

    return np.real(M)

def weighted_median(hist, bin_centers):
    """Given the result of a weighted histogram, calculates the weighted
    median."""

    hist_cdf = np.cumsum(hist)/hist.sum()
    return bin_centers[np.where(hist_cdf >= 0.5)[0][0]]

def triangular_integer_decomposition(idx):
    """For a given integer idx, finds the maximal triangular number less
    than idx and also returns remainder."""

    low_n = floor(np.maximum(0, np.sqrt(idx * 2) - 2))
    high_n = ceil(np.sqrt(idx * 2) + 2)

    n_range = list(range(low_n, high_n + 1))
    triangular_numbers = np.array([n*(n+1)/2 for n in n_range])

    n = n_range[np.where(triangular_numbers > idx)[0][0] - 1]

    return n, idx - n * (n + 1) / 2

### FROM CHAT GPT

def upper_off_diagonal(sparse_matrix):
    # Ensure the matrix is in CSC format
    csc = sparse_matrix.tocsc()

    # Find the columns that have non-zero entries
    cols = np.repeat(np.arange(csc.shape[1]), np.diff(csc.indptr))

    # Determine the rows for the upper off-diagonal
    rows = cols - 1

    # Use fancy indexing to extract the elements
    off_diag_elems = csc[rows, cols].A1

    # Filter out elements that are not part of the upper off-diagonal
    mask = (rows >= 0) & (rows < csc.shape[0] - 1)

    return off_diag_elems[mask]
