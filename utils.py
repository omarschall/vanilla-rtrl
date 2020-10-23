#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 12:08:30 2019

@author: omarschall
"""

import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy.stats import unitary_group
from scipy.signal import decimate
from functools import reduce
from pdb import set_trace
from scipy.ndimage.filters import uniform_filter1d

### --- Mathematical tools --- ###

def norm(z):
    """Computes the L2 norm of a numpy array."""

    return np.sqrt(np.sum(np.square(z)))

def clip_norm(z, max_norm=1.0):
    """Clips the norm of an array"""
    
    if norm(z) > max_norm:
        z = z * (max_norm / norm(z))
    
    return z

def split_weight_matrix(A, sizes, axis=1):
    """Splits a weight matrix along the specified axis (0 for row, 1 for
    column) into a list of sub arrays of size specified by 'sizes'."""

    idx = [0] + np.cumsum(sizes).tolist()
    if axis == 1:
        ret = [np.squeeze(A[:,idx[i]:idx[i+1]]) for i in range(len(idx) - 1)]
    elif axis == 0:
        ret = [np.squeeze(A[idx[i]:idx[i+1],:]) for i in range(len(idx) - 1)]
    return ret

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

### --- Plotting tools --- ###

def plot_eigenvalues(*matrices, fig=None, return_fig=False):
    """Plots eigenvalues of a given matrix in the complex plane, as well
    as the unit circle for reference."""

    if fig is None:
        fig = plt.figure()
    theta = np.arange(0, 2*np.pi, 0.01)
    plt.plot(np.cos(theta), np.sin(theta), 'k', linestyle='--', linewidth=0.3)
    plt.axis('equal')

    for M in matrices:
        eigs, _ = np.linalg.eig(M)
        plt.plot(np.real(eigs), np.imag(eigs), '.')

    if return_fig:
        return fig

def plot_array_of_histograms(counts, weights, ticks=None, return_fig=True,
                             plot_zero_line=True, fig_size=(6,3), **kwargs):
    """Plots count data in the shape (n_samples, n_row, n_col) as an array
    of histograms with n_row rows and n_col cols."""

    fig, ax = plt.subplots(counts.shape[1], counts.shape[2],
                           figsize=fig_size)

    n_bins = 100
    if 'n_bins' in kwargs.keys():
        n_bins = kwargs['n_bins']

    weighted_medians = np.zeros((counts.shape[1], counts.shape[2]))

    for i in range(counts.shape[1]):
        for j in range(counts.shape[2]):
            if i <= j:
                fig.delaxes(ax[i, j])
                continue
            w = weights[:, i, j]
            hist, bins = np.histogram(counts[:, i, j],
                                      weights=w/w.sum(),
                                      bins=np.linspace(-1, 1, n_bins),
                                      density=True)
            bin_centers = bins[:-1] + (bins[1] - bins[0])/2
            #weighted_medians[i, j] = weighted_median(hist, bin_centers)
            weighted_medians[i, j] = np.mean(counts[:, i, j])
            hist = uniform_filter1d(hist, 10)
            #ax[i, j].hist(counts[:, i, j], bins=np.linspace(-1, 1, n_bins))
            ax[i, j].plot(bin_centers, hist, linewidth=0.7)
            ax[i, j].fill_between(bin_centers,
                                  np.zeros_like(bin_centers),
                                  hist,
                                  color='C0',
                                  alpha=0.3)
            ax[i, j].axvline(x=weighted_medians[i, j], color='C0',
                             linestyle='--', dashes=[1.5,0.75], linewidth=0.7)
            #ax[i, j].set_ylim([0, 2])
            if plot_zero_line:
                ax[i, j].axvline(x=0, color='k', linestyle='--',
                                 dashes=[1.5,0.75], linewidth=0.7)
            ax[i, j].set_yticks([])
            if ticks is not None:
                if i == counts.shape[1] - 1:
                    pass
                    ax[i, j].set_xticks([])
                    #ax[i, j].set_xlabel(ticks[j])
                else:
                    ax[i, j].set_xticks([])
                if j == 0:
                    pass
                    #ax[i, j].set_ylabel(ticks[i])

    if return_fig:
        return (fig, weighted_medians)

def plot_array_of_downsampled_signals(signals, ticks=None, return_fig=False,
                                      plot_zero_line=True, **kwargs):
    """Plots time series data in the shape (T, n_row, n_col) as an array
    of filtered signals with n_row rows and n_col cols."""

    fig, ax = plt.subplots(signals.shape[1], signals.shape[2],
                           figsize=(30, 10))

    n_bins = 100
    if 'n_bins' in kwargs.keys():
        n_bins = kwargs['n_bins']

    for i in range(signals.shape[1]):
        for j in range(signals.shape[2]):
            if i < j:
                fig.delaxes(ax[i, j])
                continue
            signal = decimate(decimate(decimate(signals[:, i, j], 10), 10), 10)
            #time_ = np.arange(0, signals.shape[0], 1000)
            #i_time_ticks = [len(time_)//5*i for i in range(5)]
            #time_ticks = time_[np.array(i_time_ticks)]
            ax[i, j].plot(signal)
            ax[i, j].set_ylim([-1, 1])
            if plot_zero_line:
                ax[i, j].axhline(y=0, color='k', linestyle='--')
            if ticks is not None:
                if i == signals.shape[1] - 1:
                    ax[i, j].set_xlabel(ticks[j])
                    #labels = ['{}k'.format(int(time_tick/1000)) for time_tick in time_ticks]
                    #ax[i, j].set_xticks(time_ticks, labels)
                    #set_trace()
                else:
                    ax[i, j].set_xticks([])
                if j == 0:
                    ax[i, j].set_ylabel(ticks[i])
                    ax[i, j].set_yticks([-1, 0, 1])
                else:
                    ax[i, j].set_yticks([])

    if return_fig:
        return fig

### --- Programming tools --- ###

def config_generator(**kwargs):
    """Generator object that produces a Cartesian product of configurations.

    Each kwarg should be a list of possible values for the key. Yields a
    dictionary specifying a particular configuration."""

    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def rgetattr(obj, attr):
    """A "recursive" version of getattr that can handle nested objects.

    Args:
        obj (object): Parent object
        attr (string): Address of desired attribute with '.' between child
            objects.
    Returns:
        The attribute of obj referred to."""

    return reduce(getattr, [obj] + attr.split('.'))
