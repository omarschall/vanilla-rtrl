
import numpy as np
import itertools
from functools import reduce
from copy import deepcopy
from itertools import product

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

def split_weight_matrix(A, sizes, axis=1):
    """Splits a weight matrix along the specified axis (0 for row, 1 for
    column) into a list of sub arrays of size specified by 'sizes'."""

    idx = [0] + np.cumsum(sizes).tolist()
    if axis == 1:
        ret = [np.squeeze(A[:,idx[i]:idx[i+1]]) for i in range(len(idx) - 1)]
    elif axis == 0:
        ret = [np.squeeze(A[idx[i]:idx[i+1],:]) for i in range(len(idx) - 1)]
    return ret
