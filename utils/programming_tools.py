
import numpy as np
import itertools
from functools import reduce

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


def get_param_values_from_list_of_config_strings(key_iterable, root_name):
    """For a list (or any iterable) of key strings for an array of simulation
    results, stores all possible parameter values in a dict of sorted lists.

    Args:
        key_iterable (iter): Any iterable containing the key strings
        root_name (str): The root name of the array of simulations

    Returns:
        param_values (dict): A dictionary of parameter names corresponding to
            sorted lists of the possible values they may take on.
        key_order (list): An ordered list of keys for parameters (including
            seed)."""

    param_values = {}
    for i_k, k in enumerate(key_iterable):

        config_str = k.split('analyze_' + root_name)[-1]
        key_value_pairs = [s.split('=') for s in config_str.split('_')][1:]

        if i_k == 0:
            key_order = [kvp[0] for kvp in key_value_pairs]

        for kvp in key_value_pairs:
            try:
                str_value = kvp[1].replace(',', '.')
                try:
                    value = int(str_value)
                except ValueError:
                    value = float(str_value)
                param_values[kvp[0]].add(value)
            except KeyError:
                param_values[kvp[0]] = set()

    for k in param_values.keys():
        param_values[k] = sorted(list(param_values[k]))

    return param_values, key_order

### --- COLORS --- ###

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return list(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def linearly_interpolate_colors(col1, col2, N):

    if '#' in col1:
        c1 = hex_to_rgb(col1)
    if '#' in col2:
        c2 = hex_to_rgb(col2)

    cols = np.linspace(c1, c2, N).astype(int)
    return [rgb_to_hex(tuple(c)) for c in cols]