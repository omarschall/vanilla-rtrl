import numpy as np

def return_feature_vector(checkpoint, checkpoint_funcs=[]):
    """For a list of checkpoint_funcs, returns a vector with the checkpoint's
    value for each checkpoint_func."""

    feature_vector = []
    for checkpoint_func in checkpoint_funcs:

        feature_vector.append(checkpoint_func(checkpoint))

    feature_vector = np.array(feature_vector)

    return feature_vector