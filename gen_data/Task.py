from copy import copy
import numpy as np

class Task:
    """Parent class for all tasks. A Task is a class whose instances generate
    datasets to be used for training RNNs.

    A dataset is a dict of dicts with
    keys 'train' and 'test', which point to dicts with keys 'X' and 'Y' for
    inputs and labels, respectively. The values for 'X' and 'Y' are numpy
    arrays with shapes (time_steps, n_in) and (time_steps, n_out),
    respectively."""

    def __init__(self, n_in, n_out):
        """Initializes a Task with the number of input and output dimensions

        Args:
            n_in (int): Number of input dimensions.
            n_out (int): Number of output dimensions."""

        self.n_in = n_in
        self.n_out = n_out

    def gen_data(self, N_train, N_test, add_dummy_end_point=False):
        """Generates a data dict with a given number of train and test examples.

        Args:
            N_train (int): number of training examples
            N_test (int): number of testing examples
        Returns:
            data (dict): Dictionary pointing to 2 sub-dictionaries 'train'
                and 'test', each of which has keys 'X' and 'Y' for inputs
                and labels, respectively."""

        data = {'train': {}, 'test': {}}

        for mode, N in zip(['train', 'test'], [N_train, N_test]):
            X, Y, trial_type, trial_switch, loss_mask = self.gen_dataset(N)
            data[mode]['X'] = X
            data[mode]['Y'] = Y
            data[mode]['trial_type'] = trial_type
            data[mode]['trial_switch'] = trial_switch
            data[mode]['loss_mask'] = loss_mask
            if mode == 'train' and add_dummy_end_point:
                for key in data[mode].keys():
                    A = data[mode][key]
                    if A is not None:
                        data[mode][key] = np.concatenate([A, np.ones_like(A[0])[np.newaxis]])

        if hasattr(self, 'probe_dataset'):
            data['probe'] = copy(self.probe_dataset)

        return data

    def gen_dataset(self, N):
        """Function specific to each class, randomly generates a dictionary of
        inputs and labels.

        Args:
            N (int): number of examples
        Returns:
            dataset (dict): Dictionary with keys 'X' and 'Y' pointing to inputs
                and labels, respectively."""

        pass
