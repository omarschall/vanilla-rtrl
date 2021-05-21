import numpy as np
import matplotlib.pyplot as plt
from core import Simulation
from utils import *
from dynamics import *
from mpl_toolkits.mplot3d import Axes3D


class State_Space_Analysis:
    """An object that contains a coordinate transformation from high
    to low dimensions as well as a set of axes for plotting in this space."""

    def __init__(self, checkpoint, test_data, dim_reduction_method=Vanilla_PCA,
                 transform=None, **kwargs):
        """The array trajectories must have a shape of (sample, unit)"""

        if transform is None:
            self.transform = dim_reduction_method(checkpoint, test_data,
                                                  **kwargs)
        else:
            self.transform = transform

        dummy_data = np.zeros((10, checkpoint['rnn'].n_h))
        self.dim = self.transform(dummy_data).shape[1]

        self.fig = plt.figure()
        if self.dim == 2:
            self.ax = self.fig.add_subplot(111)
        if self.dim == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')

    def plot_in_state_space(self, trajectories, mark_start_and_end=False,
                            color='C0', *args, **kwargs):
        """Plots given trajectories' projection onto axes as defined in
        __init__ by training data."""

        projs = self.transform(trajectories)

        if self.dim == 2:
            self.ax.plot(projs[:, 0], projs[:, 1], *args, **kwargs,
                         color=color)
            if mark_start_and_end:
                self.ax.plot([projs[0, 0]], [projs[0, 1]], 'x', color=color)
                self.ax.plot([projs[-1, 0]], [projs[-1, 1]], 'o', color=color)
        if self.dim == 3:
            self.ax.plot(projs[:, 0], projs[:, 1], projs[:, 2],
                         *args, **kwargs, color=color)
            if mark_start_and_end:
                self.ax.plot([projs[0, 0]], [projs[0, 1]], [projs[0, 2]],
                             'x', color=color)
                self.ax.plot([projs[-1, 0]], [projs[-1, 1]], [projs[-1, 2]],
                             'o', color=color)

    def clear_plot(self):
        """Clears all plots from figure"""

        self.fig.axes[0].clear()
