import sys, os
sys.path.append(os.path.abspath('..'))

class Continual_Learning:
    """Shell class for continual learning method, simply demonstrates the
    structure of a CL method without any content."""

    def __init__(self, rnn):
        """Init function references an RNN instance that is learning to get
        relevant shapes for initialized internal variables."""

        pass

    def mini_update(self, sim):
        """The 'mini update' takes the simulation as an argument and is called
        every time step to update relevant variables of the CL method."""

        pass

    def task_switch_update(self, sim):
        """The 'task switch update' updates variables of CL method that only
        need to update when the sub-task is switched."""

        pass

    def __call__(self, grads_list):
        """The method is called at every time step to take the would-be
        gradients as produced by the learning algorithm and alter them based
        on the CL method."""

        return grads_list

