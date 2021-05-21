import sys, os
sys.path.append(os.path.abspath('..'))
import numpy as np
from continual_learning.Continual_Learning import Continual_Learning

class Synaptic_Intelligence(Continual_Learning):
    """CL Method that preferentially weights each parameter by its importance
    for learning each previous task

    Ref: Zenke and Ganguli, PMLR 2017"""

    def __init__(self, rnn, c, epsilon=0.001):
        """Initialize by specifying RNN instance, regularization constant c,
        and stability constant epsilon.

        Args:
            rnn (core.RNN): Instance of RNN undergoing continual learning.
            c (float): Constant scaling overall strength of SI regularization.
            epsilon (float): Small positive constant to ensure invertability.
        """

        self.rnn = rnn
        self.c = c
        self.epsilon = epsilon

        self.SI_Theta = [[p.copy() for p in self.rnn.params]]
        self.SI_Delta = []
        self.SI_Omega = [np.zeros(s) for s in self.rnn.shapes]
        self.SI_omega = [np.zeros(s) for s in self.rnn.shapes]

    def mini_update(self, sim):
        """Update the 'importance' omega of each parameter for the *current*
        task at each time step.

        The importance omega is the average lr-scaled update direction
        (the optimizer velocity) with the gradient as computed by the learning
        algorithm, updated online at each time step."""

        #Update omegas
        for i_param in range(len(self.SI_omega)):
            v = sim.optimizer.vel[i_param]
            g = sim.grads_list[i_param]
            self.SI_omega[i_param] -= v * g

    def task_switch_update(self, sim):
        """Update the Omega quantity for each parameter by adding parameter
        importance omega scaled by overall change SI_Delta. Reset omega."""

        self.SI_Theta.append([p.copy() for p in self.rnn.params])
        self.SI_Delta = [p - q for p, q in zip(self.SI_Theta[-1],
                                               self.SI_Theta[-2])]

        for i_param in range(len(self.SI_omega)):
            omega = self.SI_omega[i_param]
            Delta = self.SI_Delta[i_param]
            self.SI_Omega[i_param] += omega / (Delta**2 + self.epsilon)

        self.SI_omega = [np.zeros(s) for s in self.rnn.shapes]

    def __call__(self, grads_list):
        """Compute derivative of regularized loss and add to gradients as
        computed by learning algorithm."""

        new_grads = []
        diff = []
        for i_param, grad in enumerate(grads_list):

            Omega = self.SI_Omega[i_param]
            Theta = self.rnn.params[i_param]
            Theta_tilde = self.SI_Theta[-1][i_param]
            new_grads.append(grad + 2 * self.c * Omega * (Theta - Theta_tilde))
            diff.append((Theta - Theta_tilde))

        self.loss = np.mean([O * np.square(d).mean() for d, O in zip(diff,
                                                                     Omega)])

        return new_grads
