#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 15:03:11 2018

@author: omarschall
"""

import numpy as np
from pdb import set_trace
from utils import *
from functions import *
from copy import deepcopy

class Learning_Algorithm:
    """Parent class for all learning algorithms.

    Attributes:
        rnn (network.RNN): An instance of RNN to be trained by the network.
        n_* (int): Extra pointer to rnn.n_* (in, h, out) for conveneince.
        m (int): Number of recurrent "input dimensions" n_h + n_in + 1 including
            task inputs and constant 1 for bias.
        q (numpy array): Array of immediate error signals for the hidden units,
            i.e. the derivative of the current loss with respect to rnn.a, of
            shape (n_h).
        W_FB (numpy array or None): A fixed set of weights that may be provided
            for an approximate calculation of q in the manner of feedback
            alignment (Lillicrap et al. 2016).
        L2_reg (float or None): Strength of L2 regularization parameter on the
            network weights.
        a_ (numpy array): Array of shape (n_h + 1) that is the concatenation of
            the network's state and the constant 1, used to calculate the output
            errors.
        q (numpy array): The immediate loss derivative of the network state
            dL/da, calculated by propagate_feedback_to_hidden.
        q_prev (numpy array): The q value from the previous time step."""

    def __init__(self, rnn, allowed_kwargs_=set(), **kwargs):
        """Initializes an instance of learning algorithm by specifying the
        network to be trained, custom allowable kwargs, and kwargs.

        Args:
            rnn (network.RNN): An instance of RNN to be trained by the network.
            allowed_kwargs_ (set): Set of allowed kwargs in addition to those
                common to all child classes of Learning_Algorithm, 'W_FB' and
                'L2_reg'."""

        allowed_kwargs = {'W_FB', 'L1_reg', 'L2_reg',
                          'maintain_sparsity'}.union(allowed_kwargs_)

        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument passed'
                                'to Learning_Algorithm.__init__: ' + str(k))

        #Set all non-specified kwargs to None
        for attr in allowed_kwargs:
            if not hasattr(self, attr):
                setattr(self, attr, None)

        #Make kwargs attributes of the instance
        self.__dict__.update(kwargs)

        #Define basic learning algorithm properties
        self.rnn = rnn
        self.n_in = self.rnn.n_in
        self.n_h = self.rnn.n_h
        self.n_out = self.rnn.n_out
        self.m = self.n_h + self.n_in + 1
        self.q = np.zeros(self.n_h)

    def get_outer_grads(self):
        """Calculates the derivative of the loss with respect to the output
        parameters rnn.W_out and rnn.b_out.

        Calculates the outer gradients in the manner of a perceptron derivative
        by taking the outer product of the error with the "regressors" onto the
        output (the hidden state and constant 1).

        Returns:
            A numpy array of shape (rnn.n_out, self.n_h + 1) containing the
                concatenation (along column axis) of the derivative of the loss
                w.r.t. rnn.W_out and w.r.t. rnn.b_out."""

        self.a_ = np.concatenate([self.rnn.a, np.array([1])])
        return np.multiply.outer(self.rnn.error, self.a_)

    def propagate_feedback_to_hidden(self):
        """Performs one step of backpropagation from the outer-layer errors to
        the hidden state.

        Calculates the immediate derivative of the loss with respect to the
        hidden state rnn.a. By default, this is done by taking rnn.error (dL/dz)
        and applying the chain rule, i.e. taking its matrix product with the
        derivative dz/da, which is rnn.W_out. Alternatively, if 'W_FB' attr is
        provided to the instance, then these feedback weights, rather the W_out,
        are used, as in feedback alignment. (See Lillicrap et al. 2016.)

        Updates q to the current value of dL/da."""

        self.q_prev = np.copy(self.q)

        if self.W_FB is None:
            self.q = self.rnn.error.dot(self.rnn.W_out)
        else:
            self.q = self.rnn.error.dot(self.W_FB)

    def L2_regularization(self, grads):
        """Adds L2 regularization to the gradient.

        Args:
            grads (list): List of numpy arrays representing gradients before L2
                regularization is applied.
        Returns:
            A new list of grads with L2 regularization applied."""

        #Get parameters affected by L2 regularization
        L2_params = [self.rnn.params[i] for i in self.rnn.L2_indices]
        #Add to each grad the corresponding weight's current value, weighted
        #by the L2_reg hyperparameter.
        for i_L2, W in zip(self.rnn.L2_indices, L2_params):
            grads[i_L2] += self.L2_reg * W
        #Calculate L2 loss for monitoring purposes
        self.L2_loss = 0.5 * sum([norm(p)**2 for p in L2_params])
        return grads
    
    def L1_regularization(self, grads):
        """Adds L1 regularization to the gradient.

        Args:
            grads (list): List of numpy arrays representing gradients before L1
                regularization is applied.
        Returns:
            A new list of grads with L1 regularization applied."""

        #Get parameters affected by L1 regularization
        #(identical as those affected by L2)
        L1_params = [self.rnn.params[i] for i in self.rnn.L2_indices]
        #Add to each grad the sign of the corresponding parameter weighted
        #by L1 reg strength
        for i_L1, W in zip(self.rnn.L2_indices, L1_params):
            grads[i_L1] += self.L1_reg * np.sign(W)
        #Calculate L2 loss for monitoring purposes
        self.L1_loss = sum([norm(p) for p in L1_params])
        return grads

    def apply_sparsity_to_grads(self, grads):
        """"If called, modifies gradient to make 0 any parameters that are
        already 0 (only for L2 params).
        
        Args:
            grads (list): List of numpy arrays representing gradients before L2
                regularization is applied.
        Returns:
            A new list of grads with L2 regularization applied."""
            
        #Get parameters affected by L2 regularization
        L2_params = [self.rnn.params[i] for i in self.rnn.L2_indices]
        #AMultiply each gradient by 0 if it the corresponding weight is
        #already 0
        for i_L2, W in zip(self.rnn.L2_indices, L2_params):
            grads[i_L2] *= (W != 0)
        return grads

    def __call__(self):
        """Calculates the final list of grads for this time step.

        Assumes the user has already called self.update_learning_vars, a
        method specific to each child class of Real_Time_Learning_Algorithm
        that updates internal learning variables, e.g. the influence matrix of
        RTRL. Then calculates the outer grads (gradients of W_out and b_out),
        updates q using propagate_feedback_to_hidden, and finally calling the
        get_rec_grads method (specific to each child class) to get the gradients
        of W_rec, W_in, and b_rec as one numpy array with shape (n_h, m). Then
        these gradients are split along the column axis into a list of 5
        gradients for W_rec, W_in, b_rec, W_out, b_out. L2 regularization is
        applied if L2_reg parameter is not None.

        Returns:
            List of gradients for W_rec, W_in, b_rec, W_out, b_out."""

        self.outer_grads = self.get_outer_grads()
        self.propagate_feedback_to_hidden()
        self.rec_grads = self.get_rec_grads()
        rec_grads_list = split_weight_matrix(self.rec_grads,
                                             [self.n_h, self.n_in, 1])
        outer_grads_list = split_weight_matrix(self.outer_grads,
                                               [self.n_h, 1])
        grads_list = rec_grads_list + outer_grads_list

        if self.L1_reg is not None:
            grads_list = self.L1_regularization(grads_list)

        if self.L2_reg is not None:
            grads_list = self.L2_regularization(grads_list)

        if self.maintain_sparsity:
            grads_list = self.apply_sparsity_to_grads(grads_list)

        return grads_list

    def reset_learning(self):
        """Resets internal variables of the learning algorithm (relevant if
        simulation includes a trial structure). Default is to do nothing."""

        pass

class Only_Output_Weights(Learning_Algorithm):
    """Updates only the output weights W_out and b_out"""

    def __init__(self, rnn, **kwargs):

        self.name = 'Only_Output_Weights'
        allowed_kwargs_ = set()
        super().__init__(rnn, allowed_kwargs_, **kwargs)

    def update_learning_vars(self):
        """No internal variables to update."""

        pass

    def get_rec_grads(self):
        """Returns all 0s for the recurrent gradients."""

        return np.zeros((self.n_h, self.m))

class RTRL(Learning_Algorithm):
    """Implements the Real-Time Recurrent Learning (RTRL) algorithm from
    Williams and Zipser 1989.

    RTRL maintains a long-term "influence matrix" dadw that represents the
    derivative of the hidden state with respect to a flattened vector of
    recurrent update parameters. We concatenate [W_rec, W_in, b_rec] along
    the column axis and order the flattened vector of parameters by stacking
    the columns end-to-end. In other words, w_k = W_{ij} when i = k%n_h and
    j = k//n_h. The influence matrix updates according to the equation

    M' = JM + M_immediate                            (1)

    where J is the network Jacobian and M_immediate is the immediate influence
    of a parameter w on the hidden state a. (See paper for more detailed
    notation.) M_immediate is notated as papw in the code for "partial a partial
    w." For a vanilla network, this can be simply (if inefficiently) computed as
    the Kronecker product of a_hat = [a_prev, x, 1] (a concatenation of the prev
    hidden state, the input, and a constant 1 (for bias)) with the activation
    derivatives organized in a diagonal matrix. The implementation of Eq. (1)
    is in the update_learning_vars method.

    Finally, the algorithm returns recurrent gradients by projecting the
    feedback vector q onto the influence matrix M:

    dL/dw = dL/da da/dw = qM                         (2)

    Eq. (2) is implemented in the get_rec_grads method."""

    def __init__(self, rnn, M_decay=1, **kwargs):
        """Inits an RTRL instance by setting the initial dadw matrix to zero."""

        self.name = 'RTRL' #Algorithm name
        allowed_kwargs_ = set() #No special kwargs for RTRL
        super().__init__(rnn, allowed_kwargs_, **kwargs)

        #Initialize influence matrix
        self.dadw = np.zeros((self.n_h, self.rnn.n_h_params))
        self.M_decay = M_decay

    def update_learning_vars(self):
        """Updates the influence matrix via Eq. (1)."""

        #Get relevant values and derivatives from network.
        self.a_hat = np.concatenate([self.rnn.a_prev,
                                     self.rnn.x,
                                     np.array([1])])
        D = np.diag(self.rnn.activation.f_prime(self.rnn.h))
        self.papw = np.kron(self.a_hat, D) #Calculate M_immediate
        self.rnn.get_a_jacobian() #Get updated network Jacobian

        #Update influence matrix via Eq. (1).
        self.dadw = self.M_decay * self.rnn.a_J.dot(self.dadw) + self.papw

    def get_rec_grads(self):
        """Calculates recurrent grads using Eq. (2), reshapes into original
        matrix form."""

        return self.q.dot(self.dadw).reshape((self.n_h, self.m), order='F')

    def reset_learning(self):
        """Resets learning algorithm by setting influence matrix to 0."""

        self.dadw *= 0

class Stochastic_Algorithm(Learning_Algorithm):

    def sample_nu(self):
        """Sample nu from specified distribution."""

        if self.nu_dist == 'discrete' or self.nu_dist is None:
            nu = np.random.choice([-1, 1], self.n_nu)
        elif self.nu_dist == 'gaussian':
            nu = np.random.normal(0, 1, self.n_nu)
        elif self.nu_dist == 'uniform':
            nu = np.random.uniform(-1, 1, self.n_nu)

        return nu

class UORO(Stochastic_Algorithm):
    """Implements the Unbiased Online Recurrent Optimization (UORO) algorithm
    from Tallec et al. 2017.

    Full details in our review paper or in original paper. Broadly, an outer
    product approximation of M is maintained by 2 vectors A and B, which update
    by the equations

    A' = p0 J A + p1 \nu        (1)
    B' = 1/p0 B + 1/p1 \nu M_immediate      (2)

    where \nu is a vector of zero-mean iid samples. p0 and p1 are calculated by

    p0 = \sqrt{norm(B)/norm(A)}       (3)
    p1 = \sqrt{norm(\nu papw)/norm(\nu)}        (4)

    These equations are implemented in update_learning_vars by two different
    approaches. If 'epsilon' is provided as an argument, then the "forward
    differentiation" method from the original paper is used, where the matrix-
    vector product JA is estimated numerically by a perturbation of size
    epsilon in the A direction.

    Then the recurrent gradients are calculated by

    dL/dw = qM = (q A) B    (5)

    Eq. (5) is implemented in the get_rec_grads method."""

    def __init__(self, rnn, **kwargs):
        """Inits an UORO instance by setting the initial values of A and B to be
        iid samples from a standard normal distribution, to avoid dividing by
        zero in Eqs. (3) and (4).

        Keyword args:
            epsilon (float): Scaling factor on perturbation for forward
                differentiation method. If not provided, exact derivative is
                calculated instead.
            P0 (float): Overrides calculation of p0, instead uses provided value
                of P0. If not provided, p0 is calculated according to Eq. (3).
            P1 (float): Same for p1.
            A (numpy array): Initial value for A.
            B (numpy array): Initial value for B.
            nu_dist (string): Takes on the value of 'gaussian', 'discrete', or
                'uniform' to indicate what type of distribution nu should sample
                 from. Default is 'discrete'."""

        self.name = 'UORO' #Default algorithm name
        allowed_kwargs_ = {'epsilon', 'P0', 'P1', 'A', 'B', 'nu_dist'}
        super().__init__(rnn, allowed_kwargs_, **kwargs)
        self.n_nu = self.n_h

        #Initialize A and B arrays
        if self.A is None:
            self.A = np.random.normal(0, 1, self.n_h)
        if self.B is None:
            self.B = np.random.normal(0, 1, (self.n_h, self.m))

    def update_learning_vars(self, update=True):
        """Implements Eqs. (1), (2), (3), and (4) to update the outer product
        approximation of the influence matrix by A and B.

        Args:
            update (bool): If True, updates the algorithm's current outer
                product approximation B, A. If False, only prepares for calling
                get_influence_estimate."""

        #Get relevant values and derivatives from network
        self.a_hat = np.concatenate([self.rnn.a_prev,
                                     self.rnn.x,
                                     np.array([1])])
        D = self.rnn.activation.f_prime(self.rnn.h)
        #Compact form of M_immediate
        self.papw = np.multiply.outer(D, self.a_hat)
        self.rnn.get_a_jacobian() #Get updated network Jacobian

        A, B = self.get_influence_estimate()

        if update:
            self.A, self.B = A, B

    def get_influence_estimate(self):
        """Generates one random outer-product estimate of the influence matrix.

        Samples a random vector nu of iid samples with 0 mean from a
        distribution given by nu_dist, and returns an updated estimate
        of A and B from Eqs. (1)-(4).

        Returns:
            Updated A (numpy array of shape (n_h)) and B (numpy array of shape
                (n_h, m))."""

        #Sample random vector
        self.nu = self.sample_nu()

        #Get random projection of M_immediate onto \nu
        M_projection = (self.papw.T*self.nu).T

        if self.epsilon is not None: #Forward differentiation method
            eps = self.epsilon
            #Get perturbed state in direction of A
            self.a_perturbed = self.rnn.a_prev + eps * self.A
            #Get hypothetical next states from this perturbation
            self.a_perturbed_next = self.rnn.next_state(self.rnn.x,
                                                        self.a_perturbed,
                                                        update=False)
            #Get forward-propagated A
            self.A_forwards = (self.a_perturbed_next - self.rnn.a) / eps
            #Calculate scaling factors
            B_norm = norm(self.B)
            A_norm = norm(self.A_forwards)
            M_norm = norm(M_projection)
            self.p0 = np.sqrt(B_norm/(A_norm + eps)) + eps
            self.p1 = np.sqrt(M_norm/(np.sqrt(self.n_h) + eps)) + eps
        else: #Backpropagation method
            #Get forward-propagated A
            self.A_forwards = self.rnn.a_J.dot(self.A)
            #Calculate scaling factors
            B_norm = norm(self.B)
            A_norm = norm(self.A_forwards)
            M_norm = norm(M_projection)
            self.p0 = np.sqrt(B_norm/A_norm)
            self.p1 = np.sqrt(M_norm/np.sqrt(self.n_h))

        #Override with fixed P0 and P1 if given
        if self.P0 is not None:
            self.p0 = np.copy(self.P0)
        if self.P1 is not None:
            self.p1 = np.copy(self.P1)

        #Update outer product approximation
        A = self.p0 * self.A_forwards + self.p1 * self.nu
        B = (1/self.p0) * self.B + (1 / self.p1) * M_projection

        return A, B

    def get_rec_grads(self):
        """Calculates recurrent grads by taking matrix product of q with the
        estimate of the influence matrix.

        First associates q with A to calculate a "global learning signal"
        Q, which multiplies by B to compute the recurrent gradient, which
        is reshaped into original matrix form.

        Returns:
            An array of shape (n_h, m) representing the recurrent gradient."""

        self.Q = self.q.dot(self.A) #"Global learning signal"
        return (self.Q * self.B)

    def reset_learning(self):
        """Resets learning by re-randomizing the outer product approximation to
        random gaussian samples."""

        self.A = np.random.normal(0, 1, self.n_h)
        self.B = np.random.normal(0, 1, (self.n_h, self.m))

class KF_RTRL(Stochastic_Algorithm):
    """Implements the Kronecker-Factored Real-Time Recurrent Learning Algorithm
    (KF-RTRL) from Mujika et al. 2018.

    Details in review paper or original Mujika et al. 2018. Broadly, M is
    approximated as a Kronecker product between a (row) vector A and a matrix
    B, which updates as

    A' = \nu_0 p0 A + \nu_1 p1 a_hat        (1)
    B' = \nu_0 1/p0 JB + \nu_1 1/p1 \alpha diag(\phi'(h))      (2)

    where \nu = (\nu_0, \nu_1) is a vector of zero-mean iid samples, a_hat is
    the concatenation [a_prev, x, 1], and p0 and p1 are calculated by

    p0 = \sqrt{norm(JB)/norm(A)}       (3)
    p1 = \sqrt{norm(D)/norm(a_hat)}        (4)

    Then the recurrent gradients are calculated by

    dL/dw = qM = A (qB)    (5)

    Eq. (5) is implemented in the get_rec_grads method.
    """

    def __init__(self, rnn, **kwargs):
        """Inits a KF-RTRL instance by setting the initial values of A and B to
        be iid samples from a gaussian distributions, to avoid dividing by
        zero in Eqs. (3) and (4).

        Keyword args:
            P0 (float): Overrides calculation of p0, instead uses provided value
                of P0. If not provided, p0 is calculated according to Eq. (3).
            P1 (float): Same for p1.
            A (numpy array): Initial value for A.
            B (numpy array): Initial value for B.
            nu_dist (string): Takes on the value of 'gaussian', 'discrete', or
                'uniform' to indicate what type of distribution nu should sample
                 from. Default is 'discrete'."""

        self.name = 'KF-RTRL'
        allowed_kwargs_ = {'P0', 'P1', 'A', 'B', 'nu_dist'}
        super().__init__(rnn, allowed_kwargs_, **kwargs)
        self.n_nu = 2

        #Initialize A and B arrays
        if self.A is None:
            self.A = np.random.normal(0, 1, self.m)
        if self.B is None:
            self.B = np.random.normal(0, 1, (self.n_h, self.n_h))

    def update_learning_vars(self, update=True):
        """Implements Eqs. (1), (2), (3), and (4) to update the Kron. product
        approximation of the influence matrix by A and B.

        Args:
            update (bool): If True, updates the algorithm's current outer
                product approximation B, A. If False, only prepares for calling
                get_influence_estimate."""

        #Get relevant values and derivatives from network
        self.a_hat = np.concatenate([self.rnn.a_prev,
                                     self.rnn.x,
                                     np.array([1])])
        self.D = np.diag(self.rnn.activation.f_prime(self.rnn.h))
        self.rnn.get_a_jacobian()
        self.B_forwards = self.rnn.a_J.dot(self.B)

        A, B = self.get_influence_estimate()

        if update:
            self.A, self.B = A, B

    def get_influence_estimate(self):
        """Generates one random Kron.-product estimate of the influence matrix.

        Samples a random vector nu of iid samples with 0 mean from a
        distribution given by nu_dist, and returns an updated estimate
        of A and B from Eqs. (1)-(4).

        Returns:
            Updated A (numpy array of shape (m)) and B (numpy array of shape
                (n_h, n_h))."""

        #Sample random vector (shape (2) in KF-RTRL)
        self.nu = self.sample_nu()

        #Calculate p0, p1 or override with fixed P0, P1 if given
        if self.P0 is None:
            self.p0 = np.sqrt(norm(self.B_forwards)/norm(self.A))
        else:
            self.p0 = np.copy(self.P0)
        if self.P1 is None:
            self.p1 = np.sqrt(norm(self.D)/norm(self.a_hat))
        else:
            self.p1 = np.copy(self.P1)

        #Update Kronecker product approximation
        A = self.nu[0]*self.p0*self.A + self.nu[1]*self.p1*self.a_hat
        B = (self.nu[0]*(1/self.p0)*self.B_forwards +
             self.nu[1]*(1/self.p1)*self.D)
        # if np.sum(A)==0:
        #     set_trace()
        return A, B

    def get_rec_grads(self):
        """Calculates recurrent grads by taking matrix product of q with the
        estimate of the influence matrix.

        First associates q with B to calculate a vector qB, whose Kron. product
        with A (effectively an outer product upon reshaping) gives the estimated
        recurrent gradient.

        Returns:
            An array of shape (n_h, m) representing the recurrent gradient."""

        self.qB = self.q.dot(self.B) #Unit-specific learning signal
        return np.kron(self.A, self.qB).reshape((self.n_h, self.m), order='F')

    def reset_learning(self):
        """Resets learning by re-randomizing the outer product approximation to
        random gaussian samples."""

        self.A = np.random.normal(0, 1, self.m)
        self.B = np.random.normal(0, 1, (self.n_h, self.n_h))

class Reverse_KF_RTRL(Stochastic_Algorithm):
    """Implements the "Reverse" KF-RTRL (R-KF-RTRL) algorithm.

    Full details in our review paper. Broadly, an approximation of M in the form
    of a Kronecker product between a matrix B and a (row) vector A is maintained
    by the update

    A'_i = p0 A_i + p1 \nu_i        (1)
    B'_{kj} = (1/p0 \sum_{k'} J+{kk'}B_{k'j} +
               1/p1 \sum_i \nu_i M_immediate_{kij})      (2)

    where \nu is a vector of zero-mean iid samples. p0 and p1 are calculated by

    p0 = \sqrt{norm(B)/norm(A)}       (3)
    p1 = \sqrt{norm(\nu papw)/norm(\nu)}        (4)

    Then the recurrent gradients are calculated by

    dL/dw = qM = A (qB)    (5)

    Eq. (5) is implemented in the get_rec_grads method."""

    def __init__(self, rnn, **kwargs):
        """Inits an R-KF-RTRL instance by setting the initial values of A and B
        to be iid samples from gaussian distributions, to avoid dividing by
        zero in Eqs. (3) and (4).

        Keyword args:
            epsilon (float): Scaling factor on perturbation for forward
                differentiation method. If not provided, exact derivative is
                calculated instead.
            P0 (float): Overrides calculation of p0, instead uses provided value
                of P0. If not provided, p0 is calculated according to Eq. (3).
            P1 (float): Same for p1.
            A (numpy array): Initial value for A.
            B (numpy array): Initial value for B.
            nu_dist (string): Takes on the value of 'gaussian', 'discrete', or
                'uniform' to indicate what type of distribution nu should sample
                 from. Default is 'discrete'."""

        self.name = 'R-KF-RTRL'
        allowed_kwargs_ = {'P0', 'P1', 'A', 'B', 'nu_dist'}
        super().__init__(rnn, allowed_kwargs_, **kwargs)
        self.n_nu = self.n_h

        #Initialize A and B arrays
        if self.A is None:
            self.A = np.random.normal(0, 1, self.n_h)
        if self.B is None:
            self.B = np.random.normal(0, 1, (self.n_h, self.m))

    def update_learning_vars(self, update=True):
        """Implements Eqs. (1), (2), (3), and (4) to update the Kron. product
        approximation of the influence matrix by A and B.

        Args:
            update (bool): If True, updates the algorithm's current outer
                product approximation B, A. If False, only prepares for calling
                get_influence_estimate."""

        #Get relevant values and derivatives from network
        self.a_hat = np.concatenate([self.rnn.a_prev,
                                     self.rnn.x,
                                     np.array([1])])
        self.D = self.rnn.activation.f_prime(self.rnn.h)
        #Compact form of M_immediate
        self.papw = np.multiply.outer(self.D, self.a_hat)
        self.rnn.get_a_jacobian() #Get updated network Jacobian
        self.B_forwards = self.rnn.a_J.dot(self.B)

        A, B = self.get_influence_estimate()

        if update:
            self.A, self.B = A, B

    def get_influence_estimate(self):
        """Generates one random Kron.-product estimate of the influence matrix.

        Samples a random vector nu of iid samples with 0 mean from a
        distribution given by nu_dist, and returns an updated estimate
        of A and B from Eqs. (1)-(4).

        Returns:
            Updated A (numpy array of shape (n_h)) and B (numpy array of shape
                (n_h, n_m))."""

        #Sample random vector
        self.nu = self.sample_nu()

        # Get random projection of M_immediate onto \nu
        M_projection = (self.papw.T * self.nu).T

        #Calculate scaling factors
        B_norm = norm(self.B_forwards)
        A_norm = norm(self.A)
        M_norm = norm(M_projection)
        self.p0 = np.sqrt(B_norm/A_norm)
        self.p1 = np.sqrt(M_norm/np.sqrt(self.n_h))

        #Override with fixed P0 and P1 if given
        if self.P0 is not None:
            self.p0 = np.copy(self.P0)
        if self.P1 is not None:
            self.p1 = np.copy(self.P1)

        #Update "inverse" Kronecker product approximation
        A = self.p0 * self.A + self.p1 * self.nu
        B = (1/self.p0) * self.B_forwards + (1/self.p1) * M_projection

        return A, B

    def get_rec_grads(self):
        """Calculates recurrent grads by taking matrix product of q with the
        estimate of the influence matrix.

        First associates q with B to calculate qB, then takes the outer product
        with A to get an estimate of the recurrent gradient.

        Returns:
            An array of shape (n_h, m) representing the recurrent gradient."""

        self.qB = self.q.dot(self.B) #Unit-specific learning signal
        return np.multiply.outer(self.A, self.qB)

    def reset_learning(self):
        """Resets learning by re-randomizing the Kron. product approximation to
        random gaussian samples."""

        self.A = np.random.normal(0, 1, self.n_h)
        self.B = np.random.normal(0, 1, (self.n_h, self.m))

class RFLO(Learning_Algorithm):
    """Implements the Random-Feedback Local Online learning algorithm (RFLO)
    from Murray 2019.

    Maintains an eligibility trace B that is updated by temporally filtering
    the immediate influences \phi'(h_i) a_hat_j by the network's inverse time
    constant \alpha:

    B'_{ij} = (1 - \alpha) B_{ij} + \alpha \phi'(h_i) a_hat_j       (1)

    Eq. (1) is implemented by update_learning_vars method. Gradients are then
    calculated according to

    q_i B_{ij}      (2)

    which is implemented in get_rec_grads."""


    def __init__(self, rnn, alpha, **kwargs):
        """Inits an RFLO instance by specifying the inverse time constant for
        the eligibility trace.

        Args:
            alpha (float): Float between 0 and 1 specifying the inverse time
                constant of the eligilibility trace, typically chosen to be
                equal to alpha for the network.

        Keyword args:
            B (numpy array): Initial value for B (all 0s if unspecified)."""

        self.name = 'RFLO'
        allowed_kwargs_ = {'B'}
        super().__init__(rnn, allowed_kwargs_, **kwargs)

        self.alpha = alpha
        if self.B is None:
            self.B = np.zeros((self.n_h, self.m))

    def update_learning_vars(self):
        """Updates B by one time step of temporal filtration via the invesre
        time constant alpha (see Eq. 1)."""

        #Get relevant values and derivatives from network
        self.a_hat = np.concatenate([self.rnn.a_prev,
                                     self.rnn.x,
                                     np.array([1])])
        self.D = self.rnn.activation.f_prime(self.rnn.h)
        self.M_immediate = self.alpha * np.multiply.outer(self.D, self.a_hat)

        #Update eligibility traces
        self.B = (1 - self.alpha) * self.B + self.M_immediate

    def get_rec_grads(self):
        """Implements Eq. (2) from above."""

        return (self.q * self.B.T).T

    def reset_learning(self):
        """Reset eligibility trace to 0."""

        self.B *= 0

class DNI(Learning_Algorithm):
    """Implements the Decoupled Neural Interface (DNI) algorithm for an RNN from
    Jaderberg et al. 2017.

    Details are in our review, original paper, or Czarnecki et al. 2017.
    Briefly, we linearly approximate the (future-facing) credit assignment
    vector c = dL/da by the variable 'sg' for 'synthetic gradient'
    (of shape (n_h)) using

    c ~ sg = A a_tilde                                                     (1)

    where a_tilde = [a; y^*; 1] is the network state concatenated with the label
    y^* and a constant 1 (for bias), of shape (m_out = n_h + n_in + 1). Then the
    gradient is calculated by combining this estimate with the immediate
    parameter influence \phi'(h) a_hat

    dL/dW_{ij} = dL/da_i da_i/dW_{ij} = sg_i alpha \phi'(h_i) a_hat_j.     (2)

    The matrix A must be updated as well to make sure Eq. (1) is a good
    estimate. Details are in our paper or the original; briefly, a bootstrapped
    credit assignment estimate is formed via

    c^* = q_prev + sg J = q_prev + (A a_tilde_prev) J                      (3)

    where J = da^t/da^{t-1} is the network Jacobian, either calculated exactly
    or approximated (see update_J_approx method). The arrays q_prev and
    a_tilde_prev are q and a_tilde from the previous time step;
    this is because the update requires reference to the "future" (by one time
    step) network state, so we simply wait a time step, such that q and a_tilde
    are now q_prev and a_tilde_prev, respectively. This target for credit
    assignment is then used to calculate a prediction loss gradient for A:

    d/dA_{ij} 0.5 * ||A a_tilde - c^*||^2 = (A a_tilde - c^*)_i a_tilde_j  (4)

    This gradient is used to update A by a given optimizer."""

    def __init__(self, rnn, optimizer, **kwargs):
        """Inits an instance of DNI by specifying the optimizer for the A
        weights and other kwargs.

        Args:
            optimizer (optimizers.Optimizer): An Optimizer instance used to
                update the weights of A based on its credit assignment
                prediction loss gradients.

        Keywords args:
            A (numpy array): Initial value of A weights, must be of shape
                (n_h, m_out). If None, A is initialized with random Gaussian.
            J_lr (float): Learning rate for learning approximate Jacobian.
            activation (functions.Function): Activation function for the
                synthetic gradient function, applied elementwise after A a_tilde
                operation. Default is identity.
            SG_label_activation (functions.Function): Activation function for
                the synthetic gradient function as used in calculating the
                *label* for the
            use_approx_J (bool): If True, trains the network using the
                approximated Jacobian rather than the exact Jacobian.
            SG_L2_reg (float): L2 regularization strength on the A weights, by
                default 0.
            fix_A_interval (int): The number of time steps to wait between
                updating the synthetic gradient method used to bootstrap the
                label estimates. Default is 5."""

        self.name = 'DNI'
        allowed_kwargs_ = {'A', 'J_lr', 'activation', 'SG_label_activation',
                           'use_approx_J', 'SG_L2_reg', 'fix_A_interval'}
        #Default parameters
        self.optimizer = optimizer
        self.SG_L2_reg = 0
        self.fix_A_interval = 5
        self.activation = identity
        self.SG_label_activation = identity
        self.use_approx_J = False
        #Override defaults with kwargs
        super().__init__(rnn, allowed_kwargs_, **kwargs)

        self.m_out = self.n_h + self.n_out + 1
        self.J_approx = np.copy(self.rnn.W_rec)
        self.i_fix = 0
        if self.A is None:
            self.A = np.random.normal(0, np.sqrt(1/self.m_out),
                                      (self.n_h, self.m_out))
        self.A_ = np.copy(self.A)

    def update_learning_vars(self):
        """Updates the A matrix by Eqs. (3) and (4)."""

        if self.use_approx_J: #If using approximate Jacobian, update it.
            self.update_J_approx()
        else: #Otherwise get the exact Jacobian.
            self.rnn.get_a_jacobian()

        #Compute synthetic gradient estimate of credit assignment at previous
        #time step. This is NOT used to drive learning in W but rather to drive
        #learning in A.
        self.a_tilde_prev = np.concatenate([self.rnn.a_prev,
                                            self.rnn.y_prev,
                                            np.array([1])])
        self.sg = self.synthetic_grad(self.a_tilde_prev)

        #Compute the target, error and loss for the synthetic gradient function
        self.sg_target = self.get_sg_target()
        self.A_error = self.sg - self.sg_target
        self.A_loss = 0.5 * np.square(self.A_error).mean()

        #Compute gradients for A
        self.scaled_A_error = self.A_error * self.activation.f_prime(self.sg_h)
        self.A_grad = np.multiply.outer(self.scaled_A_error, self.a_tilde_prev)

        #Apply L2 regularization to A
        if self.SG_L2_reg > 0:
            self.A_grad += self.SG_L2_reg * self.A

        #Update synthetic gradient parameters
        self.A = self.optimizer.get_updated_params([self.A], [self.A_grad])[0]

        #On interval determined by self.fix_A_interval, update A_, the values
        #used to calculate the target in Eq. (3), with the latest value of A.
        if self.i_fix == self.fix_A_interval - 1:
            self.i_fix = 0
            self.A_ = np.copy(self.A)
        else:
            self.i_fix += 1

    def get_sg_target(self):
        """Function for generating the target for training A. Implements Eq. (3)
        using a different set of weights A_, which are static and only
        re-assigned to A  every fix_A_interval time steps.

        Returns:
            sg_target (numpy array): Array of shape (n_out) used to get error
                signals for A in update_learning_vars."""

        #Get latest q value, slide current q value to q_prev.
        self.propagate_feedback_to_hidden()

        self.a_tilde = np.concatenate([self.rnn.a,
                                       self.rnn.y,
                                       np.array([1])])

        #Calculate the synthetic gradient for the 'next' (really the current,
        #but next relative to the previous) time step.
        sg_next = self.synthetic_grad_(self.a_tilde)
        #Backpropagate by one time step and add to q_prev to get sg_target.
        if self.use_approx_J: #Approximate Jacobian
            sg_target = self.q_prev + sg_next.dot(self.J_approx)
        else: #Exact Jacobian
            sg_target = self.q_prev + sg_next.dot(self.rnn.a_J)

        return sg_target

    def update_J_approx(self):
        """Updates the approximate Jacobian by SGD according to a squared-error
        loss function:

        J_loss = 0.5 * || J a_prev - a ||^2.                     (6)

        Thus the gradient for the Jacobian is

        dJ_loss/dJ_{ij} = (J a_prev - a)_i a_prev_j              (7)."""

        self.J_error = self.J_approx.dot(self.rnn.a_prev) - self.rnn.a
        self.J_loss = 0.5 * np.square(self.J_error).mean()
        self.J_approx -= self.J_lr * np.multiply.outer(self.J_error,
                                                       self.rnn.a_prev)

    def synthetic_grad(self, a_tilde):
        """Computes the synthetic gradient using current values of A.

        Retuns:
            An array of shape (n_h) representing the synthetic gradient."""

        self.sg_h = self.A.dot(a_tilde)
        return self.activation.f(self.sg_h)

    def synthetic_grad_(self, a_tilde):
        """Computes the synthetic gradient using A_ and with an extra activation
        function (for DNI(b)), only for computing the label in Eq. (3).

        Retuns:
            An array of shape (n_h) representing the synthetic gradient."""

        self.sg_h_ = self.A_.dot(a_tilde)
        return self.SG_label_activation.f((self.activation.f(self.sg_h_)))

    def get_rec_grads(self):
        """Computes the recurrent grads for the network by implementing Eq. (2),
        using the current synthetic gradient function.

        Note: assuming self.a_tilde already calculated by calling get_sg_target,
        which should happen by default since update_learning_vars is always
        called before get_rec_grads.

        Returns:
            An array of shape (n_h, m) representing the network gradient for
                the recurrent parameters."""

        #Calculate synthetic gradient
        self.sg = self.synthetic_grad(self.a_tilde)
        #Combine the first 3 factors of the RHS of Eq. (2) into sg_scaled
        D = self.rnn.activation.f_prime(self.rnn.h)
        self.sg_scaled = self.sg * self.rnn.alpha * D

        self.a_hat = np.concatenate([self.rnn.a_prev,
                                     self.rnn.x,
                                     np.array([1])])
        #Final result of Eq. (2)
        return np.multiply.outer(self.sg_scaled, self.a_hat)

class Efficient_BPTT(Learning_Algorithm):
    """Implements the 'E-BPTT' version of backprop we discuss in the paper for
    an RNN.

    We describe in more detail in the paper. In brief, the network activity is
    'unrolled' for T_trunction time steps in non-overlapping intervals. The
    gradient for each interval is computed using the future-facing relation
    from Section 2. Thus 'update_learning_vars' is called at every step to
    update the memory of relevant network variables, while get_rec_grads only
    returns non-zero elements every T_truncation time steps."""

    def __init__(self, rnn, T_truncation, **kwargs):
        """Inits an instance of Efficient_BPTT by specifying the network to
        train and the truncation horizon. No default allowable kwargs."""

        self.name = 'E-BPTT'
        allowed_kwargs_ = {'c_clip_norm'}
        super().__init__(rnn, allowed_kwargs_, **kwargs)

        self.T_truncation = T_truncation

        # Initialize lists for storing network data
        self.a_hat_history = []
        self.h_history = []
        self.q_history = []

    def update_learning_vars(self):
        """Updates the memory of the algorithm with the relevant network
        variables for running E-BPTT."""

        # Add latest values to list
        self.a_hat_history.insert(0, np.concatenate([self.rnn.a_prev,
                                                     self.rnn.x,
                                                     np.array([1])]))
        self.h_history.insert(0, self.rnn.h)
        self.propagate_feedback_to_hidden()
        self.q_history.insert(0, self.q)

    def get_rec_grads(self):
        """Using the accumulated history of q, h and a_hat values over the
        truncation interval, computes the recurrent gradient.

        Returns:
            rec_grads (numpy array): Array of shape (n_h, m) representing
                the gradient dL/dW after truncation interval completed,
                otherwise an array of 0s of the same shape."""

        #Once a 'triangle' is formed (see Fig. 3 in paper), compute gradient.
        if len(self.a_hat_history) >= self.T_truncation:

            #Initialize recurrent grads at 0
            rec_grads = np.zeros((self.n_h, self.m))
            #Start with most recent credit assignment value
            c = self.q_history.pop(0)

            for i_BPTT in range(self.T_truncation):

                #Truncate credit assignment norm                
                if self.c_clip_norm is not None:
                    if norm(c) > self.c_clip_norm:
                        c = c * (self.c_clip_norm/ norm(c))
                
                # Access present values of h and a_hat
                h = self.h_history.pop(0)
                a_hat = self.a_hat_history.pop(0)

                #Use to get gradients w.r.t. weights from credit assignment
                D = self.rnn.activation.f_prime(h)
                rec_grads += np.multiply.outer(c * D, a_hat)

                if i_BPTT == self.T_truncation - 1: #Skip if at end
                    continue

                #Use future-facing relation to backpropagate by one time step.
                q = self.q_history.pop(0)
                J = self.rnn.get_a_jacobian(h=h, update=False)
                c = q + c.dot(J)

            return rec_grads

        else:

            return np.zeros((self.n_h, self.m))

class Future_BPTT(Learning_Algorithm):
    """Implements the 'F-BPTT' version of backprop we discuss in the paper for
    an RNN.

    Although more expensive than E-BPTT by a factor of the truncation horizon,
    this version covers more 'loss-parmaeter sensitivity' terms in Fig. 3 and
    produces, at each time step, an approximate 'future-facing' gradient up to
    truncation that can be used for comparison with other algorithm's outputs.

    Details of computation are in paper. When a credit assignment estimate is
    calculated, the gradient is ultimately calculated according to

    dL/dW_{ij} = c_i \phi'(h_i) a_hat_j                             (1)."""

    def __init__(self, rnn, T_truncation, **kwargs):
        """Inits an instance of Future_BPTT by specifying the network to
        train and the truncation horizon. No default allowable kwargs."""

        self.name = 'F-BPTT'
        allowed_kwargs_ = set()
        super().__init__(rnn, allowed_kwargs_, **kwargs)

        self.T_truncation = T_truncation

        self.c_history = []
        self.a_hat_history = []
        self.h_history = []

    def update_learning_vars(self):
        """Updates the list of credit assignment vectors according to Section
        4.1.2 in the paper.

        First updates relevant history with latest network variables
        a_hat, h and q. Then backpropagates the latest q to each previous time
        step, adding the result to each previous credit assignment estimate."""

        #Update history
        self.a_hat_history.insert(0, np.concatenate([self.rnn.a_prev,
                                                     self.rnn.x,
                                                     np.array([1])]))
        self.h_history.insert(0, self.rnn.h)
        self.propagate_feedback_to_hidden()
        q = np.copy(self.q)
        #Add immediate credit assignment to front of list
        self.c_history.insert(0, q)

        #Loop over truncation horizon and backpropagate q, pausing along way to
        #update credit assignment estimates
        for i_BPTT in range(1, len(self.c_history)):

            h = self.h_history[i_BPTT - 1]
            J = self.rnn.get_a_jacobian(h=h, update=False)
            q = q.dot(J)
            self.c_history[i_BPTT] += q

    def get_rec_grads(self):
        """Removes the oldest credit assignment value from the c_history list
        and uses it to produce recurrent gradients according to Eq. (1).

        Note: for the first several time steps of the simulation, before
        self.c_history fills up to T_truncation size, 0s are returned for
        the recurrent gradients."""

        if len(self.c_history) >= self.T_truncation:

            #Remove oldest c, h and a_hat from lists
            c = self.c_history.pop(-1)
            h = self.h_history.pop(-1)
            a_hat = self.a_hat_history.pop(-1)

            #Implement Eq. (1)
            D = self.rnn.activation.f_prime(h)
            rec_grads = np.multiply.outer(c * D, a_hat)

        else:

            rec_grads = np.zeros((self.n_h, self.m))

        return rec_grads

    def reset_learning(self):
        """Resets learning by deleting network variable history."""

        self.c_history = []
        self.a_hat_history = []
        self.h_history = []

class KeRNL(Learning_Algorithm):
    """Implements the Kernel RNN Learning (KeRNL) algorithm from Roth et al.
    2019.

    Details in our review or original paper. Briefly, a matrix A of shape
    (n_h, n_h) and an eligibility trace B of shape (n_h, m) are both
    maintained to approximate the influence matrix as M_{kij} ~ A_{ki} B_{ij}.
    In addition, a set of n_h learned timescales \alpha_i are maintained such
    that da^{(t+ \Delta t)}_k/da^{(t)}_i ~ A_{ki} e^{-\alpha_i \Delta t}. These
    approximations are updated at every time step. B is updated by temporally
    filtering the immediate influence via the learned timescales

    B'_{ij} = (1 - \alpha_i) B_{ij} + \alpha \phi'(h_i) a_hat_j         (1)

    while A and \alpha are updated by SGD on their ability to predict
    perturbative effects. See Algorithm 1: Pseudocode on page 5 of Roth et al.
    2019 for details."""

    def __init__(self, rnn, optimizer, sigma_noise=0.00001, **kwargs):
        """Inits an instance of KeRNL by specifying the optimizer used to train
        the A and alpha values and a noise standard deviation for the
        perturbations.

        Args:
            optimizer (optimizers.Optimizer): An instance of the Optimizer class
                for the training of A and alpha.
            sigma_noise (float): Standard deviation for the values, sampled
                i.i.d. from a zero-mean Gaussian, used to perturb the network
                state to noisy_rnn and thus estimate target predictions for
                A and alpha.

        Keyword args:
            A (numpy array): Initial value of A matrix, must be of shape
                (n_h, n_h). Default is identity.
            B (numpy array): Initial value of B matrix, must be of shape
                (n_h, m). Default is zeros.
            alpha (numpy array): Initial value of learned timescales alpha,
                must be of shape (n_h). Default is all 0.8.
            Omega (numpy array): Initial value of filtered perturbations, must
                be of shape (n_h). Default is 0.
            Gamma (numpy array): Initial value of derivative of filtered
                perturbations, must be of shape (n_h). Default is 0.
            T_reset (int): Number of time steps between regular resets of
                perturbed network to network state and Omega, Gamma, B variables
                to 0. If unspecified, default is no resetting."""

        self.name = 'KeRNL'
        allowed_kwargs_ = {'A', 'B', 'alpha', 'Omega', 'Gamma', 'T_reset'}
        super().__init__(rnn, allowed_kwargs_, **kwargs)

        self.i_t = 0
        self.sigma_noise = sigma_noise
        self.optimizer = optimizer
        self.zeta = np.random.normal(0, self.sigma_noise, self.n_h)

        #Initialize learning variables
        if self.A is None:
            self.A = np.eye(self.n_h)
        if self.B is None:
            self.B = np.zeros((self.n_h, self.m))
        if self.alpha is None:
            self.alpha = np.ones(self.n_h) * 0.8
        if self.Omega is None:
            self.Omega = np.zeros(self.n_h)
        if self.Gamma is None:
            self.Gamma = np.zeros(self.n_h)

        #Initialize noisy network as copy of original network
        self.noisy_rnn = deepcopy(self.rnn)

    def update_learning_vars(self):
        """Updates the matrices A and B, which are ultimately used to drive
        learning. In the process also updates alpha, Omega, and Gamma."""

        #Reset learning variables if desired and on schedule to do so
        if self.T_reset is not None:
            if self.i_t % self.T_reset == 0:
                self.reset_learning()

        #Match noisy network's parameters to latest network parameters
        self.noisy_rnn.W_rec = self.rnn.W_rec
        self.noisy_rnn.W_in = self.rnn.W_in
        self.noisy_rnn.b_rec = self.rnn.b_rec

        #Run perturbed network forwards
        self.noisy_rnn.a += self.zeta
        self.noisy_rnn.next_state(self.rnn.x)

        #Update learning variables (see Pseudocode in Roth et al. 2019)
        self.kernel = np.maximum(0, 1 - self.alpha)
        self.zeta = np.random.normal(0, self.sigma_noise, self.n_h)
        self.Gamma = self.kernel * self.Gamma - self.Omega
        self.Omega = self.kernel * self.Omega + self.zeta

        #Update eligibility trace (Eq. 1)
        self.D = self.rnn.activation.f_prime(self.rnn.h)
        self.a_hat = np.concatenate([self.rnn.a_prev,
                                     self.rnn.x,
                                     np.array([1])])
        self.papw = self.rnn.alpha * np.multiply.outer(self.D, self.a_hat)
        self.B = (self.B.T * self.kernel).T + self.papw

        #Get error in predicting perturbations effect (see Pseudocode)
        self.error_prediction = self.A.dot(self.Omega)
        self.error_observed = self.noisy_rnn.a - self.rnn.a
        self.noise_loss = np.square(self.error_prediction -
                                    self.error_observed).sum()
        self.noise_error = self.error_prediction - self.error_observed

        #Update A and alpha (see Pseudocode)
        self.A_grads = np.multiply.outer(self.noise_error, self.Omega)
        self.alpha_grads = self.noise_error.dot(self.A) * self.Gamma
        params = [self.A, self. alpha]
        grads = [self.A_grads, self.alpha_grads]
        self.A, self.alpha = self.optimizer.get_updated_params(params, grads)

        self.i_t += 1

    def get_rec_grads(self):
        """Using updated A and B, returns recurrent gradients according to
        final line in Pseudocode table in Roth et al. 2019."""

        return (self.B.T * self.q.dot(self.A)).T

    def reset_learning(self):
        """Resets learning variables to 0 and resets the perturbed network
        to the state of the primary network."""

        self.noisy_rnn.a = np.copy(self.rnn.a)
        self.Omega = np.zeros_like(self.Omega)
        self.Gamma = np.zeros_like(self.Gamma)
        self.B = np.zeros_like(self.B)

class REINFORCE(Learning_Algorithm):
    def __init__(self, rnn, sigma = 0, **kwargs):
        """Inits an instance of REINFORCE by specifying the optimizer used to train
        the A and alpha values and a noise standard deviation for the
        perturbations.
        Args:
            optimizer (optimizers.Optimizer): An instance of the Optimizer class
            sigma_noise (float): Standard deviation for the values, sampled
                i.i.d. from a zero-mean Gaussian, used to perturb the network
                state to noisy_rnn and thus estimate target predictions for
                A and alpha.
        Keyword args:
            decay (numpy float): value of decay for the eligibility trace.
                Must be a value between 0 and 1, default is 0, indicating no decay
            loss_decay (numpy float): time constant of the filtered average of the activations"""

        self.name = 'REINFORCE'
        allowed_kwargs_ = {'decay', 'loss_decay'}
        super().__init__(rnn, allowed_kwargs_, **kwargs)
        #Initialize learning variables
        if self.decay is None:
            self.decay = 0
        if self.loss_decay is None:
            self.loss_decay = 0.01
        self.e_trace = 0
        self.loss_avg = 0
        self.loss_prev = 0
        self.loss = 0
        self.sigma = sigma
        
    def update_learning_vars(self):
        """Updates the eligibility traces used for learning"""
        #presynaptic variables/parameters
        
        self.a_hat = np.concatenate([self.rnn.a_prev, self.rnn.x, np.array([1])])
        #postsynaptic variables/parameters
        self.D = self.rnn.activation.f_prime(self.rnn.h) * self.rnn.noise
        
        #matrix of pre/post activations
        self.e_immediate = np.outer(self.D, self.a_hat)/self.sigma**2
        self.e_trace = (1-self.decay) * self.e_trace + self.e_immediate
        self.loss_prev = self.loss
        self.loss = self.rnn.loss_
        self.loss_avg = (1 - self.loss_decay) * self.loss_avg + self.loss_decay * self.loss_prev

    def get_rec_grads(self):
        """Combine the eligibility trace and the reward to get an estimate
        of the gradient"""
        return (self.loss - self.loss_avg) * self.e_trace
    
class List_of_Gradients(Learning_Algorithm):
    """Simply prescribe a series of updates to the network"""
    
    def __init__(self, rnn, grads_list_list, allowed_kwargs_=set(), **kwargs):
        """Initializes an instance of learning algorithm by specifying the
        network to be trained, custom allowable kwargs, and kwargs.

        Args:
            rnn (network.RNN): An instance of RNN to be trained by the network.
            allowed_kwargs_ (set): Set of allowed kwargs in addition to those
                common to all child classes of Learning_Algorithm, 'W_FB' and
                'L2_reg'."""

        allowed_kwargs = {}.union(allowed_kwargs_)
        
        self.rnn = rnn
        self.grads_list_list = grads_list






























