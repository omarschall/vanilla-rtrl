#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 15:03:11 2018

@author: yanqi
"""

import numpy as np
from pdb import set_trace
from utils import *
from functions import *
from copy import deepcopy
from learning_algorithms import *

class Meta_Learning_Algorithm(Stochastic_Algorithm):
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
    def __init__(self, rnn, inner_algo, **kwargs):
        """Inits an Meta instance by setting the initial dadw matrix to zero."""

        self.name = 'Meta' #Algorithm name
        allowed_kwargs_ = set() #No special kwargs for RTRL
        super().__init__(rnn, allowed_kwargs_, **kwargs)

        #Initialize alpha and beta
        self.alpha = np.random.normal(0, 1, (self.m,self.m))
        self.beta = np.random.normal(0, 1, (self.n_h,self.n_h))

        self.inner_algo = inner_algo
    

    def approximate_H_first_term(self):
        """Calculate the first term in the Hessian"""

        alpha1 =  np.outer(self.inner_algo.A, self.inner_algo.A)
        WB = np.dot(self.rnn.W_out,self.inner_algo.B)
        beta1 = np.dot(WB,WB.T)
        return alpha1, beta1

    def approximate_H_second_term(self):
        """Calculate the second term in the Hessian"""

        self.a_hat = np.concatenate([self.rnn.a_prev,
                                     self.rnn.x,
                                     np.array([1])])

        alpha2 = np.outer(self.a_hat,self.a_hat)
        beta2 = np.diag(self.q * self.rnn.activation.f_pp(self.rnn.h))

        return alpha2,beta2

    def approximate_meta_gradient(self,lam):
        
        # get Hessian components
        F1, G1 = approximate_H_first_term()
        F2, G2 = approximate_H_second_term()

        F1_alpha = np.dot(F1,self.alpha)
        G1_beta = np.dot(G1,self.beta)
        F2_alpha = np.dot(F2,self.alpha)
        G2_beta = np.dot(G2,self.beta)

        B_diag = np.diag(self.q.dot(self.inner_algo.B))
        A_diag = np.diag(self.inner_algo.A)

        self.nu = self.sample_nu()

        p0 = np.sqrt(norm(self.alpha)/norm(self.beta))
        p1 = np.sqrt(norm(G1_beta)/norm(F1_alpha))
        p2 = np.sqrt(norm(G2)/norm(F2))
        p3 = np.sqrt(norm(A_diag)/norm(B_diag))
        
        self.alpha = self.nu[0] * p0 * self.alpha \
                    + v_i*(self.nu[1] * p1 * F1_alpha \
                    + self.nu[2] * p2 * F2_alpha)\
                    + self.nu[3] * p3 * A_diag
        self.beta = self.nu[0] * (1/p0) * self.beta \
                    + np.dot(v_i,lam) *(self.nu[1] * (1/p1) * G1_beta  \
                    + self.nu[2] * (1/p2) * G2_beta) \
                    + self.nu[3] * (1/p3) * B_diag
        
    def update_learning_vars(self):
        
        """Get the new values of alpha and beta"""
        pass
    
    def get_rec_grads(self):
        
        """Get the LR gradients in an array of shape [n_h, m]"""
        
        pass
    
    def __call__(self):
        
        """Return gradients for LR in a list of arrays"""
        
        #let's hard code in the "outer grads" as 0 for now
        self.outer_grads = np.zeros((self.rnn.n_h, self.rnn.n_h + 1))
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
        
