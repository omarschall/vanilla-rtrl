#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:29:15 2019

@author: omarschall
"""

import numpy as np
import unittest
from network import *
from numpy.testing import assert_allclose

class Test_Network(unittest.TestCase):
    """Tests methods from the network.py module."""

    @classmethod
    def setUpClass(cls):
        """Initializes a simple instance of network for testing."""

        n_in     = 2
        n_hidden = 8
        n_out    = 2

        W_in  = np.ones((n_hidden, n_in))
        W_rec = np.eye(n_hidden)
        W_out = np.ones((n_out, n_hidden))

        b_rec = np.ones(n_hidden)
        b_out = np.ones(n_out)

        alpha = 0.6

        cls.rnn = RNN(W_in, W_rec, W_out, b_rec, b_out,
                      activation=tanh,
                      alpha=alpha,
                      output=softmax,
                      loss=softmax_cross_entropy)

    def test_reset_network(self):
        """Verifies that the reset_network method works in both specified
        and random cases."""

        self.rnn.reset_network()
        self.assertTrue(np.isclose(self.rnn.activation.f(self.rnn.h),
                                   self.rnn.a).all())

        self.rnn.reset_network(h=np.ones(self.rnn.n_h))
        self.assertTrue(np.isclose(self.rnn.h,
                                   np.ones(self.rnn.n_h)).all())
        self.assertTrue(np.isclose(self.rnn.a,
                                   np.array([np.tanh(1)]*self.rnn.n_h)).all())

        #Make sure sigma method and manual assignment work for the same random
        #seed.
        np.random.seed(1)
        h = np.random.normal(0, 0.5, (self.rnn.n_h))
        self.rnn.reset_network(h=h)
        a1 = np.copy(self.rnn.a)
        np.random.seed(1)
        self.rnn.reset_network(sigma=0.5)
        a2 = np.copy(self.rnn.a)
        self.assertTrue(np.isclose(a1, a2).all())

    def test_next_state(self):
        """Verfies that next_state function works in 'update' case."""

        self.rnn.reset_network(h=np.ones(self.rnn.n_h))
        self.rnn.next_state(x=np.zeros(self.rnn.n_in))
        #Calculate the correct next state
        a_prev = np.array([np.tanh(1)]*self.rnn.n_h)
        a = ((1 - self.rnn.alpha)*a_prev +
             self.rnn.alpha*np.tanh(a_prev + np.ones(self.rnn.n_h)))
        #Compare with update from next_state
        self.assertTrue(np.isclose(self.rnn.a, a).all())

    def test_z_out(self):
        """Verifies that z_out produces correct output in a special case."""

        self.rnn.reset_network(h=np.ones(self.rnn.n_h))
        self.rnn.z_out()
        #Calculate the correct output
        z = np.array([np.sum([np.tanh(1)]*self.rnn.n_h) + 1]*2)
        #Compare with update from z_out
        self.assertTrue(np.isclose(self.rnn.z, z).all())

    def test_get_a_jacobian(self):
        """Verifies that get_a_jacobian produces correct output in a special
        case."""

        self.rnn.reset_network(h=np.ones(self.rnn.n_h))
        self.rnn.get_a_jacobian()
        J = np.diag([0.6*self.rnn.activation.f_prime(1) + 0.4]*self.rnn.n_h)
        self.assertTrue(np.isclose(J, self.rnn.a_J).all())

if __name__=='__main__':
    unittest.main()