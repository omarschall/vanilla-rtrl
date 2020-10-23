#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:19:07 2019

@author: omarschall
"""

import numpy as np
import unittest
from gen_data import *

class Test_Gen_Data(unittest.TestCase):
    """Tests methods from the network.py module."""

    @classmethod
    def setUpClass(cls):
        """Initializes a simple instance of network for testing."""

        pass

    def test_add_task(self):
        """Creates datasets for tau_task = 1 and tau_task > 1 cases and ensures
        that the defining equation holds for input-outputs sufficiently far
        away from the beginning."""

        task = Add_Task(6, 10, deterministic=True, tau_task=1)
        data = task.gen_data(40, 0)

        for i in range(12, 25):
            y = (0.5 +
                 0.5*data['train']['X'][i-6, 0] -
                 0.25*data['train']['X'][i-10, 0])
            self.assertEqual(data['train']['Y'][i, 0], y)

        task = Add_Task(6, 10, deterministic=True, tau_task=2)
        data = task.gen_data(50, 0)

        for i in range(25, 40):
            if i%2 == 1:
                x1 = data['train']['X'][i, 0]
                x2 = data['train']['X'][i-1, 0]
                self.assertEqual(x1, x2)
                y = (0.5 +
                     0.5*data['train']['X'][i-12, 0] -
                     0.25*data['train']['X'][i-20, 0])
                self.assertEqual(data['train']['Y'][i, 0], y)
            if i%2 == 0:
                x1 = data['train']['X'][i, 0]
                x2 = data['train']['X'][i+1, 0]
                self.assertEqual(x1, x2)
                y = (0.5 +
                     0.5*data['train']['X'][i-11, 0] -
                     0.25*data['train']['X'][i-19, 0])
                self.assertEqual(data['train']['Y'][i, 0], y)

    def test_mimic_task(self):
        """Verifies that the proper RNN output is returned as label in a simple
        case where the RNN simply counts the number of time steps."""

        from network import RNN
        from functions import identity, mean_squared_error

        n_in = 2
        n_h = 2
        n_out = 2

        W_in_target = np.eye(n_in)
        W_rec_target = np.eye(n_h)
        W_out_target = np.eye(n_out)
        b_rec_target = np.zeros(n_h)
        b_out_target = np.zeros(n_out)

        alpha = 1

        rnn_target = RNN(W_in_target, W_rec_target, W_out_target,
                         b_rec_target, b_out_target,
                         activation=identity,
                         alpha=alpha,
                         output=identity,
                         loss=mean_squared_error)

        task = Mimic_RNN(rnn_target, p_input=1, tau_task=1)
        data = task.gen_data(100, 0)

        y = np.arange(1, 101)
        y_correct = np.array([y, y]).T

        self.assertTrue(np.isclose(data['train']['Y'],
                                   y_correct).all())

    def test_sensorimotor_mapping(self):
        """Verifies that trials all look like one of the template trials."""

        task = Sensorimotor_Mapping(1, 1, 2, 1)
        data = task.gen_data(12, 0)

        x_1 = np.array([[0, 1, 0], [0, 0, 1]]).T
        x_2 = np.array([[0, -1, 0], [0, 0, 1]]).T
        y_1 = np.array([[0.5, 0.5, 1], [0.5, 0.5, 0]]).T
        y_2 = np.array([[0.5, 0.5, 0], [0.5, 0.5, 1]]).T

        for i in range(4):
            x = data['train']['X'][3 * i : 3 * (i + 1)]
            y = data['train']['Y'][3 * i : 3 * (i + 1)]

            self.assertTrue(np.isclose(x, x_1).all() or
                            np.isclose(x, x_2).all())

            self.assertTrue(np.isclose(y, y_1).all() or
                            np.isclose(y, y_2).all())

    def test_sequential_mnist(self):

        try:
            task = Sequential_MNIST(28)
        except FileNotFoundError:
            return
        data = task.gen_data(100, 0)

        self.assertTrue(len(data['train']['X']) == 84)
        for i in range(27):
            self.assertTrue(np.isclose(data['train']['Y'][i,:],
                                       data['train']['Y'][i+1,:]).all())
        for i in range(28, 55):
            self.assertTrue(np.isclose(data['train']['Y'][i,:],
                                       data['train']['Y'][i+1,:]).all())

    def test_flip_flop(self):

        task = Flip_Flop_Task(3, 0)
        data = task.gen_data(40, 0)
        self.assertTrue((data['train']['Y'] == 0).all())

        task = Flip_Flop_Task(3, 1)
        data = task.gen_data(40, 0)
        self.assertTrue((data['train']['Y'][1:] ==
                         data['train']['X'][1:]).all())




if __name__=='__main__':
    unittest.main()