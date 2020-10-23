#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:01:00 2019

@author: omarschall
"""

import unittest
from optimizers import *
import numpy as np

class Test_SGD(unittest.TestCase):

    @classmethod
    def setUp(cls):

        cls.optimizer = Stochastic_Gradient_Descent(lr=0.1, clip_norm=2)

    def test_clip_norm(self):

        grads = [np.ones(2)*2, np.ones(2)]
        clipped_grads = self.optimizer.clip_gradient((grads))
        grad_norm = np.sqrt(10)
        correct_clipped_grads = [np.ones(2)*4/grad_norm,
                                 np.ones(2)*2/grad_norm]

        self.assertTrue(np.isclose(clipped_grads,
                                   correct_clipped_grads).all())

    def test_update(self):

        params = [np.ones(2)]
        grads = [np.ones(2)]
        updated_params = self.optimizer.get_updated_params(params, grads)
        correct_updated_params = [np.ones(2)*0.9]

        self.assertTrue(np.isclose(updated_params,
                                   correct_updated_params).all())

    def test_lr_decay(self):

        optimizer = Stochastic_Gradient_Descent(lr=1, lr_decay_rate=0.9, min_lr=0.5)

        params = []
        grads = []

        for _ in range(3):
            params = optimizer.get_updated_params(params, grads)

        correct_lr = 0.9**3
        self.assertEqual(correct_lr, optimizer.lr)

        for _ in range(4):
            params = optimizer.get_updated_params(params, grads)

        correct_lr = 0.5
        self.assertEqual(correct_lr, optimizer.lr)

if __name__ == '__main__':
    unittest.main()
