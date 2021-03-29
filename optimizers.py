#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from utils import split_weight_matrix

class Optimizer:
    """Parent class for gradient-based optimizers."""

    def __init__(self, allowed_kwargs_, **kwargs):

        allowed_kwargs = {'lr_decay_rate', 'min_lr',
                          'clip_norm', 'normalize',
                          'rec_proj_mats', 'out_proj_mats'}.union(allowed_kwargs_)
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to Optimizer: ' + str(k))

        #Set all non-specified kwargs to None
        for attr in allowed_kwargs:
            if not hasattr(self, attr):
                setattr(self, attr, None)

        self.__dict__.update(kwargs)

    def clip_gradient(self, grads):
        """Clips each gradient by the global gradient norm if it exceeds
        self.clip_norm.

        Args:
            grads (list): List of original gradients
        Returns:
            clipped_grads (list): List of clipped gradients."""

        grad_norm = np.sqrt(sum([np.square(grad).sum() for grad in grads]))
        if grad_norm > self.clip_norm:
            clipped_grads = []
            for grad in grads:
                clipped_grads.append(grad * (self.clip_norm/grad_norm))
            return clipped_grads
        else:
            return grads
        
    def normalize_gradient(self, grads):
        
        grad_norm = np.sqrt(sum([np.square(grad).sum() for grad in grads]))
        normalized_grads = []
        for grad in grads:
            normalized_grads.append(grad / grad_norm)
        return normalized_grads

    def lr_decay(self):
        """Multiplicatively decays the learning rate by a factor of
        self.lr_decay_rate, with a floor learning rate of self.min_lr."""

        self.lr_ = self.lr_ * self.lr_decay_rate
        try:
            return np.max([self.lr_, self.min_lr])
        except AttributeError:
            return self.lr_
        
    def calculate_final_updates(self, param_changes):
        
        n_h = grads[0].shape[0]
        n_in = grads[1].shape[1]
            
        #Concatenate gradients in relevant directions
        W_grad = np.concatenate([grads[0], grads[1],
                                 grads[2].reshape(-1,1)], axis=1)
        W_out_grad = np.concatenate([grads[3], grads[4].reshape(-1,1)], axis=1)
        
        #Project along projection matrices
        W_grad_proj = self.rec_proj_mats[0].dot(W_grad)
        W_grad_proj = W_grad_proj.dot(self.rec_proj_mats[1])
        W_out_grad_proj = self.out_proj_mats[0].dot(W_out_grad)
        W_out_grad_proj = W_out_grad_proj.dot(self.out_proj_mats[1])
        
        delta_W = split_weight_matrix(W_grad_proj, [n_h, n_in, 1])
        delta_W += split_weight_matrix(W_out_grad_proj, [n_h, 1])
        
        updated_params = []
        for param, param_change in zip(params, param_changes):
            updated_params.append(param + param_change)

        return updated_params

class Adam(Optimizer):
    
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., **kwargs):
        
        allowed_kwargs_ = set()
        super().__init__(allowed_kwargs_, **kwargs)
        
        self.iterations = 0
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.decay = decay
        self.epsilon = epsilon
        self.initial_decay = decay

    def get_updated_params(self, params, grads):
        """ params and grads are list of numpy arrays
        """
        original_shapes = [x.shape for x in params]
        params = [x.flatten() for x in params]
        grads = [x.flatten() for x in grads]
        
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = np.sqrt(sum([np.sum(np.square(g)) for g in grads]))
            grads = [self.clip_norm(g, norm) for g in grads]
            
        '''
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        '''
        
        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1
        lr_t = lr * (np.sqrt(1. - np.power(self.beta_2, t)) /
                     (1. - np.power(self.beta_1, t)))

        if not hasattr(self, 'ms'):
            self.ms = [np.zeros(p.shape) for p in params]
            self.vs = [np.zeros(p.shape) for p in params]
    
        ret = [None] * len(params)
        for i, p, g, m, v in zip(range(len(params)), params, grads, self.ms, self.vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * np.square(g)
            p_t = p - lr_t * m_t / (np.sqrt(v_t) + self.epsilon)
            self.ms[i] = m_t
            self.vs[i] = v_t
            ret[i] = p_t
        
        self.iterations += 1
        
        for i in range(len(ret)):
            ret[i] = ret[i].reshape(original_shapes[i])
        
        return ret

class Stochastic_Gradient_Descent(Optimizer):
    """Implements basic stochastic gradient descent optimizer.

    Attributes:
        lr (float): learning rate."""

    def __init__(self, lr=0.001, **kwargs):

        allowed_kwargs_ = set()
        super().__init__(allowed_kwargs_, **kwargs)

        self.lr_ = np.copy(lr)
        self.lr = lr

    def get_updated_params(self, params, grads):
        """Returns a list of updated parameter values (NOT the change in value).

        Args:
            params (list): List of trainable parameters as numpy arrays
            grads (list): List of corresponding gradients as numpy arrays.
        Returns:
            updated_params (list): List of newly updated parameters."""

        if self.lr_decay_rate is not None:
            self.lr = self.lr_decay()

        if self.clip_norm is not None:
            grads = self.clip_gradient(grads)
            
        if self.normalize:
            grads = self.normalize_gradient(grads)
            
        updated_params = []
        for param, grad in zip(params, grads):
            updated_params.append(param - self.lr * grad)

        return updated_params

class SGD_Momentum(Optimizer):
    """Impelements SGD with classical momentum."""
    
    def __init__(self, lr=0.001, mu=0.8, **kwargs):

        allowed_kwargs_ = set()
        super().__init__(allowed_kwargs_, **kwargs)

        self.lr_ = np.copy(lr)
        self.lr = lr
        self.mu = mu
        self.vel = None
        
    def get_updated_params(self, params, grads):
        """Returns a list of updated parameter values (NOT the change in value).

        Args:
            params (list): List of trainable parameters as numpy arrays
            grads (list): List of corresponding gradients as numpy arrays.
        Returns:
            updated_params (list): List of newly updated parameters."""

        if self.lr_decay_rate is not None:
            self.lr = self.lr_decay()

        if self.clip_norm is not None:
            grads = self.clip_gradient(grads)
            
        if self.normalize:
            grads = self.normalize_gradient(grads)

        if self.vel is None:
            self.vel = [np.zeros_like(g) for g in params]

        self.vel = [self.mu * v - self.lr * g for v, g in zip(self.vel, grads)]

        updated_params = []
        for param, v in zip(params, self.vel):
            updated_params.append(param + v)

        return updated_params
    
class SGD_With_Projections(Optimizer):
    """Implements SGD with specified matrix operations for updates. Only works
    for Vanilla RNNs."""
    
    def __init__(self, lr=0.001, rec_proj_mats=[], out_proj_mats=[], **kwargs):

        allowed_kwargs_ = set()
        super().__init__(allowed_kwargs_, **kwargs)

        self.lr_ = np.copy(lr)
        self.lr = lr
        self.rec_proj_mats = rec_proj_mats
        self.out_proj_mats = out_proj_mats

    def get_updated_params(self, params, grads):
        """Returns a list of updated parameter values (NOT the change in value).

        Args:
            params (list): List of trainable parameters as numpy arrays
            grads (list): List of corresponding gradients as numpy arrays.
        Returns:
            updated_params (list): List of newly updated parameters."""

        if self.lr_decay_rate is not None:
            self.lr = self.lr_decay()

        if self.clip_norm is not None:
            grads = self.clip_gradient(grads)
            
        if self.normalize:
            grads = self.normalize_gradient(grads)
            
        n_h = grads[0].shape[0]
        n_in = grads[1].shape[1]
            
        #Concatenate gradients in relevant directions
        W_grad = np.concatenate([grads[0], grads[1],
                                 grads[2].reshape(-1,1)], axis=1)
        W_out_grad = np.concatenate([grads[3], grads[4].reshape(-1,1)], axis=1)
        
        #Project along projection matrices
        W_grad_proj = self.rec_proj_mats[0].dot(W_grad)
        W_grad_proj = W_grad_proj.dot(self.rec_proj_mats[1])
        W_out_grad_proj = self.out_proj_mats[0].dot(W_out_grad)
        W_out_grad_proj = W_out_grad_proj.dot(self.out_proj_mats[1])
        
        delta_W = split_weight_matrix(W_grad_proj, [n_h, n_in, 1])
        delta_W += split_weight_matrix(W_out_grad_proj, [n_h, 1])
        
        updated_params = []
        for param, delta_w in zip(params, delta_W):
            updated_params.append(param - self.lr * delta_w)

        return updated_params
    
class SGD_Momentum_With_Projections(Optimizer):
    """Impelements SGD with classical momentum."""
    
    def __init__(self, lr=0.001, mu=0.8, rec_proj_mats=[], out_proj_mats=[],
                 **kwargs):

        allowed_kwargs_ = set()
        super().__init__(allowed_kwargs_, **kwargs)

        self.lr_ = np.copy(lr)
        self.lr = lr
        self.mu = mu
        self.vel = None
        self.rec_proj_mats = rec_proj_mats
        self.out_proj_mats = out_proj_mats
        
    def get_updated_params(self, params, grads):
        """Returns a list of updated parameter values (NOT the change in value).

        Args:
            params (list): List of trainable parameters as numpy arrays
            grads (list): List of corresponding gradients as numpy arrays.
        Returns:
            updated_params (list): List of newly updated parameters."""

        if self.lr_decay_rate is not None:
            self.lr = self.lr_decay()

        if self.clip_norm is not None:
            grads = self.clip_gradient(grads)
            
        if self.normalize:
            grads = self.normalize_gradient(grads)

        if self.vel is None:
            self.vel = [np.zeros_like(g) for g in params]        

        n_h = grads[0].shape[0]
        n_in = grads[1].shape[1]
            
        #Concatenate gradients in relevant directions
        W_grad = np.concatenate([grads[0], grads[1],
                                 grads[2].reshape(-1,1)], axis=1)
        W_out_grad = np.concatenate([grads[3], grads[4].reshape(-1,1)], axis=1)
        
        #Project along projection matrices
        W_grad_proj = self.rec_proj_mats[0].dot(W_grad)
        W_grad_proj = W_grad_proj.dot(self.rec_proj_mats[1])
        W_out_grad_proj = self.out_proj_mats[0].dot(W_out_grad)
        W_out_grad_proj = W_out_grad_proj.dot(self.out_proj_mats[1])
        
        delta_W = split_weight_matrix(W_grad_proj, [n_h, n_in, 1])
        delta_W += split_weight_matrix(W_out_grad_proj, [n_h, 1])
        
        self.vel = [self.mu * v - self.lr * g for v, g in zip(self.vel, delta_W)]

        updated_params = []
        for param, v in zip(params, self.vel):
            updated_params.append(param + v)

        return updated_params
    
    
    
    