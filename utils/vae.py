#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 4 17:33:30 2020

@author: omarschall
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dims, z_dim):

        super().__init__()
        
        self.n_layers = len(hidden_dims)
        self.linear_0 = nn.Linear(input_dim, hidden_dims[0])
        for i in range(1, self.n_layers):
            setattr(self, 'linear_{}'.format(i), nn.Linear(hidden_dims[i-1],
                                                           hidden_dims[i]))
        self.mu = nn.Linear(hidden_dims[-1], z_dim)
        self.var = nn.Linear(hidden_dims[-1], z_dim)

    def forward(self, x):
        
        # x is of shape [batch_size, input_dim]
        for i in range(self.n_layers):
            layer_linear = getattr(self, 'linear_{}'.format(i))
            x = F.tanh(layer_linear(x))
        # hidden is of shape [batch_size, hidden_dim]
        z_mu = self.mu(x)
        # z_mu is of shape [batch_size, latent_dim]
        z_var = self.var(x)
        # z_var is of shape [batch_size, latent_dim]

        return z_mu, z_var

class Decoder(nn.Module):

    def __init__(self, z_dim, hidden_dims, output_dim):

        super().__init__()

        self.n_layers = len(hidden_dims)
        self.linear_0 = nn.Linear(z_dim, hidden_dims[0])
        for i in range(1, self.n_layers):
            setattr(self, 'linear_{}'.format(i), nn.Linear(hidden_dims[i-1],
                                                           hidden_dims[i]))
        self.out = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        # x is of shape [batch_size, latent_dim]

        for i in range(self.n_layers):
            layer_linear = getattr(self, 'linear_{}'.format(i))
            x = F.tanh(layer_linear(x))
        # hidden is of shape [batch_size, hidden_dim]

        predicted = self.out(x)
        # predicted is of shape [batch_size, output_dim]

        return predicted

class VAE(nn.Module):

    def __init__(self, enc, dec):
        super().__init__()

        self.enc = enc
        self.dec = dec

    def forward(self, x):
        # encode
        z_mu, z_var = self.enc(x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        # decode
        predicted = self.dec(x_sample)
        return predicted, z_mu, z_var