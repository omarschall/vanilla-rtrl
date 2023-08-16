import torch
from torch import nn

class Torch_RNN(nn.Module):
    """Module for a RNN that can be easier trained in torch."""

    def __init__(self, W_in, W_rec, W_out, b_rec, b_out,
                 activation='tanh', alpha=1):
        """Takes in 5 numpy arrays for initial parameter values, a string argument
        for 'tanh' or 'relu' and an inverse time constant alpha."""

        super().__init__()

        self.n_in = W_in.shape[1]
        self.n_h = W_rec.shape[0]
        self.n_out = W_out.shape[0]

        self.W_rec = torch.nn.Parameter(torch.tensor(W_rec, requires_grad=True, dtype=torch.float32))
        self.W_in = torch.nn.Parameter(torch.tensor(W_in, requires_grad=True, dtype=torch.float32))
        self.b_rec = torch.nn.Parameter(torch.tensor(b_rec, requires_grad=True, dtype=torch.float32))
        self.W_out = torch.nn.Parameter(torch.tensor(W_out, requires_grad=True, dtype=torch.float32))
        self.b_out = torch.nn.Parameter(torch.tensor(b_out, requires_grad=True, dtype=torch.float32))

        if activation == 'tanh':
            self.activation = torch.tanh
            self.activation_derivative = lambda x: 1 - torch.square(torch.tanh(x))
        elif activation == 'relu':
            self.activation = torch.relu
            self.activation_derivative = lambda x: torch.where(x > 0, torch.tensor(1.), torch.tensor(0.))
        self.activation_name = activation
        self.alpha = alpha

    def forward(self, state, X):
        """Network propagates one time step forward using *left-handed* matrix multiplication,
        so batch dimension is first dimension. Same formula as in core.RNN."""

        self.state_prev = state.clone().detach()
        self.h = state.matmul(self.W_rec.T) + X.matmul(self.W_in.T) + self.b_rec
        state = (1 - self.alpha) * state + self.alpha * self.activation(self.h)
        output = state.matmul(self.W_out.T) + self.b_out
        return state, output

    def compute_loss(self, output, label):
        """Hard-coded in identity-MSE loss."""

        return torch.mean(torch.square(output - label))