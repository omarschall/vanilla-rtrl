from functions.Function import Function
import numpy as np

### --- Define tanh --- ###

def tanh_(z):

    return np.tanh(z)

def tanh_derivative(z):

    return 1 - np.tanh(z) ** 2

tanh = Function(tanh_, tanh_derivative)