from functions.Function import Function
import numpy as np

### --- Define softmax --- ###

def softmax_(z):

    z = z - np.amax(z)

    return np.exp(z) / np.sum(np.exp(z))

def softmax_derivative(z):

    z = z - np.amax(z)

    return np.multiply.outer(softmax_(z), 1 - softmax_(z))

softmax = Function(softmax_, softmax_derivative)