from functions.Function import Function
import numpy as np

### --- Define softplus --- ###

def softplus_(z):

    return np.log(1 + np.exp(z))

def softplus_derivative(z):

    return sigmoid_(z)

softplus = Function(softplus_, softplus_derivative)