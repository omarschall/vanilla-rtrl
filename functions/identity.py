from functions.Function import Function
import numpy as np

### --- Define Identity --- ###

def identity_(z):

    return z

def identity_derivative(z):

    return np.ones_like(z)

identity = Function(identity_, identity_derivative)