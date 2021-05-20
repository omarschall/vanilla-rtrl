from functions.Function import Function
import numpy as np

### --- Define sigmoid cross entropy --- ###

def sigmoid_cross_entropy_(z, y):

    p = sigmoid.f(z)

    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def sigmoid_cross_entropy_derivative(z, y):

    p = sigmoid.f(z)

    return (-y / p + (1 - y) / (1 - p)) * sigmoid.f_prime(z)

sigmoid_cross_entropy = Function(sigmoid_cross_entropy_,
                                 sigmoid_cross_entropy_derivative)