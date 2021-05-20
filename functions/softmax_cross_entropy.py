from functions.Function import Function
import numpy as np

### --- Define softmax cross-entropy --- ###

def softmax_cross_entropy_(z, y, epsilon=0.0001):

    p = softmax_(z)
    p = np.maximum(p, epsilon) #Regularize in case p is small

    return -y.dot(np.log(p))

def softmax_cross_entropy_derivative(z, y):

    return softmax_(z) - y

softmax_cross_entropy = Function(softmax_cross_entropy_,
                                 softmax_cross_entropy_derivative)