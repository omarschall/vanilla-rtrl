from functions.Function import Function
import numpy as np

### --- Define ReLu --- ###

right_slope = 1
left_slope = 0
def relu_(h):

    return np.maximum(0, right_slope * h) - np.maximum(0, left_slope * (-h))

def relu_derivative(h):

    return (h > 0) * (right_slope - left_slope) + left_slope

relu = Function(relu_, relu_derivative)