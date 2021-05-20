from functions.Function import Function
import numpy as np

### --- Define Mean-Squared Error --- ###

def mean_squared_error_(z, y):

    return 0.5*np.square(z - y).mean()

def mean_squared_error_derivative(z, y):

    return z - y

mean_squared_error = Function(mean_squared_error_,
                              mean_squared_error_derivative)