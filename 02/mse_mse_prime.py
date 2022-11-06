"""Loss function and it's derivative"""

import numpy as np

def mse(y_pred, y_true):
    return np.power(y_pred-y_true, 2) / 2

def mse_prime(y_pred, y_true):
    return y_true - y_pred