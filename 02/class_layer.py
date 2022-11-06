import numpy as np
from mse_mse_prime import *
from relu_relu_prime import *

class Layer:

    learning_rate = 0.04

    def __init__(self, n_units, input_units):
        self.n_units = n_units
        self.input_units = input_units
        self.bias = np.zeros((1, n_units))
        self.weights = np.random.rand(input_units, n_units)

        self.input = None
        self.preactivation = None
        self.activation = None

    def set_input(self, input):
        self.input = input
    
    def set_preactivation(self, preactivation):
        self.preactivation = preactivation
    
    def set_activation(self, activation):
        self.activation = activation

    def forward_step(self):
        net_input = self.input @ self.weights + self.bias  

        # ReLU activation
        self.activation = relu(net_input)
    
        return self.activation
        
    def backward_step(self):
        # if output layer
        if self.n_units = 1:
            dL_da = mse_prime(self.activation)
            da_dd = relu_prime(self.preactivation)
            
            dL_dW = np.transpose(self.input) @ np.multiply(da_dd, dL_da)
            dL_db = np.multiply(da_dd, dL_da)
        else:
            #? How for other layers? l+1 

        # updating weights and biases
        self.weights = self.weights - self.learning_rate*dL_dW
        self.bias = self.bias - self.learning_rate*dL_db


# l1 = Layer(2, 8)
# print(l1.bias, "\n\n", l1.weights)

# # inp = np.random.rand(1,8)
# inp = np.random.randint(10, size=(1,8))
# print(l1.forward_step(inp))


