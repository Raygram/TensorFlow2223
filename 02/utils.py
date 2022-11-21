import numpy as np
import matplotlib.pyplot as plt


def ReLU(x):
    return np.maximum(0,x).astype(np.float64)

def ReLU_derivative(x):
    return (x>0).astype(np.float64)

def linear(x):
    return x.astype(np.float64)

def linear_derivative(x):
    return np.ones(x.shape,dtype=np.float64)

# Loss Functions
def mse(target, prediction): 
    return np.mean((target - prediction) ** 2).astype(np.float64)

def mse_derivative(target, prediction):
    return (prediction - target).astype(np.float64)




class Layer():
    def __init__(self, n_inputs, n_units, act_func = ReLU, der_act_func = ReLU_derivative, learning_rate = 0.002, variance = 0.25) -> None:
        # initialise random weights with variance to control data complexity
        self.weights = np.random.normal(scale = variance, size = (n_inputs + 1,n_units)).astype(np.float64) # shape is add one row to for the biases
        # set attributes
        self.act_func = act_func
        self.der_act_func = der_act_func
        self.lr = learning_rate
        # dummy variables
        self.layer_input = None
        self.layer_preactivation = None
        self.layer_activation = None


    # Forward Step 
    def __call__(self, inputs:np.array) :        # defined in __call__ function, since it is faster 
        self.layer_input = np.insert(inputs,0,1) # insert a 1 since bias is included in weight matrix
        self.layer_preactivation = self.layer_input.T @ self.weights    # transpose input and mulptiply by weight matrix
        self.layer_activation = self.act_func(self.layer_preactivation) # apply activation function 
        return self.layer_activation

    def backward_step(self, delta):
        # get gradient by multiplying derivative of activation function with delta 
        grad = self.der_act_func(self.layer_preactivation) * delta
        # get gradient of weights by multiplying correctly shaped input with gradient
        grad_weights = self.layer_input.reshape(-1,1) * grad
        # update delta by taking sum of weights multiplied with the gradient
        delta = (self.weights * grad).sum(axis=1)
        delta = np.delete(delta,0) # remove bias entry from delta
        
        # update weights
        self.weights -= grad_weights * self.lr


        return delta



class MLP(object):
    def __init__ (self, n_inputs, layers : list ,lr = 0.0002,variance = 0.25):
        
        self.layer_list = []        # keeping track of layers
        self.n_inputs = n_inputs    

        last = n_inputs             # get number of inputs from previous layer

        # iterate over all layer-tuples which consist of n_units, an activation function and its derivative
        for n_units, act_func, der_act_func in layers:
            # add the corersponding layers to list
            self.layer_list.append(
                Layer(n_inputs=last, n_units=n_units, act_func=act_func, der_act_func=der_act_func, learning_rate=lr,variance=variance)
            )
            last = n_units # get next layers inputs by taking this layers outputs

    def __call__(self, x):
        # iterate over layers and call forward pass their outputs recursively to next layer in list
        for layer in self.layer_list:                                           
            x = layer(x)    # layer(x) returns the activation of layer for input x
        
        return x    # last x obtained is end result

    def backward(self,delta):
        # go through layer list backwards
        for layer in reversed(self.layer_list):
            delta = layer.backward_step(delta)  # applay backward step reusing previous delta

        


def step(m, x_vals, y_vals,loss,der_loss):
    mse = []
    # iterate over targets and predictions
    for i,(x,y) in enumerate(zip(x_vals,y_vals)):

        pred = m(x.reshape(1,1))    # get prediction
        mse.append(loss(pred,y))    # get loss (calculated by mse)

        delta = der_loss(y,pred)    # derivative of loss function applied to target and prediction 

        m.backward(delta)           # backpropagate

    return np.mean(mse)             # get the mean of the squared errors

