import numpy as np
import matplotlib.pyplot as plt


x_vals = np.random.uniform(low=0,high=1,size=100)
print(x_vals.shape)
t = [i**3 - i**2 for i in x_vals]
t_vals = np.array(t)

plt.scatter(x_vals,t)
# plt.show()


def ReLU(x):
    return np.maximum(0,x)

def ReLU_derivative(x):
    # CURRENTLY LINEAR. TURN INTO BOOLEAN
    return (x>0).astype(np.float64)

def mse(target, output): 
    return np.mean((target -output) ** 2)


def mse_derivative(target, output):
    return output - target

def liner(x):

    return x

def linar_derivation(x):
    return 1

class Layer():
    def __init__(self, n_inputs, n_units, act_func = ReLU, der_act_func = ReLU_derivative, learning_rate = 0.002, variance = 0.1) -> None:
        
        self.weights = np.random.normal(scale = variance, size = (n_inputs + 1,n_units))

        self.act_func = act_func
        self.der_act_func = der_act_func
        self.lear = learning_rate

        self.layer_input = None
        self.layer_preactivation = None
        self.layer_activation = None


    def __call__(self, inputs:np.array) :
        self.layer_input = np.insert(inputs,0,1) # bias in the weights

        self.layer_preactivation = self.layer_input.T @ self.weights
        self.layer_activation = self.act_func(self.layer_preactivation)
        return self.layer_activation

    def backward_step(self, delta):

        grad = self.der_act_func(self.layer_preactivation) * delta

        grad_weights = self.layer_input.reshape(-1,1) * grad

        delta = (self.weights * grad).sum(axis=1)
        delta = np.delete(delta,0) # remove bias from weights

        self.weights -= grad_weights * self.lear

        return delta



class MLP(object):
    def __init__ (self, n_inputs, layers : list ,lr = 0.0002):
        
        self.layer_list = []
        self.n_inputs = n_inputs

        last = n_inputs
        for n_units, act_func, dev_act_func in layers:

            self.layer_list.append(
                Layer(last,n_units,act_func,dev_act_func,learning_rate=lr)
            )
            last = n_units

    def __call__(self, x):
        
        for layer in self.layer_list:
            x = layer(x)

        return x

    def backward(self,delta):

        for layer in reversed(self.layer_list):
            delta = layer.backward_step(delta)

        



layers = [(50,ReLU,ReLU_derivative),(90,ReLU,ReLU_derivative),(1,linar,linar_derivation)]
m = MLP(1,layers)



def step(m, x_vals, y_vals,loss,dev_loss):
    
    mse = []

    for x,y in zip(x_vals,y_vals):

        pred = m(x.reshape(1,1))
        mse.append(loss(pred,y))

        delta = dev_loss(y,pred)

        m.backward(delta)

    return np.mean(mse)

learning = []
for i in range(10000):
    learning.append( step(m,x_vals,t_vals,mse,mse_derivative))
    
plt.scatter(x_vals,[m(x.reshape(1,1)) for x in x_vals])
# plt.plot(learning[10: ])
plt.show()
print(learning)