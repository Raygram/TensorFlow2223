"""ReLU activation function and it's derivative"""

def relu(x):
    return x * (x > 0)

def relu_prime(x):
    return 1 * (x > 0)