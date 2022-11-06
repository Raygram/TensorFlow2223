import numpy as np
import matplotlib.pyplot as plt
#import class_layer

x = np.random.random(100)
t = np.zeros(100)

for i in range(len(x)):
    t[i] = x[i]**3-x[i]**2

print(x,t)

# plt.scatter(x,x)
# plt.show()

epochs = 1000

for i in range(epochs):
    

