import numpy as np
import matplotlib.pyplot as plt



def step_function(x):
    y = x > 0
    return y.astype(np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

## graph of activatioin functions
# x = np.arange(-5.0, 5.0, 0.1)

# ## step function graph
# y = step_function(x)

# ## sigmoid function graph
# z = sigmoid(x)

# plt.title('activation functions')

# plt.ylim([-0.1, 1.1])
# plt.plot(x, y, label = 'step function', linestyle = '--')
# plt.plot(x, z, label = 'sigmoid function')
# plt.legend()
# plt.show()


def identity_function(x):
    return x

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4], [0.2,0.5], [0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2]) 

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    z = identity_function(a3)

    return z

network = init_network()
x = np.array([1.0, 0.5])
tmp = forward(network, x)
print(tmp)