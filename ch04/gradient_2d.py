import numpy as np
import matplotlib.pyplot as plt

def _numerical_gradient_no_batch(f,x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad

def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)
        
def gradient_descent(f, init_x, lr=0.1, step_num = 100):
    x = init_x
    plt.scatter(x[0], x[1], s=10)

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        plt.scatter(x[0], x[1], s=10)

 
    return x

x = np.array([-2.0, 1.0])
_min = gradient_descent(function_2, x)
print(_min)
print("minimum value: " + str(function_2(_min)) )


x = np.arange(-2.0, 2.1, 0.25)
y = np.arange(-2.0, 2.1, 0.25)

# # from github source code
# X, Y = np.meshgrid(x, y)

# X = X.flatten()
# Y = Y.flatten()

# grad = numerical_gradient(function_2, np.array([X, Y]))

# plt.figure()
# plt.quiver(X, Y, -grad[0], -grad[1], angles = 'xy')
# plt.xlim([-2,2])
# plt.ylim([-2,2])
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid()
# plt.legend()
# plt.draw()
# plt.show()

# by Youngjun --------------------------------------------------
for i in range(len(x)):
    for j in range(len(y)):
        origin = np.array([x[i],y[j]])
        grad = -numerical_gradient(function_2, origin)
        plt.quiver(*origin, grad[0], grad[1], width = 0.003, scale = 100)

plt.grid(b=True, linestyle="--")        
plt.show()
#----------------------------------------------------------------