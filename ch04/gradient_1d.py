import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pylab as plt

def function_1(x):
    return 0.01*x**2 + 0.1*x

def function_2(x):
    return np.sum(x**2)

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2 * h)

def function_tmp1(x0):
    return x0**2 + 4**2

def function_tmp2(x1):
    return 3**2 + x1**2

def test_function(x, y):
    return x**2 + y**2

def tangent_line(f,x):
    d = numerical_diff(f, x)
    y = f(x) - d*x  # y-intercept
    return lambda t: d*t + y

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel('x')
plt.ylabel('f(x)')

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x,y)
plt.plot(x,y2)
plt.show()



# # df/dx0 at x0=3, x1=4 where f(x0,x1) = x0^2 + x1^2
# df_dx0 = numerical_diff(function_tmp1, 3)
# print('df/dx0 = ' + str(df_dx0))

# # df/dx1 at x0=3, x1=4 where f(x0,x1) = x0^2 + x1^2
# df_dx1 = numerical_diff(function_tmp2, 4)
# print('df/dx1 = ' + str(df_dx1))

# x = np.arange(0.0, 20.0, 0.1)
# y = function_1(x)
# plot1 = plt.figure(1)
# plt.xlabel('x')
# plt.ylabel('f(x)')

# gradient_5 = numerical_diff(function_1, 5)
# y1 = gradient_5 * (x-5) + function_1(5)

# plt.plot(x,y)
# plt.plot(x,y1)

# ## graph of f(x, y) = x^2 + y^2
# x = np.linspace(-6, 6, 30)
# y = np.linspace(-6, 6, 30)

# X, Y = np.meshgrid(x, y)
# Z = test_function(X, Y)

# plot2 = plt.figure(2)
# ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# ax.set_title('surface')
# plt.show()
