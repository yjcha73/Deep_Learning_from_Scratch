import numpy as np
import matplotlib.pyplot as plt

x = np.array([2,3,4,5,6,7,8,9,10,11])
y = np.array([8,14,33,47,80,97,118,123,139,153])

m, b = np.polyfit(x, y, 1)
x1 = np.arange(2,11,0.1)
y1 = m*x1 + b

lbl = 'y='+str(m)+'x+'+str(b)
print(lbl)

plt.scatter(x,y,label='data')
plt.plot(x1, y1, label = lbl)

plt.show()