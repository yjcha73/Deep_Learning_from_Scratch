import sys, os
import numpy as np
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

# Hyper parameters
iters_num = 1200       # number of interation
train_size = x_train.shape[0]
batch_size = 100        # mini batch size
learning_rate = 0.1

network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    start_time = datetime.now()
    # get mini batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # calculate gradient
    grad = network.numerical_gradient(x_batch, t_batch)

    # update params
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key] 

    # record learing record
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    end_time = datetime.now()
    interval = end_time - start_time
    print(str(i) + ": loss = " + str(loss) + ", time: " + str(interval))

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc: " + str(train_acc) + ", " + str(test_acc))

plt.plot(train_loss_list)
plt.show()