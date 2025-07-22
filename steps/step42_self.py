# Add import path for the dezero directory.
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
import numpy as np
import dezeroSelf.functions
from dezeroSelf import Variable


# toy dataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))

def predict(x):
    y = dezeroSelf.functions.matmul(x, W) + b
    return y

def mean_squared_error_simple(x0, x1):
    diff = x0 - x1
    return dezeroSelf.functions.sum(diff ** 2) / len(diff)

lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    # will not store inner computation graph using Function class
    loss = dezeroSelf.functions.mean_squared_error(y, y_pred)
    W.cleargrad()
    b.cleargrad()
    loss.backward()
    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(W, b, loss)