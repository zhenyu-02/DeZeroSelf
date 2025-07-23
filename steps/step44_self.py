# Add import path for the dezero directory.
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
import numpy as np
import dezeroSelf
from dezeroSelf import Variable
from dezeroSelf.layers import Linear
from dezeroSelf.functions import sigmoid

# toy dataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2.0 * np.pi * x) + np.random.rand(100, 1)

l1 = Linear(10)
l2 = Linear(3)

def predict(x):
    y = l1(x)
    y = sigmoid(y)
    y = l2(y)
    return y

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = dezeroSelf.functions.mean_squared_error(y, y_pred)
    l1.cleargrads()
    l2.cleargrads()
    loss.backward()
    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss)