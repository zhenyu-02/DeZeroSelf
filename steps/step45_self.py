# Add import path for the dezero directory.
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
import numpy as np
from dezeroSelf import Variable, Model
import dezeroSelf.layers as L
import dezeroSelf.functions as F
from dezeroSelf.models import MLP

# toy dataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2.0 * np.pi * x) + np.random.rand(100, 1)


lr = 0.2
max_iter = 10000
hidden_size = 10

# class TwoLayerNet(Model):
#     def __init__(self, hidden_size, out_size):
#         super().__init__()
#         self.l1 = L.Linear(hidden_size)
#         self.l2 = L.Linear(out_size)

#     def forward(self, x):
#         y = F.sigmoid(self.l1(x))
#         y = self.l2(y)
#         return y

# model = TwoLayerNet(hidden_size, 1)

model = MLP([hidden_size, 300, 20, 1])



for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print(loss)

model.plot(x, to_file='step45.png')