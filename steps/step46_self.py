# Add import path for the dezero directory.
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
import numpy as np
from dezeroSelf import Variable, Model
import dezeroSelf.layers as L
import dezeroSelf.functions as F
from dezeroSelf.models import MLP
from dezeroSelf.optimizer import MomentumSGD

# toy dataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2.0 * np.pi * x) + np.random.rand(100, 1)


lr = 0.2
max_iter = 10000
hidden_size = 10

model = MLP([hidden_size, 300, 20, 1])

optimizer = MomentumSGD(lr=lr)
optimizer.setup(model)

for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()

    if i % 1000 == 0:
        print(loss)

model.plot(x, to_file='step46.png')