# Add import path for the dezero directory.
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
import dezeroSelf
from dezeroSelf import Variable
from dezeroSelf.functions import tanh
from dezeroSelf.utils import plot_dot_graph

x = Variable(np.array(1.0))
y = tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)

print(y.grad)

iters = 0

for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)
    print(i, x.grad)

gx = x.grad
gx.name = 'gx'+str(iters+1)
plot_dot_graph(gx, to_file='tanh.png')