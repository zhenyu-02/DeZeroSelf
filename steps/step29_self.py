# Add import path for the dezero directory.
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))



import numpy as np
from dezeroSelf import Function, Variable
from dezeroSelf.utils import plot_dot_graph

def f(x):
    y = x**4 - 2*x**2 
    return y

def gx2(x):
    y = 12*x**2 - 4
    return y

x = Variable(np.array(2.0))
iters = 10
lr = 0.1

for i in range(iters):
    print(i, x)
    y = f(x)
    x.cleargrad()
    y.backward()
    x.data -= lr * x.grad

print(x)    
x = Variable(np.array(2.0))

for i in range(iters):
    print(i, x)
    y = f(x)
    x.cleargrad()
    y.backward()
    x.data -= x.grad / gx2(x).data

print(x)