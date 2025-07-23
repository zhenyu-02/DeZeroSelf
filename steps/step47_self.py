# Add import path for the dezero directory.
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
from dezeroSelf.core import as_variable
from dezeroSelf.models import MLP
from dezeroSelf.optimizer import MomentumSGD
import dezeroSelf.functions as F
import numpy as np



def softmax_simple(x, axis=1):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y, axis=axis, keepdims=True)
    return y / sum_y


# softmax cross entropy
model = MLP([100, 100, 100, 100, 3])
optimizer = MomentumSGD(lr=0.01)

x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])

y = model(x)
loss = F.softmax_cross_entropy(y, t)
print(loss)



