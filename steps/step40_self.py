# Add import path for the dezero directory.
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezeroSelf
from dezeroSelf import Variable

x = Variable(np.array([1, 2, 3, 4, 5, 6]))
x1 = Variable(np.array([1]))
y = x + x1
print(y)

y.backward()
print(x.grad)