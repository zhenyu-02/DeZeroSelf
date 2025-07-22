if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))



import numpy as np
from dezeroSelf import Variable
import dezeroSelf.functions as F


x = Variable(np.array([[1,2,3], [4,5,6]]))


# y = x.reshape(2,3)
# y.backward(retain_grad=True)

y = F.transpose(x)
y = x.transpose()
y = x.T
y.backward()

print(x.grad)