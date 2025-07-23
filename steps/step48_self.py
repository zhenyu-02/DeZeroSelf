if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
import math
from dezeroSelf import optimizer
from dezeroSelf.core import as_variable
from dezeroSelf.models import MLP
from dezeroSelf.optimizer import MomentumSGD
import dezeroSelf.functions as F
import numpy as np

from dezeroSelf.datasets import get_spiral

x, t = get_spiral(train=True)
print(x.shape, t.shape)


# hyperparameters
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

# load dataset, split into train and test
x, t = get_spiral(train=True)
model = MLP([hidden_size, 3])
optimizer = optimizer.SGD(lr).setup(model)

data_size = len(x)
max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch):
    #rerandomize
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)
    
    avg_loss = sum_loss / data_size
    print('epoch %d, loss %.2f' % (epoch + 1, avg_loss))





        









