if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pandas.io.formats.style import plt
import dezeroSelf
from dezeroSelf.datasets import MNIST
from dezeroSelf.dataloaders import DataLoader
from dezeroSelf.models import MLP
from dezeroSelf import optimizer
import dezeroSelf.functions as F
import numpy as np

def f(x):
    x = x.flatten()
    x = x.astype(np.float32)
    x /= 255.0
    return x


train_set = MNIST(train=True, transform=f, target_transform=None)
test_set = MNIST(train=False, transform=f, target_transform=None)


max_epoch = 20
batch_size = 10
hidden_size = 1000
lr = 0.01


train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 10), activation=F.relu)
optimizer = optimizer.Adam(lr).setup(model)

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0
    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)
    train_loss_list.append(sum_loss / len(train_set))
    train_acc_list.append(sum_acc / len(train_set))
    print('epoch %d, loss %.4f, accuracy %.4f' % (epoch + 1, sum_loss / len(train_set), sum_acc / len(train_set)))
    
    sum_loss, sum_acc = 0, 0
    with dezeroSelf.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
    test_loss_list.append(sum_loss / len(test_set))
    test_acc_list.append(sum_acc / len(test_set))
    print('test loss %.4f, accuracy %.4f' % (sum_loss / len(test_set), sum_acc / len(test_set)))

plt.plot(train_loss_list, label='train loss')
plt.plot(test_loss_list, label='test loss')
plt.legend()
plt.show()

plt.plot(train_acc_list, label='train acc')
plt.plot(test_acc_list, label='test acc')
plt.legend()
plt.show()


