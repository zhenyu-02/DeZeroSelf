if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import dezeroSelf
from dezeroSelf.datasets import Spiral
from dezeroSelf.dataloaders import DataLoader
from dezeroSelf.models import MLP
from dezeroSelf import optimizer
import dezeroSelf.functions as F
import matplotlib.pyplot as plt

batch_size = 10
max_epoch = 100
hidden_size = 10
lr = 1.0

train_set = Spiral(train=True)
test_set = Spiral(train=False)

train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 3))
optimizer = optimizer.SGD(lr).setup(model)

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

    if epoch % 10 == 0:
        print('epoch %d, loss %.4f, accuracy %.4f' % (epoch + 1, sum_loss / len(train_set), sum_acc / len(train_set)))

plt.plot(train_loss_list, label='train loss')
plt.plot(test_loss_list, label='test loss')
plt.legend()
plt.show()
plt.plot(train_acc_list, label='train acc')
plt.plot(test_acc_list, label='test acc')
plt.legend()
plt.show()