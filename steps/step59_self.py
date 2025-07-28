# Add import path for the dezero directory.
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))



import dezeroSelf
from dezeroSelf.layers import RNN, Linear
from dezeroSelf.models import Model
import dezeroSelf.functions as F
import numpy as np
import matplotlib.pyplot as plt
from dezero.datasets import SinCurve
from dezeroSelf import optimizer


class SimpleRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = RNN(hidden_size)
        self.fc = Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, x):
        h = self.rnn(x)
        y = self.fc(h)
        return y
    



train_set = SinCurve(train=True)
print(train_set[0])
xs = [example[0] for example in train_set]
ts = [example[1] for example in train_set]
print(xs[0])
print(ts[0])
# plt.plot(np.arange(len(xs)), xs, label='xs')
# plt.plot(np.arange(len(ts)), ts, label='ts')
# plt.show()

max_epoch = 50
hidden_size = 100
bptt_length = 10

train_set = SinCurve(train=True)
seqlen = len(train_set)

model = SimpleRNN(hidden_size, 1)
optimizer = optimizer.Adam().setup(model)

for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0, 0

    for x, t in train_set:
        x = x.reshape(1, 1)
        y = model(x)
        loss += F.mean_squared_error(y, t)
        count += 1

        if count % bptt_length == 0 or count == seqlen:
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()

    avg_loss = float(loss.data) / count
    print('| epoch %d | loss %.4f' % (epoch + 1, avg_loss))
    loss, count = 0, 0

xs = np.cos(np.linspace(0, 4*np.pi, 1000))

model.reset_state()
pred_list = []

with dezeroSelf.no_grad():
    for x in xs:
        x = np.array(x).reshape(1, 1)
        y = model(x)
        pred_list.append(float(y.data))
        
plt.plot(np.arange(len(xs)), xs, label='xs')
plt.plot(np.arange(len(pred_list)), pred_list, label='preds')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


