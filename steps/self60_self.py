if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from dezeroSelf import optimizer
import dezeroSelf
from dezeroSelf.layers import LSTM, Linear
from dezeroSelf.models import Model
from dezeroSelf.dataloaders import DataLoader
from dezero.datasets import SinCurve
import dezeroSelf.functions as F
import matplotlib.pyplot as plt
import numpy as np

train_set = SinCurve(train=True)

max_epoch = 100
batch_size = 10
hidden_size = 50
bptt_length = 50

dataloader = DataLoader(train_set, batch_size)
seqlen = len(train_set)

class BetterRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = LSTM(hidden_size)
        self.fc = Linear(out_size)
    
    def reset_state(self):
        self.rnn.reset_state()
    
    def forward(self, x):
        h = self.rnn(x)
        y = self.fc(h)
        return y
    


model = BetterRNN(hidden_size, 1)
optimizer = optimizer.Adam().setup(model)

for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0, 0
    
    for x, t in dataloader:
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


