from email.charset import QP
from turtle import forward
from dezeroSelf.layers import Layer
from dezeroSelf import utils
import dezeroSelf.functions as F
from dezeroSelf.layers import Linear

class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)

class MLP(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid, use_dropout=False):
        super().__init__()
        self.activation = activation
        self.use_dropout = use_dropout
        self.layers = []
        for i, out_size in enumerate(fc_output_sizes):
            layer = Linear(out_size)
            setattr(self, 'l'+str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            if self.use_dropout:
                x = F.dropout(x)
            x = self.activation(l(x))
        if self.use_dropout:
            x = F.dropout(x)
        x = self.layers[-1](x)
        return x
    