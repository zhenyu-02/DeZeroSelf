from dezeroSelf.core import Parameter
import numpy as np
from dezeroSelf.functions import linear
import dezeroSelf.functions as F

class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj
            # yield self.__dict__[name]

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_size is not None:
            self.W.data = np.random.randn(self.in_size, self.out_size).astype(self.dtype)
        
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(self.out_size, dtype=self.dtype), name='b')
    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1/I)
        return W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            self.W.data = self._init_W()
        y = linear(x, self.W, self.b)
        return y
    

class RNN(Layer):
    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        self.x2h = Linear(hidden_size, in_size)
        self.h2h = Linear(out_size=hidden_size, in_size=hidden_size, nobias=True)
        self.h = None

    def reset_state(self):
        self.h = None

    def forward(self, x):
        if self.h is None:
            h_new = F.tanh(self.x2h(x))
        else:
            h_new = F.tanh(self.x2h(x) + self.h2h(self.h))
        self.h = h_new
        return h_new


class LSTM(Layer):
    def __init__(self, hidden_size, in_size=None):
        super().__init__()

        H, I = hidden_size, in_size
        self.x2f = Linear(in_size=I, out_size=H)
        self.x2i = Linear(in_size=I, out_size=H)
        self.x2o = Linear(in_size=I, out_size=H)
        self.x2g = Linear(in_size=I, out_size=H)

        self.h2f = Linear(in_size=H, out_size=H, nobias=True)
        self.h2i = Linear(in_size=H, out_size=H, nobias=True)
        self.h2o = Linear(in_size=H, out_size=H, nobias=True)
        self.h2g = Linear(in_size=H, out_size=H, nobias=True)

        self.reset_state()

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self, x):
        h, c = self.h, self.c

        if h is not None:
            if h.shape[0] != x.shape[0]:
                h = h[:x.shape[0]]
                c = c[:x.shape[0]]

        if h is None:
            f = F.sigmoid(self.x2f(x))
            i = F.sigmoid(self.x2i(x))
            o = F.sigmoid(self.x2o(x))
            g = F.tanh(self.x2g(x))
        else:
            f = F.sigmoid(self.x2f(x) + self.h2f(h))
            i = F.sigmoid(self.x2i(x) + self.h2i(h))
            o = F.sigmoid(self.x2o(x) + self.h2o(h))
            g = F.tanh(self.x2g(x) + self.h2g(h))

        if c is None:
            c_new = i * g
        else:
            c_new = f * c + i * g

        h_new = o * F.tanh(c_new)
        self.c = c_new
        self.h = h_new
        return h_new


class LayerNorm(Layer):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        
        # normalized_shape can be an integer or tuple
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Initialize learnable parameters
        self.gamma = Parameter(np.ones(normalized_shape, dtype=np.float32), name='gamma')
        self.beta = Parameter(np.zeros(normalized_shape, dtype=np.float32), name='beta')
    
    def forward(self, x):
        return F.layer_norm(x, self.gamma, self.beta, self.eps)


class BatchNorm(Layer):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = Parameter(np.ones(num_features, dtype=np.float32), name='gamma')
        self.beta = Parameter(np.zeros(num_features, dtype=np.float32), name='beta')
        
        # Running average parameters (no gradients needed)
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)
    
    def forward(self, x):
        from dezeroSelf.core import Config, Variable
        
        if Config.train:
            # Training mode: compute current batch statistics
            batch_mean = np.mean(x.data, axis=0, keepdims=False)
            batch_var = np.var(x.data, axis=0, keepdims=False)
            
            # Update running averages
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            # Use current batch statistics
            mean = Variable(batch_mean)
            var = Variable(batch_var)
        else:
            # Test mode: use running averages
            mean = Variable(self.running_mean)
            var = Variable(self.running_var)
            
        return F.batch_norm(x, self.gamma, self.beta, mean, var, self.eps)


