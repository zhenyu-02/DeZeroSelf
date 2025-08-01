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
            self.W.data = self._init_W()
        
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(self.out_size, dtype=self.dtype), name='b')
    def _init_W(self):
        I, O = self.in_size, self.out_size
        # More conservative initialization: reduce variance
        if hasattr(self, '_is_attention_layer') and self._is_attention_layer:
            # Attention layers use smaller initialization
            scale = np.sqrt(1.0 / (I + O))
        else:
            # Standard layers use Xavier initialization but with smaller scaling
            scale = np.sqrt(1.0 / (I + O)) * 0.5
        
        W_data = np.random.randn(I, O).astype(self.dtype) * scale
        return W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[-1]  # Use the last dimension as the number of input features
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


class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout_rate = dropout_rate
        
        # Linear transformation layers
        self.w_q = Linear(d_model, nobias=True)  # Query weights
        self.w_k = Linear(d_model, nobias=True)  # Key weights
        self.w_v = Linear(d_model, nobias=True)  # Value weights
        self.w_o = Linear(d_model, nobias=True)  # Output weights
        
        # Mark as attention layer for more conservative initialization
        self.w_q._is_attention_layer = True
        self.w_k._is_attention_layer = True
        self.w_v._is_attention_layer = True
        self.w_o._is_attention_layer = True
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        
        # Linear transformations to get Q, K, V
        Q = self.w_q(query)  # (batch_size, seq_len, d_model)
        K = self.w_k(key)    # (batch_size, seq_len, d_model)
        V = self.w_v(value)  # (batch_size, seq_len, d_model)
        
        # Reshape for multi-head and transpose: (batch_size, seq_len, num_heads, d_k) -> (batch_size, num_heads, seq_len, d_k)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Transpose to correct dimension order
        Q = F.transpose(Q, (0, 2, 1, 3))  # (batch_size, num_heads, seq_len, d_k)
        K = F.transpose(K, (0, 2, 1, 3))  # (batch_size, num_heads, seq_len, d_k)
        V = F.transpose(V, (0, 2, 1, 3))  # (batch_size, num_heads, seq_len, d_k)
        
        # For simplified implementation, process each head separately
        attention_outputs = []
        for i in range(self.num_heads):
            # Extract Q, K, V for the i-th head
            q_i = Q[:, i, :, :]  # (batch_size, seq_len, d_k)
            k_i = K[:, i, :, :]  # (batch_size, seq_len, d_k)
            v_i = V[:, i, :, :]  # (batch_size, seq_len, d_k)
            
            # Compute attention
            output_i, _ = F.scaled_dot_product_attention(q_i, k_i, v_i, mask)
            attention_outputs.append(output_i)
        
        # Concatenate outputs of all heads
        concat_output = F.concatenate(attention_outputs, axis=-1)
        
        # Apply dropout (if in training mode)
        if self.dropout_rate > 0:
            concat_output = F.dropout(concat_output, self.dropout_rate)
        
        # Final linear transformation
        output = self.w_o(concat_output)
        
        return output


class TransformerBlock(Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # Feed Forward Network
        self.ff1 = Linear(d_ff)
        self.ff2 = Linear(d_model)
        self.dropout_rate = dropout_rate
    
    def forward(self, x, mask=None):
        # Multi-head self-attention + residual connection + layer normalization
        attn_output = self.attention(x, x, x, mask)
        if self.dropout_rate > 0:
            attn_output = F.dropout(attn_output, self.dropout_rate)
        x = self.norm1(x + attn_output)
        
        # Feed Forward + residual connection + layer normalization
        ff_output = self.ff2(F.relu(self.ff1(x)))
        if self.dropout_rate > 0:
            ff_output = F.dropout(ff_output, self.dropout_rate)
        x = self.norm2(x + ff_output)
        
        return x


class PositionalEncoding(Layer):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Generate positional encoding
        pe = F.positional_encoding(seq_len, d_model, self.max_len)
        
        # Add positional encoding to input
        return x + pe


