from turtle import forward
import numpy as np
from dezeroSelf.core import Function, as_variable
from dezeroSelf import utils


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx

def sin(x):
    return Sin()(x)

class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

def cos(x):
    return Cos()(x)

class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y**2)
        return gx
    
def tanh(x):
    return Tanh()(x)

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y
        return gx

def exp(x):
    return Exp()(x)


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape #target shape

    def forward(self, x):
        self.x_shape = x.shape # source shape
        y = x.reshape(self.shape)
        return y
    
    def backward(self, gy):
        return reshape(gy, self.x_shape)
    
def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)
    
class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes
        
    def forward(self, x):
        if self.axes is None:
            y = np.transpose(x)
        else:
            y = np.transpose(x, self.axes)
        return y
    
    def backward(self, gy):
        if self.axes is None:
            gx = transpose(gy)
        else:
            # 反向axes
            inv_axes = np.argsort(self.axes)
            gx = transpose(gy, inv_axes)
        return gx

def transpose(x, axes=None):
    return Transpose(axes)(x)

class BatchMatMul(Function):
    def forward(self, x, y):
        # 批量矩阵乘法: (batch_size, m, k) @ (batch_size, k, n) -> (batch_size, m, n)
        z = np.matmul(x, y)
        return z
    
    def backward(self, gz):
        x, y = self.inputs
        gx = batch_matmul(gz, transpose(y, (0, 2, 1)))
        gy = batch_matmul(transpose(x, (0, 2, 1)), gz)
        return gx, gy

def batch_matmul(x, y):
    return BatchMatMul()(x, y)

class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx
def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.x_shape)
        return y
    
    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx
def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx
def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y
    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW
def matmul(x, W):
    return MatMul()(x, W)

class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y
    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2.0 / len(diff))
        gx1 = -gx0
        return gx0, gx1
def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


class Linear(Function):
    def forward(self, x, W, b):
        # Handle batch input: if x is 3D tensor (batch_size, seq_len, input_dim)
        if len(x.shape) == 3:
            batch_size, seq_len, input_dim = x.shape
            # Reshape to 2D for matrix multiplication
            x_reshaped = x.reshape(-1, input_dim)  # (batch_size * seq_len, input_dim)
            y = x_reshaped.dot(W)  # (batch_size * seq_len, output_dim)
            # Reshape back to 3D
            y = y.reshape(batch_size, seq_len, -1)  # (batch_size, seq_len, output_dim)
        else:
            y = x.dot(W)
        
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        
        # Handle gradient computation for 3D tensors
        if len(x.shape) == 3:
            batch_size, seq_len, input_dim = x.shape
            output_dim = W.shape[1]
            
            # Reshape gradients
            gy_reshaped = gy.reshape(-1, output_dim)  # (batch_size * seq_len, output_dim)
            x_reshaped = x.reshape(-1, input_dim)     # (batch_size * seq_len, input_dim)
            
            # Compute gradients
            gx_reshaped = matmul(gy_reshaped, transpose(W))  # (batch_size * seq_len, input_dim)
            gx = gx_reshaped.reshape(batch_size, seq_len, input_dim)  # Restore 3D shape
            
            gW = matmul(transpose(x_reshaped), gy_reshaped)  # (input_dim, output_dim)
        else:
            gx = matmul(gy, transpose(W))
            gW = matmul(transpose(x), gy)
        
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


class Sigmoid(Function):
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y) * y
        return gx
def sigmoid(x):
    return Sigmoid()(x)

class ReLU(Function):
    def forward(self, x):
        y = np.maximum(x, 0)
        return y
    def backward(self, gy):
        x, = self.inputs
        gx = (x.data > 0) * gy
        return gx
def relu(x):
    return ReLU()(x)


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)
class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        gx = np.zeros(self.in_shape, dtype=gy.dtype)
        np.add.at(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


def get_item(x, slices):
    f = GetItem(slices)
    return f(x)


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        # Numerically stable softmax: subtract max value to prevent overflow
        x_max = np.max(x, axis=self.axis, keepdims=True)
        x_shifted = x - x_max
        # Clip extreme values to prevent overflow
        x_shifted = np.clip(x_shifted, -50, 50)
        y = np.exp(x_shifted)
        sum_y = np.sum(y, axis=self.axis, keepdims=True)
        return y / (sum_y + 1e-8)  # Add small value to prevent division by zero
    
    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sum_gx = sum(gx, axis=self.axis, keepdims=True)
        gx -= y * sum_gx
        return gx
def softmax(x, axis=1):
    return Softmax(axis)(x)

def softmax_cross_entropy(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]

    p = softmax(x)
    p = clip(p, 1e-15, 1.0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * sum(tlog_p) / N
    return y

class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        y = np.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)
class Log(Function):
    def forward(self, x):
        y = np.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx
def log(x):
    return Log()(x)


def dropout(x, dropout_ratio=0.5):
    # Inverted dropout
    x = as_variable(x)
    from dezeroSelf.core import Config
    if Config.train:
        mask = np.random.rand(*x.shape) > dropout_ratio
        scale = np.array(mask).size / np.array(mask).sum()
        return x * mask * scale
    else:
        return x

def accuracy(y, t):
    y, t = as_variable(y), as_variable(t)
    pred = y.data.argmax(axis=1).reshape(t.shape)

    result = (pred == t.data)
    acc = result.mean()
    from dezeroSelf.core import Variable, as_array
    return Variable(as_array(acc))

class LayerNorm(Function):
    def __init__(self, eps=1e-5):
        self.eps = eps
    
    def forward(self, x, gamma, beta):
        # Support 2D and 3D inputs
        if len(x.shape) == 2:
            N, D = x.shape
            # Calculate mean and variance
            self.mean = x.mean(axis=1, keepdims=True)
            self.var = x.var(axis=1, keepdims=True)
            
            # Normalize
            self.x_normalized = (x - self.mean) / np.sqrt(self.var + self.eps)
            
            # Scale and shift
            y = gamma * self.x_normalized + beta
            
        elif len(x.shape) == 3:
            batch_size, seq_len, D = x.shape
            # Normalize over the last dimension
            self.mean = x.mean(axis=-1, keepdims=True)  # (batch_size, seq_len, 1)
            self.var = x.var(axis=-1, keepdims=True)    # (batch_size, seq_len, 1)
            
            # Normalize
            self.x_normalized = (x - self.mean) / np.sqrt(self.var + self.eps)
            
            # Scale and shift (gamma and beta should have shape (D,))
            y = gamma * self.x_normalized + beta
        else:
            raise ValueError(f"LayerNorm does not support {len(x.shape)}D input")
            
        return y
    
    def backward(self, gy):
        x, gamma, beta = self.inputs
        
        if len(x.shape) == 2:
            N, D = x.shape
            
            # Gradients for gamma and beta (ensure shape matching)
            dgamma = sum(gy * self.x_normalized, axis=0, keepdims=False)
            dbeta = sum(gy, axis=0, keepdims=False)
            
            # Gradient for x
            dx_normalized = gy * gamma
            dvar = sum(dx_normalized * (x.data - self.mean) * (-0.5) * (self.var + self.eps) ** (-1.5), axis=1, keepdims=True)
            dmean = sum(dx_normalized * (-1.0 / np.sqrt(self.var + self.eps)), axis=1, keepdims=True) + \
                    dvar * sum(-2.0 * (x.data - self.mean), axis=1, keepdims=True) / D
            
            dx = dx_normalized / np.sqrt(self.var + self.eps) + \
                 dvar * 2.0 * (x.data - self.mean) / D + \
                 dmean / D
                 
        elif len(x.shape) == 3:
            batch_size, seq_len, D = x.shape
            
            # Gradients for gamma and beta (sum along batch and seq dimensions)
            dgamma = sum(sum(gy * self.x_normalized, axis=0), axis=0, keepdims=False)  # (D,)
            dbeta = sum(sum(gy, axis=0), axis=0, keepdims=False)  # (D,)
            
            # Gradient for x
            dx_normalized = gy * gamma
            dvar = sum(dx_normalized * (x.data - self.mean) * (-0.5) * (self.var + self.eps) ** (-1.5), axis=-1, keepdims=True)
            dmean = sum(dx_normalized * (-1.0 / np.sqrt(self.var + self.eps)), axis=-1, keepdims=True) + \
                    dvar * sum(-2.0 * (x.data - self.mean), axis=-1, keepdims=True) / D
            
            dx = dx_normalized / np.sqrt(self.var + self.eps) + \
                 dvar * 2.0 * (x.data - self.mean) / D + \
                 dmean / D
        else:
            raise ValueError(f"LayerNorm does not support {len(x.shape)}D input backward propagation")
        
        return as_variable(dx), dgamma, dbeta

def layer_norm(x, gamma, beta, eps=1e-5):
    return LayerNorm(eps)(x, gamma, beta)

class BatchNorm(Function):
    def __init__(self, eps=1e-5):
        self.eps = eps
    
    def forward(self, x, gamma, beta, mean, var):
        # Normalize using provided mean and variance
        self.x_normalized = (x - mean) / np.sqrt(var + self.eps)
        # Scale and shift
        y = gamma * self.x_normalized + beta
        return y
    
    def backward(self, gy):
        x, gamma, beta, mean, var = self.inputs
        N = x.shape[0]
        
        # Gradients for gamma and beta (ensure shape matching)
        dgamma = sum(gy * self.x_normalized, axis=0, keepdims=False)
        dbeta = sum(gy, axis=0, keepdims=False)
        
        # Gradient for x (simplified version, assuming mean and variance are constants)
        dx = gy * gamma / np.sqrt(var.data + self.eps)
        
        return as_variable(dx), dgamma, dbeta, None, None

def batch_norm(x, gamma, beta, mean, var, eps=1e-5):
    return BatchNorm(eps)(x, gamma, beta, mean, var)

# ===================== Attention Mechanism Related Functions =====================

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Compute scaled dot product attention
    Args:
        query: (batch_size, seq_len, d_k)
        key: (batch_size, seq_len, d_k) 
        value: (batch_size, seq_len, d_v)
        mask: Optional mask to prevent attention to padding positions
    Returns:
        output: (batch_size, seq_len, d_v)
        attention_weights: (batch_size, seq_len, seq_len)
    """
    query, key, value = as_variable(query), as_variable(key), as_variable(value)
    
    d_k = query.shape[-1]
    
    # Numerical stability improvement: use more conservative scaling factor
    scale_factor = np.sqrt(max(d_k, 1.0))
    
    # Compute attention scores: Q * K^T / sqrt(d_k)
    # Need to transpose last two dimensions of key: (batch_size, seq_len, d_k) -> (batch_size, d_k, seq_len)
    key_transposed = transpose(key, (0, 2, 1)) if len(key.shape) == 3 else transpose(key)
    scores = batch_matmul(query, key_transposed) / scale_factor
    
    # Clip scores to prevent extreme values
    scores_data = scores.data if hasattr(scores, 'data') else scores
    scores_data = np.clip(scores_data, -10, 10)
    scores = as_variable(scores_data)
    
    # Apply mask (if provided)
    if mask is not None:
        mask = as_variable(mask)
        # Set masked positions to smaller negative numbers (not too extreme)
        scores = scores + (mask * -1e4)
    
    # Apply softmax to get attention weights
    attention_weights = softmax(scores, axis=-1)
    
    # Compute weighted output
    output = batch_matmul(attention_weights, value)
    
    return output, attention_weights

def positional_encoding(seq_len, d_model, max_len=10000):
    """
    Generate positional encoding
    Args:
        seq_len: Sequence length
        d_model: Model dimension
        max_len: Maximum sequence length
    Returns:
        Positional encoding matrix (seq_len, d_model)
    """
    pos = np.arange(seq_len).reshape(-1, 1).astype(np.float32)
    
    # More stable div_term computation
    i = np.arange(0, d_model, 2).astype(np.float32)
    div_term = np.exp(-i * np.log(max_len) / d_model)
    
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    
    # Compute sin and cos, ensuring proper handling when d_model is even
    sin_indices = np.arange(0, d_model, 2)
    cos_indices = np.arange(1, d_model, 2)
    
    if len(sin_indices) > 0:
        pe[:, sin_indices] = np.sin(pos * div_term[:len(sin_indices)])
    if len(cos_indices) > 0:
        pe[:, cos_indices] = np.cos(pos * div_term[:len(cos_indices)])
    
    # Scale positional encoding to prevent it from being too large
    pe = pe * 0.1
    
    from dezeroSelf.core import Variable
    return Variable(pe)

class MultiHeadAttentionFunction(Function):
    """Core computation function for multi-head attention"""
    def __init__(self, num_heads):
        self.num_heads = num_heads
    
    def forward(self, query, key, value, w_q, w_k, w_v, w_o, mask=None):
        batch_size, seq_len, d_model = query.shape
        d_k = d_model // self.num_heads
        
        # Linear transformations to get Q, K, V
        Q = matmul(query, w_q)  # (batch_size, seq_len, d_model)
        K = matmul(key, w_k)    # (batch_size, seq_len, d_model)
        V = matmul(value, w_v)  # (batch_size, seq_len, d_model)
        
        # Reshape for multi-head form
        Q = Q.reshape(batch_size, seq_len, self.num_heads, d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, d_k).transpose(0, 2, 1, 3)
        
        # Apply scaled dot product attention to each head
        attention_output = []
        for i in range(self.num_heads):
            q_i = Q[:, i, :, :]  # (batch_size, seq_len, d_k)
            k_i = K[:, i, :, :]  # (batch_size, seq_len, d_k)
            v_i = V[:, i, :, :]  # (batch_size, seq_len, d_k)
            
            output_i, _ = scaled_dot_product_attention(q_i, k_i, v_i, mask)
            attention_output.append(output_i)
        
        # Concatenate outputs from all heads
        concat_output = concatenate(attention_output, axis=-1)
        
        # Final linear transformation
        final_output = matmul(concat_output, w_o)
        
        return final_output

class Concatenate(Function):
    def __init__(self, axis):
        self.axis = axis
    
    def forward(self, *inputs):
        self.input_shapes = [x.shape for x in inputs]
        y = np.concatenate(inputs, axis=self.axis)
        return y
    
    def backward(self, gy):
        # Split gradients according to original shapes
        split_indices = []
        cumsum = 0
        for shape in self.input_shapes[:-1]:
            cumsum += shape[self.axis]
            split_indices.append(cumsum)
        
        # Extract data from Variable, perform split, then wrap back to Variable
        gy_data = gy.data if hasattr(gy, 'data') else gy
        grad_arrays = np.split(gy_data, split_indices, axis=self.axis)
        grads = tuple(as_variable(grad) for grad in grad_arrays)
        return grads

def concatenate(inputs, axis):
    return Concatenate(axis)(*inputs)