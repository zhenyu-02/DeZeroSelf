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
    

class Transformer(Model):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len=512, dropout_rate=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Word embedding layer
        from dezeroSelf.layers import Linear, PositionalEncoding, TransformerBlock
        self.embedding = Linear(d_model, nobias=True)  # Simplified embedding layer
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = []
        for i in range(num_layers):
            block = TransformerBlock(d_model, num_heads, d_ff, dropout_rate)
            setattr(self, f'transformer_{i}', block)
            self.transformer_blocks.append(block)
        
        # Output layer
        self.output_linear = Linear(vocab_size)
        self.dropout_rate = dropout_rate
    
    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, input_dim) or already embedded vectors
        
        # If input is word IDs, convert to embedding first (simplified here)
        if x.shape[-1] != self.d_model:
            x = self.embedding(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply dropout
        if self.dropout_rate > 0:
            x = F.dropout(x, self.dropout_rate)
        
        # Pass through all Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Output projection
        output = self.output_linear(x)
        return output


class SimpleAttentionClassifier(Model):
    """Simple classifier based on attention mechanism"""
    def __init__(self, input_dim, d_model, num_heads, num_classes, dropout_rate=0.1):
        super().__init__()
        
        from dezeroSelf.layers import Linear, MultiHeadAttention, LayerNorm
        
        self.input_projection = Linear(d_model)  # Project input to d_model dimension
        self.attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.norm = LayerNorm(d_model)
        self.classifier = Linear(num_classes)
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        
        # Project to model dimension
        x = self.input_projection(x)
        
        # Self-attention
        attn_output = self.attention(x, x, x)
        
        # Residual connection and layer normalization
        x = self.norm(x + attn_output)
        
        # Global average pooling (simplified sequence aggregation method)
        x = F.sum(x, axis=1) / x.shape[1]  # (batch_size, d_model)
        
        # Classification
        if self.dropout_rate > 0:
            x = F.dropout(x, self.dropout_rate)
        output = self.classifier(x)
        
        return output 