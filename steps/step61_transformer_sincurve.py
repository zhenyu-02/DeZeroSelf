if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezeroSelf import optimizer
import dezeroSelf
from dezeroSelf.layers import Linear, LayerNorm, TransformerBlock
from dezeroSelf.models import Model, Transformer, SimpleAttentionClassifier
from dezeroSelf.dataloaders import DataLoader
from dezero.datasets import SinCurve
import dezeroSelf.functions as F
import matplotlib.pyplot as plt
import numpy as np

# Dataset configuration
train_set = SinCurve(train=True)
max_epoch = 200
batch_size = 8
hidden_size = 64
bptt_length = 30

dataloader = DataLoader(train_set, batch_size)
seqlen = len(train_set)

# Transformer model for sequence modeling
class TransformerSeqModel(Model):
    def __init__(self, input_dim, d_model, num_heads, num_layers, d_ff, output_dim, dropout_rate=0.1):
        super().__init__()
        
        self.input_projection = Linear(d_model, nobias=True)  # Project input to d_model dimension
        
        # Transformer blocks
        self.transformer_blocks = []
        for i in range(num_layers):
            block = TransformerBlock(d_model, num_heads, d_ff, dropout_rate)
            setattr(self, f'transformer_{i}', block)
            self.transformer_blocks.append(block)
        
        self.norm = LayerNorm(d_model)
        self.output_projection = Linear(output_dim)
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim) for SinCurve input_dim=1
        
        # Project to model dimension
        x = self.input_projection(x)
        
        # Pass through all Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)  # Self-attention
        
        # Layer normalization
        x = self.norm(x)
        
        # Output projection
        output = self.output_projection(x)
        
        return output

# Simplified attention classifier (for sequence-to-single-value prediction)
class SimpleSeqToSeq(Model):
    def __init__(self, input_dim, d_model, num_heads, output_dim, dropout_rate=0.1):
        super().__init__()
        
        self.input_projection = Linear(d_model)
        self.attention = F.MultiHeadAttentionFunction(num_heads)  # This may need adjustment
        # Use MultiHeadAttention from layers instead
        from dezeroSelf.layers import MultiHeadAttention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.norm = LayerNorm(d_model)
        self.output_projection = Linear(output_dim)
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        # Project to model dimension
        x = self.input_projection(x)
        
        # Self-attention
        attn_output = self.attention(x, x, x)
        
        # Residual connection and layer normalization
        x = self.norm(x + attn_output)
        
        # Output projection
        output = self.output_projection(x)
        
        return output

# Baseline LSTM model (for comparison)
class SimpleLSTM(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        from dezeroSelf.layers import LSTM
        self.rnn = LSTM(hidden_size)
        self.fc = Linear(out_size)
    
    def reset_state(self):
        self.rnn.reset_state()
    
    def forward(self, x):
        h = self.rnn(x)
        y = self.fc(h)
        return y

# Training function
def train_model(model, model_name, max_epoch=max_epoch, use_lstm=False):
    print(f"\n=== Training {model_name} ===")
    # Use smaller learning rate to ensure training stability
    optimizer_instance = optimizer.Adam(lr=0.0005).setup(model)
    loss_history = []
    
    for epoch in range(max_epoch):
        if use_lstm:
            model.reset_state()
        loss, count = 0, 0
        
        for x, t in dataloader:
            # Ensure correct input shape: (batch_size, 1, 1) -> (batch_size, 1, 1)
            if len(x.shape) == 2:
                x = x.reshape(x.shape[0], 1, 1)  # (batch_size, seq_len=1, feature=1)
            if len(t.shape) == 2:
                t = t.reshape(t.shape[0], 1, 1)
            
            y = model(x)
            loss += F.mean_squared_error(y, t)
            count += 1
            
            if count % bptt_length == 0 or count == seqlen:
                model.cleargrads()
                loss.backward()
                loss.unchain_backward()
                optimizer_instance.update()

        avg_loss = float(loss.data) / count
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:  # Print every 10 epochs
            print('| epoch %d | loss %.6f' % (epoch + 1, avg_loss))
        loss, count = 0, 0
    
    return loss_history

# Prediction function
def predict_and_plot(model, model_name, color, use_lstm=False):
    xs = np.cos(np.linspace(0, 4*np.pi, 1000))
    if use_lstm:
        model.reset_state()
    pred_list = []

    with dezeroSelf.no_grad():
        for x in xs:
            x_input = np.array(x).reshape(1, 1, 1)  # (batch_size=1, seq_len=1, feature=1)
            y = model(x_input)
            pred_list.append(float(y.data))
    
    return xs, pred_list

# Run Transformer comparison experiment
def run_transformer_experiment():
    print("Starting Transformer vs LSTM comparison experiment...")
    
    # Model parameters
    input_dim = 10
    d_model = 64  # Smaller model to adapt to simple task
    num_heads = 8
    num_layers = 2
    d_ff = 128
    output_dim = 1
    dropout_rate = 0.1
    
    # Create different models
    models = {
        'LSTM': SimpleLSTM(hidden_size, 1),
        'SimpleAttention': SimpleSeqToSeq(input_dim, d_model, num_heads, output_dim, dropout_rate),
        'Transformer': TransformerSeqModel(input_dim, d_model, num_heads, num_layers, d_ff, output_dim, dropout_rate)
    }
    
    # Train all models and record losses
    loss_histories = {}
    trained_models = {}
    
    for name, model in models.items():
        use_lstm = (name == 'LSTM')
        loss_history = train_model(model, name, use_lstm=use_lstm)
        loss_histories[name] = loss_history
        trained_models[name] = model
        print(f"{name} Final Loss: {loss_history[-1]:.6f}")
    
    # Plot comparison results
    plt.figure(figsize=(15, 5))
    
    # Training loss comparison
    plt.subplot(1, 3, 1)
    colors = ['blue', 'red', 'green']
    for i, (name, history) in enumerate(loss_histories.items()):
        plt.plot(history, label=name, color=colors[i])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.yscale('log')
    
    # Prediction results comparison
    plt.subplot(1, 3, 2)
    true_xs = np.cos(np.linspace(0, 4*np.pi, 1000))
    plt.plot(np.arange(len(true_xs)), true_xs, label='True', color='black', linewidth=2)
    
    for i, (name, model) in enumerate(trained_models.items()):
        use_lstm = (name == 'LSTM')
        xs, pred_list = predict_and_plot(model, name, colors[i], use_lstm=use_lstm)
        plt.plot(np.arange(len(pred_list)), pred_list, label=f'Pred {name}', 
                color=colors[i], alpha=0.7)
    
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Prediction Comparison')
    plt.legend()
    
    # Local prediction comparison (first 200 points)
    plt.subplot(1, 3, 3)
    plt.plot(np.arange(200), true_xs[:200], label='True', color='black', linewidth=2)
    
    for i, (name, model) in enumerate(trained_models.items()):
        use_lstm = (name == 'LSTM')
        xs, pred_list = predict_and_plot(model, name, colors[i], use_lstm=use_lstm)
        plt.plot(np.arange(200), pred_list[:200], label=f'Pred {name}', 
                color=colors[i], alpha=0.7)
    
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Local Prediction (First 200 steps)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('transformer_sincurve_comparison.png', dpi=150, bbox_inches='tight')
    print("Chart saved to transformer_sincurve_comparison.png")
    
    # Print final results summary
    print("\n=== Experiment Results Summary ===")
    final_losses = {name: history[-1] for name, history in loss_histories.items()}
    sorted_results = sorted(final_losses.items(), key=lambda x: x[1])
    
    print("Ranked by final loss (smallest to largest):")
    for i, (name, loss) in enumerate(sorted_results):
        print(f"{i+1}. {name}: {loss:.6f}")
    
    # Calculate relative improvement
    lstm_loss = final_losses['LSTM']
    print(f"\nImprovement relative to LSTM model:")
    for name, loss in final_losses.items():
        if name != 'LSTM':
            improvement = (lstm_loss - loss) / lstm_loss * 100
            print(f"{name}: {improvement:+.2f}% {'✓' if improvement > 0 else '✗'}")
    
    return loss_histories, trained_models

if __name__ == "__main__":
    # Run Transformer comparison experiment
    print("Starting Transformer training experiment...")
    try:
        loss_histories, trained_models = run_transformer_experiment()
        print("Experiment completed!")
    except Exception as e:
        print(f"Error occurred during experiment: {e}")
        import traceback
        traceback.print_exc() 