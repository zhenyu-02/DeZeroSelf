if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from dezeroSelf import optimizer
import dezeroSelf
from dezeroSelf.layers import LSTM, Linear, LayerNorm, BatchNorm
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

# 原始模型（无归一化）
class BaseRNN(Model):
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

# 添加LayerNorm的模型
class LayerNormRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = LSTM(hidden_size)
        self.layer_norm = LayerNorm(normalized_shape=hidden_size)
        self.fc = Linear(out_size)
    
    def reset_state(self):
        self.rnn.reset_state()
    
    def forward(self, x):
        h = self.rnn(x)
        h = self.layer_norm(h)  # LSTM输出后添加LayerNorm
        y = self.fc(h)
        return y

# 添加BatchNorm的模型
class BatchNormRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = LSTM(hidden_size)
        self.batch_norm = BatchNorm(num_features=hidden_size)
        self.fc = Linear(out_size)
    
    def reset_state(self):
        self.rnn.reset_state()
    
    def forward(self, x):
        h = self.rnn(x)
        h = self.batch_norm(h)  # LSTM输出后添加BatchNorm
        y = self.fc(h)
        return y

# 同时添加LayerNorm和BatchNorm的模型
class DualNormRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = LSTM(hidden_size)
        self.batch_norm = BatchNorm(num_features=hidden_size)
        self.layer_norm = LayerNorm(normalized_shape=hidden_size)
        self.fc = Linear(out_size)
    
    def reset_state(self):
        self.rnn.reset_state()
    
    def forward(self, x):
        h = self.rnn(x)
        h = self.batch_norm(h)   # 先BatchNorm
        h = self.layer_norm(h)   # 再LayerNorm
        y = self.fc(h)
        return y

# 训练函数
def train_model(model, model_name, max_epoch=max_epoch):
    print(f"\n=== 训练 {model_name} ===")
    optimizer_instance = optimizer.Adam().setup(model)
    loss_history = []
    
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
                optimizer_instance.update()

        avg_loss = float(loss.data) / count
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:  # 每20个epoch打印一次
            print('| epoch %d | loss %.6f' % (epoch + 1, avg_loss))
        loss, count = 0, 0
    
    return loss_history

# 预测函数
def predict_and_plot(model, model_name, color):
    xs = np.cos(np.linspace(0, 4*np.pi, 1000))
    model.reset_state()
    pred_list = []

    with dezeroSelf.no_grad():
        for x in xs:
            x = np.array(x).reshape(1, 1)
            y = model(x)
            pred_list.append(float(y.data))
    
    return xs, pred_list

# 运行对比实验
def run_comparison_experiment():
    print("开始LSTM归一化层对比实验...")
    
    # 创建不同的模型
    models = {
        'Base (无归一化)': BaseRNN(hidden_size, 1),
        'LayerNorm': LayerNormRNN(hidden_size, 1),
        'BatchNorm': BatchNormRNN(hidden_size, 1),
        'LayerNorm + BatchNorm': DualNormRNN(hidden_size, 1)
    }
    
    # 训练所有模型并记录损失
    loss_histories = {}
    trained_models = {}
    
    for name, model in models.items():
        loss_history = train_model(model, name)
        loss_histories[name] = loss_history
        trained_models[name] = model
        print(f"{name} 最终损失: {loss_history[-1]:.6f}")
    
    # 绘制训练损失对比
    plt.figure(figsize=(15, 5))
    
    # 损失曲线对比
    plt.subplot(1, 3, 1)
    colors = ['blue', 'red', 'green', 'orange']
    for i, (name, history) in enumerate(loss_histories.items()):
        plt.plot(history, label=name, color=colors[i])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失对比')
    plt.legend()
    plt.yscale('log')  # 使用对数尺度更好地显示差异
    
    # 预测结果对比
    plt.subplot(1, 3, 2)
    true_xs = np.cos(np.linspace(0, 4*np.pi, 1000))
    plt.plot(np.arange(len(true_xs)), true_xs, label='True', color='black', linewidth=2)
    
    for i, (name, model) in enumerate(trained_models.items()):
        xs, pred_list = predict_and_plot(model, name, colors[i])
        plt.plot(np.arange(len(pred_list)), pred_list, label=f'Pred {name}', 
                color=colors[i], alpha=0.7)
    
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('预测结果对比')
    plt.legend()
    
    # 最后50个epoch的损失对比（局部放大）
    plt.subplot(1, 3, 3)
    for i, (name, history) in enumerate(loss_histories.items()):
        plt.plot(range(50, 100), history[-50:], label=name, color=colors[i])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('最后50个Epoch损失对比')
    plt.legend()
    
    plt.tight_layout()
    # plt.show()  # 注释掉显示图形，避免中文字体问题
    plt.savefig('lstm_normalization_comparison.png', dpi=150, bbox_inches='tight')
    print("图表已保存到 lstm_normalization_comparison.png")
    
    # 打印最终结果总结
    print("\n=== 实验结果总结 ===")
    final_losses = {name: history[-1] for name, history in loss_histories.items()}
    sorted_results = sorted(final_losses.items(), key=lambda x: x[1])
    
    print("按最终损失排序（从小到大）:")
    for i, (name, loss) in enumerate(sorted_results):
        print(f"{i+1}. {name}: {loss:.6f}")
    
    # 计算相对改进
    base_loss = final_losses['Base (无归一化)']
    print(f"\n相对于基础模型的改进:")
    for name, loss in final_losses.items():
        if name != 'Base (无归一化)':
            improvement = (base_loss - loss) / base_loss * 100
            print(f"{name}: {improvement:+.2f}% {'✓' if improvement > 0 else '✗'}")
    
    return loss_histories, trained_models

if __name__ == "__main__":
    # 运行对比实验
    loss_histories, trained_models = run_comparison_experiment()


