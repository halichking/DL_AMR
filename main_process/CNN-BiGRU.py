import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import TensorDataset, DataLoader
import random

# ============================================================
# [新增] 固定全局随机数种子的函数
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 保证卷积等操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 调用函数，锁死种子为 42
set_seed(42)

# ============================================================
# 0. 全局配置
# ============================================================
# 数据集路径（根据实际路径修改）
DATA_DIR  = '../Signal_Dataset'
# 7个固定SNR数据集 + 1个混合SNR数据集
SNR_LIST  = [-10, -5, 0, 5, 10, 15, 20]
MOD_NAMES = ['ASK', 'FSK', 'PSK', 'LFM']   # 标签对应顺序

# 结果保存目录
RESULT_DIR = 'CNN-BiGRU_results'
os.makedirs(RESULT_DIR, exist_ok=True)

# 超参数
LR         = 0.004
EPOCH_NUM  = 20
BATCH_SIZE = 128


# ============================================================
# 1. 加载数据
# ============================================================
def create_dataset(csv_path):
    # 1.1 读取CSV文件
    df = pd.read_csv(csv_path)

    # 1.2 提取I路和Q路信号作为特征，共256列
    i_cols = [f'I_{i}' for i in range(128)]
    q_cols = [f'Q_{i}' for i in range(128)]
    X = df[i_cols + q_cols].values           # (N, 256)
    X = X.reshape(-1, 2, 128)               # (N, 2, 128)

    # 1.3 每个样本单独标准化，消除不同SNR下幅度差异的影响
    mean = X.mean(axis=(1, 2), keepdims=True)
    std  = X.std(axis=(1, 2),  keepdims=True) + 1e-8
    X    = (X - mean) / std

    # 1.4 提取调制类型标签
    y = df['mod_label'].values               # (N,)

    # 1.5 划分训练集(80%)和测试集(20%)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1.6 转换为Tensor
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test  = torch.tensor(x_test,  dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_test  = torch.tensor(y_test,  dtype=torch.int64)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset  = TensorDataset(x_test,  y_test)
    return train_dataset, test_dataset


# ============================================================
# 2. 定义模型：CNN + GRU
# ============================================================
class CNN_GRU(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # --- CNN 部分：提取局部时域特征 ---
        # 输入形状: (N, 2, 128)，2为IQ两路
        self.cnn = nn.Sequential(
            # 第一层卷积：2通道 → 32通道
            nn.Conv1d(in_channels=2,  out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),      # 128 → 64

            # 第二层卷积：32通道 → 64通道
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),      # 64 → 32
        )
        # CNN输出形状: (N, 64, 32)

        # --- GRU 部分：捕捉时序依赖 ---
        # 将(N, 64, 32)转置为(N, 32, 64)，即32个时间步
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        # 取最后一个时间步输出: (N, 128)

        # --- 全连接分类层 ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)                      # (N, 64, 32)
        x = x.permute(0, 2, 1)              # (N, 32, 64)
        x, _ = self.gru(x)                  # (N, 32, 128)
        # x = x[:, -1, :]                     # (N, 128) 取最后时间步
        x = x.mean(dim=1)  # 聚合 32 个时间步的所有信息 -> (N, 256)
        x = self.classifier(x)              # (N, num_classes)
        return x


# ============================================================
# 3. 训练与测试函数
# ============================================================
def train_test(model, train_dataset, test_dataset, lr, epoch_num, batch_size, device):

    # 3.1 Xavier参数初始化
    def init_weights(layer):
        if isinstance(layer, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(layer.weight)

    model.apply(init_weights)
    model.to(device)

    # 3.2 损失函数、优化器、学习率调度
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH_NUM, eta_min=1e-6)

    train_loss_list, test_loss_list = [], []
    train_acc_list,  test_acc_list  = [], []

    # 3.3 迭代训练
    for epoch in range(epoch_num):

        # ---- 训练阶段 ----
        model.train()
        train_loader      = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loss_total  = 0
        train_correct_num = 0

        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            output     = model(X)
            loss_value = loss_fn(output, y)
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss_total  += loss_value.item() * X.shape[0]
            pred               = output.argmax(dim=1)
            train_correct_num += pred.eq(y).sum().item()

            print(f"\rEpoch:{epoch+1:0>2} [{'='*int((batch_idx+1)/len(train_loader)*40):<40}]", end="")

        this_train_loss = train_loss_total  / len(train_dataset)
        this_train_acc  = train_correct_num / len(train_dataset)
        train_loss_list.append(this_train_loss)
        train_acc_list.append(this_train_acc)

        # ---- 测试阶段 ----
        model.eval()
        test_loader      = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_loss_total  = 0
        test_correct_num = 0

        with torch.no_grad():
            for X, y in test_loader:
                X, y   = X.to(device), y.to(device)
                output = model(X)
                loss_value = loss_fn(output, y)
                test_loss_total  += loss_value.item() * X.shape[0]
                pred              = output.argmax(dim=1)
                test_correct_num += pred.eq(y).sum().item()

        this_test_loss = test_loss_total  / len(test_dataset)
        this_test_acc  = test_correct_num / len(test_dataset)
        test_loss_list.append(this_test_loss)
        test_acc_list.append(this_test_acc)

        scheduler.step()

        print(f"  train_loss:{this_train_loss:.4f}  train_acc:{this_train_acc:.4f}"
              f"  test_loss:{this_test_loss:.4f}  test_acc:{this_test_acc:.4f}")

    return train_loss_list, test_loss_list, train_acc_list, test_acc_list


# ============================================================
# 4. 收集测试集上所有预测结果（用于混淆矩阵）
# ============================================================
def get_predictions(model, test_dataset, batch_size, device):
    model.eval()
    loader    = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    all_true  = []
    with torch.no_grad():
        for X, y in loader:
            X      = X.to(device)
            output = model(X)
            preds  = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y.numpy())
    return np.array(all_true), np.array(all_preds)


# ============================================================
# 5. 画图并保存（Loss曲线 + 准确率曲线 + 混淆矩阵）
# ============================================================
def plot_and_save(train_loss_list, test_loss_list,
                  train_acc_list,  test_acc_list,
                  y_true, y_pred, tag):
    """
    一张图包含三个子图：Loss曲线、准确率曲线、混淆矩阵
    tag: 图片文件名前缀，如 'snr_10' 或 'mixed'
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Results  [{tag}]', fontsize=14)

    # ---- 子图1：Loss曲线 ----
    axes[0].plot(train_loss_list, 'r-',  label='train loss', linewidth=2)
    axes[0].plot(test_loss_list,  'b--', label='test loss',  linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()

    # ---- 子图2：准确率曲线 ----
    axes[1].plot(train_acc_list, 'r-',  label='train acc', linewidth=2)
    axes[1].plot(test_acc_list,  'b--', label='test acc',  linewidth=2)
    # [新增] 获取最后一个 epoch 的测试准确率并打上标记
    final_epoch = len(test_acc_list) - 1
    final_acc = test_acc_list[-1]
    # 画一个黑点突出最后的位置
    axes[1].plot(final_epoch, final_acc, 'ko', markersize=5)
    # 动态调整文字的垂直偏移量（准确率高时文字放下面，低时放上面）
    y_offset = -0.02 if final_acc > 0.8 else 0.02
    # 使用纯文本在点附近标注百分比，例如 "48.81%"
    axes[1].text(final_epoch-0.25, final_acc + y_offset,
                 f'{final_acc * 100:.2f}%',
                 ha='center', va='center', fontsize=10, color='black')
    # # 添加带箭头的文本注释
    # axes[1].annotate(f'{final_acc:.4f}',
    #                  xy=(final_epoch, final_acc),
    #                  xytext=(-30, -25) if final_acc > 0.8 else (-30, 20),  # 如果准确率太高文字就往下放，防止出界
    #                  textcoords='offset points',
    #                  ha='center',
    #                  fontsize=10,
    #                  color='black',
    #                  arrowprops=dict(arrowstyle='->', color='black'))
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()

    # ---- 子图3：混淆矩阵 ----
    # 行为真实标签，列为预测标签，对角线越亮说明分类越准确
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=MOD_NAMES)
    disp.plot(ax=axes[2], colorbar=False, cmap='Blues')
    axes[2].set_title('Confusion Matrix')

    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, f'{tag}.png')
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"图片已保存: {save_path}\n")


# ============================================================
# 6. 主流程
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}\n")

# 用于最终汇总 SNR vs 准确率
summary_snr_list = []
summary_acc_list = []

# ------ 6.1 依次训练7个固定SNR数据集 ------
for snr in SNR_LIST:
    tag      = f'snr_{snr}'
    csv_name = f'dataset_snr_{snr}.csv'
    csv_path = os.path.join(DATA_DIR, csv_name)

    print(f"\n{'='*60}")
    print(f"  开始训练：SNR = {snr} dB  ({csv_name})")
    print(f"{'='*60}")

    # 加载当前SNR的数据集
    train_dataset, test_dataset = create_dataset(csv_path)

    # 每次重新初始化一个新模型（从头训练，互不干扰）
    model = CNN_GRU(num_classes=4)

    # 训练与测试
    train_loss_list, test_loss_list, train_acc_list, test_acc_list = train_test(
        model, train_dataset, test_dataset, LR, EPOCH_NUM, BATCH_SIZE, device
    )

    # 获取预测结果，用于绘制混淆矩阵
    y_true, y_pred = get_predictions(model, test_dataset, BATCH_SIZE, device)

    # 最终测试准确率
    final_acc = test_acc_list[-1]
    print(f"\nSNR={snr:4d}dB  最终测试准确率: {final_acc:.4f}")

    # 画图并保存
    plot_and_save(train_loss_list, test_loss_list,
                  train_acc_list,  test_acc_list,
                  y_true, y_pred, tag)

    # 记录到汇总列表
    summary_snr_list.append(snr)
    summary_acc_list.append(final_acc)


# ------ 6.2 训练混合SNR数据集 ------
print(f"\n{'='*60}")
print(f"  开始训练：混合SNR数据集  (dataset_snr_mixed.csv)")
print(f"{'='*60}")

csv_path = os.path.join(DATA_DIR, 'dataset_snr_mixed.csv')
train_dataset, test_dataset = create_dataset(csv_path)
model = CNN_GRU(num_classes=4)

train_loss_list, test_loss_list, train_acc_list, test_acc_list = train_test(
    model, train_dataset, test_dataset, LR, EPOCH_NUM, BATCH_SIZE, device
)

y_true, y_pred     = get_predictions(model, test_dataset, BATCH_SIZE, device)
final_acc_mixed    = test_acc_list[-1]
print(f"\n混合SNR  最终测试准确率: {final_acc_mixed:.4f}")

plot_and_save(train_loss_list, test_loss_list,
              train_acc_list,  test_acc_list,
              y_true, y_pred, 'mixed')


# ============================================================
# 7. 汇总图：准确率 vs SNR 曲线（论文核心图）
# ============================================================
plt.figure(figsize=(8, 5))
plt.plot(summary_snr_list, summary_acc_list, 'bo-', linewidth=2,
         markersize=8, label='CNN+BiGRU (fixed SNR)')

# [新增] 遍历所有的 SNR 和对应的准确率，把数值标在点的上方
for x, y in zip(summary_snr_list, summary_acc_list):
    plt.text(x, y + 0.02, f'{y:.4f}', ha='center', va='bottom', fontsize=10, color='darkblue')

# 混合SNR准确率画一条水平参考线
plt.axhline(y=final_acc_mixed, color='r', linestyle='--', linewidth=2,
            label=f'CNN+BiGRU (mixed SNR) = {final_acc_mixed:.4f}')
# [新增可选] 如果你也想把混合 SNR 的值标在红线上，可以加下面这行
plt.text(summary_snr_list[0], final_acc_mixed + 0.02, f'{final_acc_mixed:.4f}', color='red', fontsize=10)

plt.xlabel('SNR (dB)')
plt.ylabel('Test Accuracy')
plt.title('Accuracy vs SNR')
plt.xticks(summary_snr_list)
plt.ylim(0, 1.1)
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
summary_path = os.path.join(RESULT_DIR, 'accuracy_vs_snr.png')
plt.savefig(summary_path, dpi=150)
plt.show()
print(f"汇总图已保存: {summary_path}")

# 打印最终汇总表
print("\n========== 汇总结果 ==========")
for snr, acc in zip(summary_snr_list, summary_acc_list):
    print(f"  SNR={snr:4d}dB  →  test_acc={acc:.4f}")
print(f"  Mixed SNR  →  test_acc={final_acc_mixed:.4f}")
print("================================")