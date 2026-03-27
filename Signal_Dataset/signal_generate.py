import numpy as np
import pandas as pd
import os

# =============================================================
#  深度学习通信信号数据集生成器 (M-ary 随机进制增强版) - Python 版
#  总共生成 8 个 CSV 文件，每个文件 16000 样本，随机打乱
# =============================================================

# 1. 全局参数初始化
np.random.seed(42)  # ⚠️ 核心要求：全局随机种子设为 42

N = 128  # 采样点数
sps = 8  # 过采样率
num_symbols = N // sps  # 符号数 = 16
total_samples = 16000  # 样本总数

mod_names = ['ASK', 'FSK', 'PSK', 'LFM']
chan_names = ['AWGN', 'Rayleigh', 'Rician', 'Nakagami']

# 每个文件 16 种组合 (4调制 x 4信道)，每种组合需生成的样本数 (1000个)
samples_per_cond = total_samples // (len(mod_names) * len(chan_names))

# 定义要生成的文件任务
snr_tasks = [-10, -5, 0, 5, 10, 15, 20, 'mixed']
M_options = [2, 4, 8, 16, 32, 64]  # 候选进制集合

# 2. 构造 CSV 表头 (261 列)
varNames = []
for i in range(128):
    varNames.append(f'I_{i}')
for i in range(128):
    varNames.append(f'Q_{i}')
varNames.extend(['mod_label', 'chan_label', 'mod_name', 'chan_name', 'snr'])

# 时间索引向量
t = np.arange(N)

# 3. 开始批量生成数据
for task_idx, current_task in enumerate(snr_tasks):
    if isinstance(current_task, (int, float)):
        filename = f'dataset_snr_{current_task}.csv'
        print(f"正在生成固定 SNR 数据集: {filename}")
    else:
        filename = 'dataset_snr_mixed.csv'
        print(f"正在生成混合 SNR 数据集: {filename}")

    # 预分配 NumPy 数组以提高性能
    I_data = np.zeros((total_samples, 128))
    Q_data = np.zeros((total_samples, 128))

    # 记录标签: [mod_idx, chan_idx, snr]
    num_labels = np.zeros((total_samples, 3))
    # 记录字符串标签: [mod_name, chan_name]
    str_labels = np.empty((total_samples, 2), dtype=object)

    idx = 0

    # 遍历 4 种调制方式 (0到3)
    for mod_idx in range(4):
        for chan_idx in range(4):
            for s in range(samples_per_cond):

                fc = 0.05 + np.random.rand() * 0.10

                # 为当前的样本随机分配一个进制 M
                M = np.random.choice(M_options)

                # --- A. 生成纯净基带符号 (多进制映射) ---
                if mod_idx == 0:  # M-ASK (M-PAM)
                    data = np.random.randint(0, M, num_symbols)
                    amps = 2 * data - (M - 1)
                    syms = np.repeat(amps, sps)

                elif mod_idx == 1:  # M-FSK (CPFSK)
                    data = np.random.randint(0, M, num_symbols)
                    freq_dev = 2 * data - (M - 1)
                    syms = np.exp(1j * np.cumsum(np.repeat(freq_dev, sps)) * (np.pi / sps * 0.5))

                elif mod_idx == 2:  # M-PSK
                    data = np.random.randint(0, M, num_symbols)
                    syms = np.repeat(np.exp(1j * (np.pi / M + data * 2 * np.pi / M)), sps)

                elif mod_idx == 3:  # LFM (Chirp)
                    syms = np.exp(1j * 2 * np.pi * (-0.2 * t + 0.4 / (2 * N) * t ** 2))

                # 上变频
                sig = syms * np.exp(1j * 2 * np.pi * fc * t)

                # 功率归一化
                sig = sig / np.sqrt(np.mean(np.abs(sig) ** 2))

                # --- B. 经过衰落信道 ---
                if chan_idx == 0:  # AWGN
                    h = 1
                elif chan_idx == 1:  # Rayleigh
                    h = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
                elif chan_idx == 2:  # Rician
                    K = 1.0 + np.random.rand() * 9.0
                    mu = np.sqrt(K / (K + 1))
                    sigma = np.sqrt(1 / (2 * (K + 1)))
                    h = (mu + sigma * np.random.randn()) + 1j * (sigma * np.random.randn())
                elif chan_idx == 3:  # Nakagami
                    m = 0.5 + np.random.rand() * 2.5
                    # Python 的 gamma 分布: shape=m, scale=1/m 对应 MATLAB 的 gamrnd(m, 1/m)
                    h_amp = np.sqrt(np.random.gamma(shape=m, scale=1 / m))
                    h_phase = np.random.rand() * 2 * np.pi
                    h = h_amp * np.exp(1j * h_phase)

                sig = sig * h

                # --- C. 叠加噪声 ---
                if isinstance(current_task, (int, float)):
                    current_snr = current_task
                else:
                    current_snr = -10 + np.random.rand() * 30

                sig_power = np.mean(np.abs(sig) ** 2)
                noise_power = sig_power / (10 ** (current_snr / 10))
                noise = np.sqrt(noise_power / 2) * (np.random.randn(N) + 1j * np.random.randn(N))
                rx_sig = sig + noise

                # --- D. 数据记录 ---
                I_data[idx, :] = np.real(rx_sig)
                Q_data[idx, :] = np.imag(rx_sig)
                num_labels[idx, :] = [mod_idx, chan_idx, current_snr]
                str_labels[idx, :] = [mod_names[mod_idx], chan_names[chan_idx]]

                idx += 1

    # --- E. 样本打乱与导出 ---
    # 生成 0 到 total_samples-1 的随机索引序列
    shuffle_idx = np.random.permutation(total_samples)

    # 根据随机索引打乱所有数组
    I_data_shuffled = I_data[shuffle_idx, :]
    Q_data_shuffled = Q_data[shuffle_idx, :]
    num_labels_shuffled = num_labels[shuffle_idx, :]
    str_labels_shuffled = str_labels[shuffle_idx, :]

    # 拼装数据框 DataFrame
    df = pd.DataFrame(
        np.hstack((I_data_shuffled, Q_data_shuffled)),
        columns=varNames[:256]
    )
    df['mod_label'] = num_labels_shuffled[:, 0].astype(int)
    df['chan_label'] = num_labels_shuffled[:, 1].astype(int)
    df['mod_name'] = str_labels_shuffled[:, 0]
    df['chan_name'] = str_labels_shuffled[:, 1]
    df['snr'] = num_labels_shuffled[:, 2]

    # 导出 CSV
    df.to_csv(filename, index=False)
    print(f"  -> 完成保存: {filename}")

print('✅ 所有 8 个多进制数据集文件生成完毕！')