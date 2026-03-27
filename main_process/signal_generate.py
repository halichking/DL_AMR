"""
调制识别数据集生成脚本
信号类型: ASK, FSK, PSK, DPSK, LFM
信道类型: AWGN, Rayleigh, Rician, Nakagami
输出格式: 单个 dataset.csv 文件
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================
# 参数配置
# ============================================================
N_TOTAL           = 15000
N_POINTS          = 128               # 每个样本的采样点数
FC                = 0.1               # 归一化载波频率
SNR_LEVELS        = [-10, -5, 0, 5, 10, 15, 20]   # dB
MOD_TYPES         = ['ASK', 'FSK', 'PSK', 'DPSK', 'LFM']
CHAN_TYPES        = ['AWGN', 'Rayleigh', 'Rician', 'Nakagami']
SAVE_PATH         = './dataset.csv'

N_COMBOS          = len(MOD_TYPES) * len(CHAN_TYPES) * len(SNR_LEVELS)  # 140
SAMPLES_PER_COMBO = N_TOTAL // N_COMBOS   # ≈107

MOD_LABEL  = {m: i for i, m in enumerate(MOD_TYPES)}
CHAN_LABEL  = {c: i for i, c in enumerate(CHAN_TYPES)}


# ============================================================
# 信号生成
# ============================================================

def generate_ask(n, M=4):
    sps = 8
    syms = np.random.randint(0, M, n // sps)
    amp  = 2 * syms / (M - 1) - 1
    base = np.repeat(amp, sps)[:n]
    t    = np.arange(n)
    sig  = base * np.exp(1j * 2 * np.pi * FC * t)
    return sig / (np.std(np.abs(sig)) + 1e-8)

def generate_fsk(n, M=4):
    sps   = 8
    syms  = np.random.randint(0, M, n // sps)
    freqs = np.linspace(0.05, 0.20, M)
    sig   = np.zeros(n, dtype=complex)
    for i, s in enumerate(syms):
        t_seg = np.arange(i * sps, min((i + 1) * sps, n))
        sig[i * sps: i * sps + len(t_seg)] = np.exp(1j * 2 * np.pi * freqs[s] * t_seg)
    return sig / (np.std(np.abs(sig)) + 1e-8)

def generate_psk(n, M=4):
    sps   = 8
    syms  = np.random.randint(0, M, n // sps)
    phase = np.repeat(2 * np.pi * syms / M, sps)[:n]
    t     = np.arange(n)
    sig   = np.exp(1j * (2 * np.pi * FC * t + phase))
    return sig / (np.std(np.abs(sig)) + 1e-8)

def generate_dpsk(n, M=2):
    sps   = 8
    syms  = np.random.randint(0, M, n // sps)
    phase = np.repeat(np.cumsum(np.pi * syms), sps)[:n]
    t     = np.arange(n)
    sig   = np.exp(1j * (2 * np.pi * FC * t + phase))
    return sig / (np.std(np.abs(sig)) + 1e-8)

def generate_lfm(n):
    t   = np.arange(n) / n
    sig = np.exp(1j * 2 * np.pi * (0.05 * t + 0.20 * t ** 2))
    return sig / (np.std(np.abs(sig)) + 1e-8)

SIGNAL_GEN = {
    'ASK' : generate_ask,
    'FSK' : generate_fsk,
    'PSK' : generate_psk,
    'DPSK': generate_dpsk,
    'LFM' : generate_lfm,
}


# ============================================================
# 信道模型
# ============================================================

def add_noise(sig, snr_db):
    power = np.mean(np.abs(sig) ** 2) / (10 ** (snr_db / 10))
    noise = np.sqrt(power / 2) * (np.random.randn(len(sig)) + 1j * np.random.randn(len(sig)))
    return sig + noise

def awgn(sig, snr_db):
    return add_noise(sig, snr_db)

def rayleigh(sig, snr_db):
    h = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
    return add_noise(h * sig, snr_db)

def rician(sig, snr_db, K=3.0):
    los     = np.sqrt(K / (K + 1)) * np.exp(1j * 2 * np.pi * np.random.rand())
    scatter = np.sqrt(1 / (K + 1)) * (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
    return add_noise((los + scatter) * sig, snr_db)

def nakagami(sig, snr_db, m=2.0):
    h = np.sqrt(np.random.gamma(m, 1.0 / m)) * np.exp(1j * 2 * np.pi * np.random.rand())
    return add_noise(h * sig, snr_db)

CHANNEL_FN = {
    'AWGN'    : awgn,
    'Rayleigh': rayleigh,
    'Rician'  : rician,
    'Nakagami': nakagami,
}


# ============================================================
# 数据集生成
# ============================================================

if __name__ == '__main__':
    np.random.seed(42)

    print(f"信号类型 : {MOD_TYPES}")
    print(f"信道类型 : {CHAN_TYPES}")
    print(f"SNR 范围 : {SNR_LEVELS} dB")
    print(f"总组合数 : {N_COMBOS}，每组合样本数 : {SAMPLES_PER_COMBO}")
    print(f"总样本数 : {N_COMBOS * SAMPLES_PER_COMBO}\n")

    # 列名：I_0~I_127，Q_0~Q_127，加4个标签列
    columns = (
        [f'I_{i}' for i in range(N_POINTS)] +
        [f'Q_{i}' for i in range(N_POINTS)] +
        ['mod_label', 'chan_label', 'mod_name', 'chan_name', 'snr']
    )

    rows = []

    with tqdm(total=N_COMBOS, desc="生成中") as pbar:
        for mod in MOD_TYPES:
            for chan in CHAN_TYPES:
                for snr in SNR_LEVELS:
                    for _ in range(SAMPLES_PER_COMBO):
                        sig = SIGNAL_GEN[mod](N_POINTS)
                        rx  = CHANNEL_FN[chan](sig, snr)
                        row = (
                            rx.real.astype(np.float32).tolist() +
                            rx.imag.astype(np.float32).tolist() +
                            [MOD_LABEL[mod], CHAN_LABEL[chan], mod, chan, snr]
                        )
                        rows.append(row)
                    pbar.update(1)

    df = pd.DataFrame(rows, columns=columns)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(SAVE_PATH, index=False)

    print(f"\n已保存至 {SAVE_PATH}")
    print(f"CSV 大小: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"\n列结构:")
    print(f"  I_0  ~ I_127  →  I 路信号（128列）")
    print(f"  Q_0  ~ Q_127  →  Q 路信号（128列）")
    print(f"  mod_label     →  调制标签（整数）{MOD_LABEL}")
    print(f"  chan_label    →  信道标签（整数）{CHAN_LABEL}")
    print(f"  mod_name      →  调制类型名称")
    print(f"  chan_name     →  信道类型名称")
    print(f"  snr           →  信噪比（dB）")