%% =============================================================
%  深度学习通信信号数据集生成器 (M-ary 随机进制增强版)
%  总共生成 8 个 CSV 文件，每个文件 16000 样本，随机打乱
% =============================================================
clear; clc; close all;

%% 1. 全局参数初始化
rng(42); % ⚠️ 核心要求：全局随机种子设为 42

N = 128;               % 采样点数
sps = 8;               % 过采样率
num_symbols = N / sps; % 符号数 = 16

% [修改] 样本总数微调为 16000，确保能被 4(调制) x 4(信道) = 16 整除
total_samples = 16000; 

% [修改] 移除了 DPSK，剩余 4 类
mod_names = {'ASK', 'FSK', 'PSK', 'LFM'};
chan_names = {'AWGN', 'Rayleigh', 'Rician', 'Nakagami'};

% 每个文件 16 种组合 (4调制 x 4信道)，每种组合需生成的样本数 (1000个)
samples_per_cond = total_samples / (length(mod_names) * length(chan_names)); 

% 定义要生成的文件任务
snr_tasks = {-10, -5, 0, 5, 10, 15, 20, 'mixed'}; 

%% 2. 构造 CSV 表头 (261 列)
varNames = cell(1, 261);
for i = 1:128
    varNames{i} = sprintf('I_%d', i-1);       
    varNames{i+128} = sprintf('Q_%d', i-1);   
end
varNames{257} = 'mod_label';
varNames{258} = 'chan_label';
varNames{259} = 'mod_name';
varNames{260} = 'chan_name';
varNames{261} = 'snr';

%% 3. 开始批量生成数据
t = (0:N-1)'; % 时间索引向量
M_options = [2, 4, 8, 16, 32, 64]; % [新增] 候选进制集合

for task_idx = 1:length(snr_tasks)
    current_task = snr_tasks{task_idx};
    
    if isnumeric(current_task)
        filename = sprintf('dataset_snr_%d.csv', current_task);
        disp(['正在生成固定 SNR 数据集: ', filename]);
    else
        filename = 'dataset_snr_mixed.csv';
        disp(['正在生成混合 SNR 数据集: ', filename]);
    end
    
    I_data = zeros(total_samples, 128);
    Q_data = zeros(total_samples, 128);
    num_labels = zeros(total_samples, 3); 
    str_labels = strings(total_samples, 2); 
    
    idx = 1;
    
    % 遍历 4 种调制方式 (0到3)
    for mod_idx = 0:3
        for chan_idx = 0:3
            for s = 1:samples_per_cond
                
                fc = 0.05 + rand() * 0.10; 
                
                % [新增] 为当前的样本随机分配一个进制 M
                M = M_options(randi(length(M_options)));
                
                % --- A. 生成纯净基带符号 (多进制映射) ---
                switch mod_idx
                    case 0 % M-ASK (M-PAM)
                        data = randi([0 M-1], num_symbols, 1);
                        % 映射到对称的双极性幅度，例如 M=4 时映射为 [-3, -1, 1, 3]
                        amps = 2 * data - (M - 1); 
                        syms = repelem(amps, sps);
                        
                    case 1 % M-FSK (CPFSK)
                        data = randi([0 M-1], num_symbols, 1);
                        % 频率偏移映射
                        freq_dev = 2 * data - (M - 1); 
                        syms = exp(1j * cumsum(repelem(freq_dev, sps)) * (pi / sps * 0.5));
                        
                    case 2 % M-PSK
                        data = randi([0 M-1], num_symbols, 1);
                        % 通用 M-PSK 相位映射，加入 pi/M 初始偏移让星座图更居中对称
                        syms = repelem(exp(1j * (pi/M + data * 2*pi/M)), sps);
                        
                    case 3 % LFM (Chirp) - 与进制无关，保持原样
                        syms = exp(1j * 2 * pi * (-0.2 * t + 0.4 / (2 * N) * t.^2));
                end
                
                % 上变频
                sig = syms .* exp(1j * 2 * pi * fc * t);
                
                % 功率归一化 (这一步非常关键，它会自动把 M-ASK 不同幅度的平均功率拉平到 1)
                sig = sig / sqrt(mean(abs(sig).^2)); 
                
                % --- B. 经过衰落信道 ---
                switch chan_idx
                    case 0 % AWGN
                        h = 1; 
                    case 1 % Rayleigh
                        h = (randn() + 1j * randn()) / sqrt(2);
                    case 2 % Rician
                        K = 1.0 + rand() * 9.0;
                        mu = sqrt(K / (K + 1));
                        sigma = sqrt(1 / (2 * (K + 1)));
                        h = (mu + sigma * randn()) + 1j * (sigma * randn());
                    case 3 % Nakagami
                        m = 0.5 + rand() * 2.5;
                        h_amp = sqrt(gamrnd(m, 1/m)); 
                        h_phase = rand() * 2 * pi;
                        h = h_amp * exp(1j * h_phase);
                end
                sig = sig * h;
                
                % --- C. 叠加噪声 ---
                if isnumeric(current_task)
                    current_snr = current_task;
                else
                    current_snr = -10 + rand() * 30; 
                end
                
                sig_power = mean(abs(sig).^2);
                noise_power = sig_power / (10^(current_snr / 10));
                noise = sqrt(noise_power / 2) * (randn(size(sig)) + 1j * randn(size(sig)));
                rx_sig = sig + noise;
                
                % --- D. 数据记录 ---
                I_data(idx, :) = real(rx_sig)';
                Q_data(idx, :) = imag(rx_sig)';
                num_labels(idx, :) = [mod_idx, chan_idx, current_snr];
                str_labels(idx, :) = [string(mod_names{mod_idx+1}), string(chan_names{chan_idx+1})];
                
                idx = idx + 1;
            end
        end
    end
    
    % --- E. 样本打乱与导出 ---
    shuffle_idx = randperm(total_samples);
    
    T_num = array2table([I_data(shuffle_idx, :), Q_data(shuffle_idx, :), num_labels(shuffle_idx, 1:2)], ...
        'VariableNames', varNames(1:258));
    T_str = table(str_labels(shuffle_idx, 1), str_labels(shuffle_idx, 2), ...
        'VariableNames', varNames(259:260));
    T_snr = table(num_labels(shuffle_idx, 3), 'VariableNames', varNames(261));
    
    T_final = [T_num, T_str, T_snr];
    writetable(T_final, filename);
    disp(['  -> 完成保存: ', filename]);
end

disp('✅ 所有 8 个多进制数据集文件生成完毕！');