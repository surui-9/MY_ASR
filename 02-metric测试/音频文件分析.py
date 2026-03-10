import os

# 根目录（替换为你的sichuan文件夹路径）
root_dir = r"/02-metric测试\sichuan\wav"

# 遍历所有子文件夹，收集wav文件路径
audio_paths = []
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".wav"):
            audio_paths.append(os.path.join(subdir, file))

print(f"共找到 {len(audio_paths)} 个wav音频文件")
# print("前5个文件路径：", audio_paths[:5])


import librosa
import numpy as np
import pandas as pd

# 初始化分析结果
local_analysis = {
    "file_path": [],
    "duration": [],  # 时长（秒）
    "sampling_rate": [],  # 采样率
    "bit_depth": [],  # 位深（16bit）
    "snr": [],  # 信噪比
    "silence_ratio": [],  # 静音占比
    "is_distorted": []  # 是否失真
}

for path in audio_paths[:1000]:  # 先采样1000条，全量可去掉[:1000]
    # 加载音频（librosa自动转成float32，不影响分析）
    y, sr = librosa.load(path, sr=None)  # sr=None保留原采样率

    # 1. 时长
    duration = len(y) / sr
    # 2. 采样率
    # 3. 位深：wav文件16bit对应原数据范围[-32768,32767]，加载后是[-1,1]，所以位深=16
    bit_depth = 16
    # 4. 信噪比
    silence_len = int(len(y) * 0.1)
    if silence_len > 0:
        noise = np.concatenate([y[:silence_len], y[-silence_len:]])
        snr = 10 * np.log10(np.sum(y ** 2) / np.sum(noise ** 2)) if np.sum(noise ** 2) > 0 else 0
    else:
        snr = 0
    # 5. 静音占比
    silence_ratio = np.sum(np.abs(y) < 0.01) / len(y)
    # 6. 是否失真（振幅超出[-1,1]）
    is_distorted = np.max(y) > 1 or np.min(y) < -1

    # 存入结果
    local_analysis["file_path"].append(path)
    local_analysis["duration"].append(duration)
    local_analysis["sampling_rate"].append(sr)
    local_analysis["bit_depth"].append(bit_depth)
    local_analysis["snr"].append(snr)
    local_analysis["silence_ratio"].append(silence_ratio)
    local_analysis["is_distorted"].append(is_distorted)

# 转成DataFrame看统计结果
local_df = pd.DataFrame(local_analysis)
print("本地音频基础特征统计：")
print(
    f"时长：平均{local_df['duration'].mean():.2f}s，最小{local_df['duration'].min():.2f}s，最大{local_df['duration'].max():.2f}s")
print(f"采样率：{local_df['sampling_rate'].unique()}（需统一为16000）")
print(f"信噪比：平均{local_df['snr'].mean():.2f}dB（<0dB为噪声大）")
print(f"静音占比：平均{local_df['silence_ratio'].mean():.2f}")
print(f"失真文件数：{local_df['is_distorted'].sum()}个")