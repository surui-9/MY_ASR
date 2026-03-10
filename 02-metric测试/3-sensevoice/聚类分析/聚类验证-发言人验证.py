import pandas as pd
import numpy as np

# 加载你的聚类结果文件（替换成实际路径）
df = pd.read_csv("sensevoice_high_cer_full_sample_clusters.csv", encoding="utf-8")

# 假设你的数据里有「发言人ID/音频文件名包含发言人标识」（比如audio_001_xxx.wav里的001是发言人ID）
# 方法1：如果有明确的speaker_id字段
if "speaker_id" in df.columns:
    # 统计每个簇对应的发言人数量
    cluster_speaker = df.groupby("cluster")["speaker_id"].nunique()
    print("=== 各簇对应的发言人数量 ===")
    print(cluster_speaker)

    # 统计每个发言人对应的簇（如果1个发言人只在1个簇里，说明簇=发言人）
    speaker_cluster = df.groupby("speaker_id")["cluster"].nunique()
    print("\n=== 每个发言人跨簇数量（1=仅在1个簇，>1=跨簇） ===")
    print(speaker_cluster.value_counts())

# 方法2：如果没有speaker_id，从音频文件名提取（比如文件名格式：speaker_01_xxx.wav）
else:
    # 自定义提取逻辑（根据你的音频文件名格式调整）
    def extract_speaker(audio_path):
        # 示例：从"G:/data/speaker_05_123.wav"提取"05"
        filename = audio_path.split("/")[-1]
        speaker_id = filename.split("_")[1]  # 按你的文件名格式调整
        return speaker_id


    df["speaker_id"] = df["full_audio_path"].apply(extract_speaker)
    cluster_speaker = df.groupby("cluster")["speaker_id"].nunique()
    print("=== 各簇对应的发言人数量 ===")
    print(cluster_speaker)