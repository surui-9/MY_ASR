import json
import numpy as np
import pandas as pd
import librosa
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 解决Windows MKL内存泄漏警告 + 修复OMP_NUM_THREADS
os.environ["OMP_NUM_THREADS"] = "8"  # 按警告提示设为8，消除内存泄漏警告
os.environ["MKL_NUM_THREADS"] = "8"

# ====================== 1. 基础配置 ======================
report_path = "../测评结果9-保留音译字.json"
high_cer_threshold = 0.1ll

# ====================== 2. 中文字体配置 ======================
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ====================== 3. 加载数据 + 验证路径 ======================
with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

all_samples = report["all_samples_results"] if "all_samples_results" in report else report["top10_worst_cases"]
df = pd.DataFrame(all_samples)

# 筛选高CER样本
high_cer_df = df[df["cer"] > high_cer_threshold].copy()
print(f"筛选出高CER样本总数：{len(high_cer_df)} 条")
print(f"高CER样本平均CER：{high_cer_df['cer'].mean():.4f}")

# 验证路径有效性
high_cer_df["full_audio_path"] = high_cer_df["audio_filepath"]
high_cer_df["audio_exists"] = high_cer_df["full_audio_path"].apply(os.path.exists)
high_cer_df_valid = high_cer_df[high_cer_df["audio_exists"]].copy()
print(f"有效高CER样本数（路径正确）：{len(high_cer_df_valid)} 条")

if len(high_cer_df_valid) == 0:
    raise ValueError("❌ 仍无有效音频文件！请确认JSON中的audio_filepath已正确")

# 全样本分析（无采样限制）
high_cer_sample = high_cer_df_valid.copy()
print(f"最终分析的高CER全样本数：{len(high_cer_sample)} 条")


# ====================== 4. 提取MFCC特征 ======================
def extract_mfcc(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
    except Exception as e:
        print(f"⚠️ 提取{os.path.basename(audio_path)}特征失败：{str(e)[:50]}")
        return np.zeros(26)


# 批量提取
print("\n开始提取所有高CER样本的MFCC特征...")
mfcc_vectors = [extract_mfcc(path) for path in high_cer_sample["full_audio_path"]]
mfcc_vectors = np.array(mfcc_vectors)
print(f"MFCC特征提取完成，形状：{mfcc_vectors.shape}")

# 归一化
mfcc_scaled = StandardScaler().fit_transform(mfcc_vectors)


# ====================== 5. 聚类分析 ======================
def find_best_k(vectors):
    max_k = min(10, len(vectors) - 1)
    if max_k < 2:
        print("⚠️ 样本数不足，默认K=2")
        return 2
    sil_scores = []
    inertia_list = []
    print(f"\n正在计算K=2到{max_k}的聚类效果...")
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # 增加n_init，提升稳定性
        labels = kmeans.fit_predict(vectors)
        sil_score = silhouette_score(vectors, labels) if len(set(labels)) > 1 else 0
        sil_scores.append(sil_score)
        inertia_list.append(kmeans.inertia_)
        print(f"K={k} → 轮廓系数：{sil_score:.4f} | 簇内误差：{kmeans.inertia_:.2f}")

    # 可视化
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(range(2, max_k + 1), sil_scores, 'o-', color='orange', linewidth=2)
    plt.title("轮廓系数（值越高聚类效果越好）", fontsize=12)
    plt.xlabel("聚类数K")
    plt.ylabel("轮廓系数")
    plt.grid(alpha=0.3)
    best_k_idx = sil_scores.index(max(sil_scores))
    plt.scatter(2 + best_k_idx, sil_scores[best_k_idx], color='red', s=100, zorder=5)

    plt.subplot(122)
    plt.plot(range(2, max_k + 1), inertia_list, 'o-', color='blue', linewidth=2)
    plt.title("肘部法则（下降拐点为最优K）", fontsize=12)
    plt.xlabel("聚类数K")
    plt.ylabel("簇内误差（Inertia）")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("best_k_full_sample.png", dpi=150, bbox_inches='tight')
    plt.show()

    best_k = 2 + best_k_idx
    print(f"\n最优聚类数K={best_k}（轮廓系数最大）")
    return best_k


# 执行聚类
best_k = find_best_k(mfcc_scaled)
kmeans_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
high_cer_sample["cluster"] = kmeans_model.fit_predict(mfcc_scaled)


# ====================== 6. 提取基础特征（修复NaN问题） ======================
def get_basic_features(row):
    try:
        audio_path = row["full_audio_path"]
        if not os.path.exists(audio_path):
            return 0.0, 0.0, 0.0

        y, sr = librosa.load(audio_path, sr=16000)
        if len(y) == 0:
            return 0.0, 0.0, 0.0

        duration = len(y) / sr
        # 信噪比计算（更鲁棒）
        silence_part = y[:int(len(y) * 0.1)] if len(y) > 100 else y
        silence_energy = np.sum(silence_part ** 2)
        signal_energy = np.sum(y ** 2)
        snr = 10 * np.log10(signal_energy / silence_energy) if (silence_energy != 0 and signal_energy != 0) else 0.0
        # 语速计算
        char_count = len(str(row["ref_text_cleaned"]).strip()) if pd.notna(row["ref_text_cleaned"]) else 0
        speed = char_count / duration if duration > 0 else 0.0

        return duration, speed, snr
    except Exception as e:
        print(f"⚠️ 提取{os.path.basename(row['full_audio_path'])}基础特征失败：{str(e)[:30]}")
        return 0.0, 0.0, 0.0  # 修复NaN：返回0而非空


# 批量提取
print("\n开始提取所有高CER样本的基础特征（时长/语速/信噪比）...")
features = [get_basic_features(row) for _, row in high_cer_sample.iterrows()]
high_cer_sample[["duration", "speed", "snr"]] = pd.DataFrame(features)

# 填充可能的剩余NaN（兜底）
high_cer_sample = high_cer_sample.fillna(0.0)


# ====================== 7. 新增：自动分析每个簇的核心特征 ======================
def analyze_cluster_characteristics(cluster_data, cluster_id):
    """分析单个簇的核心特征（高CER原因）"""
    # 基础指标
    avg_cer = cluster_data["cer"].mean()
    avg_duration = cluster_data["duration"].mean()
    avg_speed = cluster_data["speed"].mean()
    avg_snr = cluster_data["snr"].mean()
    cluster_size = len(cluster_data)

    # 特征判断逻辑（基于行业经验阈值）
    characteristics = []

    # 1. 信噪比维度（噪音判断）
    if avg_snr < 30:
        characteristics.append("低信噪比（<30dB），音频背景噪音大")
    elif avg_snr > 40:
        characteristics.append("高信噪比（>40dB），音频音质好")
    else:
        characteristics.append("信噪比适中（30-40dB），音质正常")

    # 2. 语速维度
    if avg_speed > 4.0:
        characteristics.append("语速快（>4字/秒），易导致识别漏字/错字")
    elif avg_speed < 2.5:
        characteristics.append("语速慢（<2.5字/秒），易导致重复识别")
    else:
        characteristics.append("语速适中（2.5-4字/秒），语速不是核心问题")

    # 3. 时长维度
    if avg_duration > 4.0:
        characteristics.append("音频偏长（>4秒），长文本识别误差累积")
    elif avg_duration < 3.0:
        characteristics.append("音频偏短（<3秒），上下文不足")
    else:
        characteristics.append("音频时长适中（3-4秒），时长不是核心问题")

    # 4. CER严重程度
    if avg_cer > 0.25:
        characteristics.append("CER偏高（>0.25），识别错误率高")
    elif avg_cer < 0.20:
        characteristics.append("CER偏低（<0.20），识别效果较好")
    else:
        characteristics.append("CER适中（0.20-0.25），识别效果一般")

    # 5. 极端情况（CER=1.0）
    cer_1_count = len(cluster_data[cluster_data["cer"] == 1.0])
    if cer_1_count / cluster_size > 0.5:
        characteristics.append("超50%样本CER=1.0，识别完全错误（如仅识别出'我'）")

    # 6. 文本错误类型（从示例中提取）
    sample_preds = cluster_data["pred_text_cleaned"].head(5).tolist()
    sample_refs = cluster_data["ref_text_cleaned"].head(5).tolist()
    error_types = []
    for pred, ref in zip(sample_preds, sample_refs):
        pred = str(pred).strip()
        ref = str(ref).strip()
        if len(pred) == 0 or pred == "我":
            error_types.append("完全识别错误（空文本/仅'我'）")
        elif any(char in pred for char in ["的", "地", "得"] if char in ref):
            error_types.append("同音字/近音字错误（如'跟讲明'→'跟江民'）")
        elif len(pred) != len(ref):
            error_types.append("字符增删错误（漏字/多字）")

    # 去重并取最常见的错误类型
    if error_types:
        most_common_error = max(set(error_types), key=error_types.count)
        characteristics.append(f"主要错误类型：{most_common_error}")

    return "; ".join(characteristics)


# ====================== 8. 全样本聚类分析（带特征总结） ======================
print("\n=== 高CER全样本聚类簇特征分析（含核心原因） ===")
cluster_summary = []
for cluster in range(best_k):
    cluster_data = high_cer_sample[high_cer_sample["cluster"] == cluster]
    cluster_size = len(cluster_data)
    cluster_ratio = cluster_size / len(high_cer_sample) * 100
    avg_cer = cluster_data["cer"].mean()
    avg_duration = cluster_data["duration"].mean()
    avg_speed = cluster_data["speed"].mean()
    avg_snr = cluster_data["snr"].mean()

    # 核心：分析簇特征（高CER原因）
    cluster_char = analyze_cluster_characteristics(cluster_data, cluster)

    # 保存汇总
    cluster_summary.append({
        "簇编号": cluster,
        "样本数": cluster_size,
        "占比(%)": round(cluster_ratio, 2),
        "平均CER": round(avg_cer, 4),
        "平均时长(s)": round(avg_duration, 2),
        "平均语速(字/秒)": round(avg_speed, 2),
        "平均信噪比(dB)": round(avg_snr, 2),
        "核心特征（高CER原因）": cluster_char
    })

    # 打印详细信息
    print(f"\n🔹 簇{cluster}（{cluster_size}条，占比{cluster_ratio:.2f}%）")
    print(
        f"   基础指标：平均CER={avg_cer:.4f} | 平均时长={avg_duration:.2f}s | 平均语速={avg_speed:.2f}字/秒 | 平均信噪比={avg_snr:.2f}dB")
    print(f"   核心特征（高CER原因）：{cluster_char}")
    # 示例
    print("   典型示例：")
    for i, (_, row) in enumerate(cluster_data.head(3).iterrows()):
        ref_text = str(row["ref_text_cleaned"])[:30]
        pred_text = str(row["pred_text_cleaned"])[:30]
        print(f"     示例{i + 1}：标注={ref_text} → 识别={pred_text}（CER={row['cer']:.4f}）")

# 打印汇总表（含核心特征）
print("\n=== 聚类簇汇总表（核心特征版） ===")
summary_df = pd.DataFrame(cluster_summary)
# 调整列顺序，让核心特征更显眼
summary_df = summary_df[
    ["簇编号", "样本数", "占比(%)", "平均CER", "核心特征（高CER原因）", "平均时长(s)", "平均语速(字/秒)",
     "平均信噪比(dB)"]]
print(summary_df.to_string(index=False, max_colwidth=80))

# ====================== 9. 导出结果（含核心特征） ======================
# 新增：给原始数据添加核心特征列
high_cer_sample["cluster_characteristics"] = high_cer_sample["cluster"].apply(
    lambda x: cluster_summary[x]["核心特征（高CER原因）"]
)

# 导出全量结果
export_cols = ["full_audio_path", "ref_text_cleaned", "pred_text_cleaned", "cer",
               "cluster", "cluster_characteristics", "duration", "speed", "snr"]
high_cer_sample[export_cols].to_csv("sensevoice_high_cer_full_sample_clusters.csv",
                                    index=False, encoding="utf-8-sig")

# 按簇导出
for cluster in range(best_k):
    cluster_data = high_cer_sample[high_cer_sample["cluster"] == cluster]
    cluster_data[export_cols].to_csv(f"cluster_{cluster}_full_sample.csv",
                                     index=False, encoding="utf-8-sig")
    print(f"\n✅ 簇{cluster}结果已导出：cluster_{cluster}_full_sample.csv（{len(cluster_data)}条）")

print("\n🎉 高CER全样本分析完成！")
print(f"📊 核心结果：")
print(f"   - 分析样本数：{len(high_cer_sample)} 条")
print(f"   - 最优聚类数：{best_k} 个")
print(f"   - 全样本结果文件：sensevoice_high_cer_full_sample_clusters.csv（含每个样本的簇特征）")
print(f"   - K值选择图：best_k_full_sample.png")