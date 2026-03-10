import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from jiwer import cer, wer
import torch
from funasr import AutoModel

# ====================== 1. 配置项（无需修改，适配你的环境） ======================
# Paraformer模型名称
MODEL_NAME = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
# 设备配置（自动检测GPU）
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# 文本清洗规则（和SenseVoice保持一致，确保指标可比）
CLEAN_RULES = {
    "special_tags": ["<|zh|>", "<|NEUTRAL|>", "<|Speech|>", "<|woitn|>"],
    "punctuations": "，。！？；：""''（）[]{}、《》·~@#￥%……&*——+=-_",
    "whitespace": [" ", "　", "\n", "\t", "\r"]
}

# 新增：簇分组配置（核心修改点）
CLUSTER_GROUPS = {
    "cluster1": ["cluster_0_full_sample.csv", "cluster_1_full_sample.csv", "cluster_2_full_sample.csv",
                 "cluster_3_full_sample.csv", "cluster_4_full_sample.csv", "cluster_5_full_sample.csv",
                 "cluster_7_full_sample.csv", "cluster_8_full_sample.csv"],  # 低音质高错误簇
    "cluster2": ["cluster_6_full_sample.csv"]  # 高音质中等错误簇
}


# ====================== 2. 核心工具函数 ======================
def clean_text(text):
    """
    文本清洗（和SenseVoice评测脚本保持一致）
    :param text: 原始文本
    :return: 清洗后的文本
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # 1. 移除SenseVoice特殊标签
    for tag in CLEAN_RULES["special_tags"]:
        text = text.replace(tag, "")

    # 2. 移除标点符号
    for punc in CLEAN_RULES["punctuations"]:
        text = text.replace(punc, "")

    # 3. 移除空白字符
    for ws in CLEAN_RULES["whitespace"]:
        text = text.replace(ws, "")

    # 4. 转为小写（可选，保持一致性）
    text = text.lower()

    return text.strip()


def load_merged_cluster_data(csv_paths, group_name):
    """
    加载并合并多个聚类簇的CSV数据
    :param csv_paths: 多个簇的CSV文件路径列表
    :param group_name: 分组名称（如cluster1/cluster2）
    :return: 合并后的DataFrame
    """
    all_dfs = []
    total_samples = 0
    valid_samples = 0

    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"⚠️ 聚类文件不存在，跳过：{csv_path}")
            continue

        df = pd.read_csv(csv_path, encoding="utf-8-sig")

        # 必要字段检查
        required_fields = ["full_audio_path", "ref_text_cleaned"]
        missing_fields = [f for f in required_fields if f not in df.columns]
        if missing_fields:
            print(f"⚠️ CSV{csv_path}缺少字段{missing_fields}，跳过")
            continue

        # 过滤空值和无效音频路径
        df = df[df["full_audio_path"].notna() & df["ref_text_cleaned"].notna()]
        df = df[df["full_audio_path"].apply(os.path.exists)]

        total_samples += len(df)
        valid_samples += len(df)
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError(f"{group_name}没有有效数据，请检查CSV路径")

    # 合并所有簇的DataFrame
    merged_df = pd.concat(all_dfs, ignore_index=True)
    print(f"✅ 加载{group_name}：合并{len(csv_paths)}个簇 | 总样本数={total_samples} | 有效样本数={len(merged_df)}")
    return merged_df


def eval_paraformer_on_cluster(cluster_df, model, cluster_name):
    """
    在指定聚类分组上评测Paraformer
    :param cluster_df: 合并后的聚类数据DataFrame
    :param model: Paraformer模型实例
    :param cluster_name: 分组名称（如cluster1/cluster2）
    :return: 评测结果字典
    """
    # 存储结果
    results = {
        "cluster_name": cluster_name,
        "total_samples": len(cluster_df),
        "valid_samples": 0,
        "cer_list": [],
        "wer_list": [],
        "all_samples_results": []
    }

    # 逐样本评测
    pbar = tqdm(cluster_df.iterrows(), total=len(cluster_df), desc=f"评测{cluster_name}")
    for idx, row in pbar:
        audio_path = row["full_audio_path"]
        ref_text = row["ref_text_cleaned"]

        try:
            # 1. 模型推理
            res = model.generate(input=audio_path, batch_size=1)
            pred_text = res[0]["text"] if res and len(res) > 0 else ""

            # 2. 文本清洗（和标注保持一致）
            ref_text_clean = clean_text(ref_text)
            pred_text_clean = clean_text(pred_text)

            # 3. 跳过空标注/空预测
            if ref_text_clean == "" and pred_text_clean == "":
                continue
            if ref_text_clean == "":
                continue

            # 4. 计算CER/WER
            current_cer = cer(ref_text_clean, pred_text_clean)
            current_wer = wer(ref_text_clean, pred_text_clean)

            # 5. 存储结果
            results["cer_list"].append(current_cer)
            results["wer_list"].append(current_wer)
            results["valid_samples"] += 1

            # 详细样本结果（保留原簇信息）
            results["all_samples_results"].append({
                "audio_filepath": audio_path,
                "speaker_id": row.get("speaker_id", ""),
                "original_cluster": row.get("cluster", ""),  # 保留原始簇编号
                "ref_text_original": ref_text,
                "pred_text_original": pred_text,
                "ref_text_cleaned": ref_text_clean,
                "pred_text_cleaned": pred_text_clean,
                "cer": current_cer,
                "wer": current_wer,
                "acc": 1 - current_cer
            })

            # 更新进度条
            if results["valid_samples"] > 0:
                pbar.set_postfix({"平均CER": f"{np.mean(results['cer_list']):.4f}"})

        except Exception as e:
            print(f"\n⚠️ 处理{audio_path}失败：{str(e)[:80]}")
            continue

    # 计算整体指标
    if results["valid_samples"] > 0:
        results["overall_metrics"] = {
            "average": {
                "cer": np.mean(results["cer_list"]),
                "wer": np.mean(results["wer_list"]),
                "acc": 1 - np.mean(results["cer_list"])
            },
            "median": {
                "cer": np.median(results["cer_list"]),
                "wer": np.median(results["wer_list"]),
                "acc": 1 - np.median(results["cer_list"])
            }
        }
    else:
        results["overall_metrics"] = {
            "average": {"cer": 1.0, "wer": 1.0, "acc": 0.0},
            "median": {"cer": 1.0, "wer": 1.0, "acc": 0.0}
        }

    return results


def save_eval_report(results, output_dir="."):
    """
    保存评测报告（JSON+CSV）
    :param results: 评测结果字典
    :param output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 1. 保存JSON报告（完整结果）
    json_path = os.path.join(output_dir, f"paraformer_{results['cluster_name']}_eval_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # 2. 保存CSV报告（样本级结果）
    csv_path = os.path.join(output_dir, f"paraformer_{results['cluster_name']}_eval_samples.csv")
    if results["all_samples_results"]:
        df = pd.DataFrame(results["all_samples_results"])
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ 评测报告已保存：")
    print(f"   - JSON完整报告：{json_path}")
    print(f"   - CSV样本结果：{csv_path}")


def main():
    """
    主函数：评测两个聚类分组
    """
    # 1. 加载Paraformer模型（自动下载权重）
    print(f"\n📥 加载Paraformer模型：{MODEL_NAME}")
    print(f"   设备：{DEVICE}")
    model = AutoModel(
        model=MODEL_NAME,
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        device=DEVICE
    )
    print("✅ 模型加载完成")

    # 2. 加载聚类分组数据（核心修改：合并多个簇）
    cluster1_df = load_merged_cluster_data(CLUSTER_GROUPS["cluster1"], "cluster1（低音质高错误簇）")
    cluster2_df = load_merged_cluster_data(CLUSTER_GROUPS["cluster2"], "cluster2（高音质中等错误簇）")

    # 3. 评测cluster1
    print(f"\n========== 评测cluster1（低音质高错误簇：0/1/2/3/4/5/7/8）==========")
    cluster1_results = eval_paraformer_on_cluster(cluster1_df, model, "cluster1")
    save_eval_report(cluster1_results)

    # 4. 评测cluster2
    print(f"\n========== 评测cluster2（高音质中等错误簇：6）==========")
    cluster2_results = eval_paraformer_on_cluster(cluster2_df, model, "cluster2")
    save_eval_report(cluster2_results)

    # 5. 输出对比总结
    print(f"\n========== 评测总结（Paraformer）==========")
    print(f"📊 cluster1（低音质高错误簇）：")
    print(f"   有效样本数：{cluster1_results['valid_samples']}")
    print(f"   平均CER：{cluster1_results['overall_metrics']['average']['cer']:.4f}")
    print(f"   中位数CER：{cluster1_results['overall_metrics']['median']['cer']:.4f}")
    print(f"   平均准确率：{cluster1_results['overall_metrics']['average']['acc']:.4f}")

    print(f"\n📊 cluster2（高音质中等错误簇）：")
    print(f"   有效样本数：{cluster2_results['valid_samples']}")
    print(f"   平均CER：{cluster2_results['overall_metrics']['average']['cer']:.4f}")
    print(f"   中位数CER：{cluster2_results['overall_metrics']['median']['cer']:.4f}")
    print(f"   平均准确率：{cluster2_results['overall_metrics']['average']['acc']:.4f}")

    # 可选：对比SenseVoice参考值（你可以根据实际情况修改）
    print(f"\n========== Paraformer vs SenseVoice 对比 ==========")
    print(
        f"cluster1 - Paraformer平均CER：{cluster1_results['overall_metrics']['average']['cer']:.4f} | SenseVoice参考CER：0.31-0.37")
    print(
        f"cluster2 - Paraformer平均CER：{cluster2_results['overall_metrics']['average']['cer']:.4f} | SenseVoice参考CER：0.3787")


# ====================== 3. 脚本运行入口 ======================
if __name__ == "__main__":
    # 【关键修改】替换为你的CSV文件实际路径（必填）
    # 方式1：CSV文件在脚本同目录，直接写文件名
    # CLUSTER_GROUPS["cluster1"] = ["cluster_0.csv", "cluster_1.csv", "cluster_2.csv", 
    #                              "cluster_3.csv", "cluster_4.csv", "cluster_5.csv", 
    #                              "cluster_7.csv", "cluster_8.csv"]
    # CLUSTER_GROUPS["cluster2"] = ["cluster_6.csv"]

    # 方式2：使用绝对路径（推荐，避免路径问题）
    # 示例：
    # CLUSTER_GROUPS["cluster1"] = [
    #     "G:/ruc/MY_ASR/02-metric测试/3-sensevoice/聚类分析/MFCC聚类分析/cluster_0.csv",
    #     "G:/ruc/MY_ASR/02-metric测试/3-sensevoice/聚类分析/MFCC聚类分析/cluster_1.csv",
    #     "G:/ruc/MY_ASR/02-metric测试/3-sensevoice/聚类分析/MFCC聚类分析/cluster_2.csv",
    #     "G:/ruc/MY_ASR/02-metric测试/3-sensevoice/聚类分析/MFCC聚类分析/cluster_3.csv",
    #     "G:/ruc/MY_ASR/02-metric测试/3-sensevoice/聚类分析/MFCC聚类分析/cluster_4.csv",
    #     "G:/ruc/MY_ASR/02-metric测试/3-sensevoice/聚类分析/MFCC聚类分析/cluster_5.csv",
    #     "G:/ruc/MY_ASR/02-metric测试/3-sensevoice/聚类分析/MFCC聚类分析/cluster_7.csv",
    #     "G:/ruc/MY_ASR/02-metric测试/3-sensevoice/聚类分析/MFCC聚类分析/cluster_8.csv"
    # ]
    # CLUSTER_GROUPS["cluster2"] = [
    #     "G:/ruc/MY_ASR/02-metric测试/3-sensevoice/聚类分析/MFCC聚类分析/cluster_6.csv"
    # ]

    # 运行主函数
    main()