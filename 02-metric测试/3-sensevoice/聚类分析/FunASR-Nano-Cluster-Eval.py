import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from jiwer import cer, wer
import torch
from funasr import AutoModel

# ====================== 1. 核心配置（适配Fun-ASR-Nano-2512） ======================
# Fun-ASR-Nano模型名称
MODEL_NAME = "FunAudioLLM/Fun-ASR-Nano-2512"
REMOTE_CODE_PATH = "model.py"  # 你的自定义model.py路径（无则留空，模型会自动下载）
# 设备配置（自动检测GPU/CPU）
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# 文本清洗规则（和SenseVoice/Paraformer完全一致，保证指标可比）
CLEAN_RULES = {
    "special_tags": ["<|zh|>", "<|NEUTRAL|>", "<|Speech|>", "<|woitn|>"],
    "punctuations": "，。！？；：""''（）[]{}、《》·~@#￥%……&*——+=-_",
    "whitespace": [" ", "　", "\n", "\t", "\r"]
}

# 新增：簇分组配置（适配你的实际文件名：带_full_sample后缀）
CLUSTER_GROUPS = {
    "cluster1": [
        "cluster_0_full_sample.csv",  # 低音质高错误簇：0/1/2/3/4/5/7/8
        "cluster_1_full_sample.csv",
        "cluster_2_full_sample.csv",
        "cluster_3_full_sample.csv",
        "cluster_4_full_sample.csv",
        "cluster_5_full_sample.csv",
        "cluster_7_full_sample.csv",
        "cluster_8_full_sample.csv"
    ],
    "cluster2": ["cluster_6_full_sample.csv"]  # 高音质中等错误簇：6
}


# ====================== 2. 文本清洗工具（和之前保持一致） ======================
def clean_text(text):
    """统一文本清洗规则，确保CER/WER计算公平"""
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # 1. 移除特殊标签
    for tag in CLEAN_RULES["special_tags"]:
        text = text.replace(tag, "")

    # 2. 移除标点符号
    for punc in CLEAN_RULES["punctuations"]:
        text = text.replace(punc, "")

    # 3. 移除空白字符
    for ws in CLEAN_RULES["whitespace"]:
        text = text.replace(ws, "")

    # 4. 小写化（保持一致性）
    text = text.lower()

    return text.strip()


# ====================== 3. 数据加载函数（支持多簇合并） ======================
def load_merged_cluster_data(csv_paths, group_name):
    """加载并合并多个聚类簇的CSV数据，过滤无效样本"""
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

        # 过滤空值和不存在的音频
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


# ====================== 4. 核心评测函数 ======================
def eval_nano_on_cluster(cluster_df, model, cluster_name):
    """在指定聚类分组上评测Fun-ASR-Nano"""
    results = {
        "cluster_name": cluster_name,
        "total_samples": len(cluster_df),
        "valid_samples": 0,
        "cer_list": [],
        "wer_list": [],
        "all_samples_results": []
    }

    # 逐样本推理+评测
    pbar = tqdm(cluster_df.iterrows(), total=len(cluster_df), desc=f"评测{cluster_name}")
    for idx, row in pbar:
        audio_path = row["full_audio_path"]
        ref_text = row["ref_text_cleaned"]

        try:
            # Fun-ASR-Nano推理（适配模型特性）
            res = model.generate(
                input=audio_path,
                batch_size=1,
                # Nano模型专属配置：关闭VAD/PUNC（模型已内置）
                vad_infer_mode="off",
                punc_infer_mode="off"
            )
            pred_text = res[0]["text"] if res and len(res) > 0 else ""

            # 文本清洗
            ref_clean = clean_text(ref_text)
            pred_clean = clean_text(pred_text)

            # 跳过无效样本
            if ref_clean == "" or (ref_clean == "" and pred_clean == ""):
                continue

            # 计算CER/WER
            current_cer = cer(ref_clean, pred_clean)
            current_wer = wer(ref_clean, pred_clean)

            # 存储结果
            results["cer_list"].append(current_cer)
            results["wer_list"].append(current_wer)
            results["valid_samples"] += 1

            # 样本级详细结果（保留原始簇信息）
            results["all_samples_results"].append({
                "audio_filepath": audio_path,
                "speaker_id": row.get("speaker_id", ""),
                "original_cluster": row.get("cluster", ""),  # 保留原始簇编号
                "ref_text_original": ref_text,
                "pred_text_original": pred_text,
                "ref_text_cleaned": ref_clean,
                "pred_text_cleaned": pred_clean,
                "cer": current_cer,
                "wer": current_wer,
                "acc": 1 - current_cer
            })

            # 实时更新进度条
            if results["valid_samples"] > 0:
                pbar.set_postfix({"平均CER": f"{np.mean(results['cer_list']):.4f}"})

        except Exception as e:
            print(f"\n⚠️ 处理{os.path.basename(audio_path)}失败：{str(e)[:80]}")
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


# ====================== 5. 结果保存函数 ======================
def save_eval_report(results, output_dir="."):
    """保存评测报告（JSON+CSV），和Paraformer脚本输出格式一致"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. JSON完整报告
    json_path = os.path.join(output_dir, f"funasr_nano_{results['cluster_name']}_eval_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # 2. CSV样本级结果
    csv_path = os.path.join(output_dir, f"funasr_nano_{results['cluster_name']}_eval_samples.csv")
    if results["all_samples_results"]:
        df = pd.DataFrame(results["all_samples_results"])
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ {results['cluster_name']}评测报告已保存：")
    print(f"   - JSON完整报告：{json_path}")
    print(f"   - CSV样本结果：{csv_path}")


# ====================== 6. 主函数 ======================
def main():
    """主流程：加载模型→加载合并数据→评测→保存结果→跨模型对比"""
    # 1. 加载Fun-ASR-Nano模型
    print(f"\n📥 加载Fun-ASR-Nano模型：{MODEL_NAME}")
    print(f"   运行设备：{DEVICE}")

    # 适配remote_code参数（无自定义model.py则设为None）
    model_kwargs = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "trust_remote_code": True,
        "disable_update": True,
        # Nano模型无需额外VAD/PUNC，关闭以提升速度
        "vad_model": None,
        "punc_model": None
    }
    if REMOTE_CODE_PATH and os.path.exists(REMOTE_CODE_PATH):
        model_kwargs["remote_code"] = REMOTE_CODE_PATH

    model = AutoModel(**model_kwargs)
    print("✅ Fun-ASR-Nano模型加载完成")

    # 2. 加载聚类分组数据（合并多簇）
    cluster1_df = load_merged_cluster_data(CLUSTER_GROUPS["cluster1"], "cluster1（低音质高错误簇）")
    cluster2_df = load_merged_cluster_data(CLUSTER_GROUPS["cluster2"], "cluster2（高音质中等错误簇）")

    # 3. 评测cluster1（0/1/2/3/4/5/7/8簇）
    print(f"\n========== 评测cluster1（低音质高错误簇：0/1/2/3/4/5/7/8）==========")
    cluster1_results = eval_nano_on_cluster(cluster1_df, model, "cluster1")
    save_eval_report(cluster1_results)

    # 4. 评测cluster2（6簇）
    print(f"\n========== 评测cluster2（高音质中等错误簇：6）==========")
    cluster2_results = eval_nano_on_cluster(cluster2_df, model, "cluster2")
    save_eval_report(cluster2_results)

    # 5. 输出Fun-ASR-Nano核心指标
    print(f"\n========== Fun-ASR-Nano 评测总结 ==========")
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

    # 6. 跨模型对比（需替换Paraformer的实际评测值）
    print(f"\n========== 三方模型对比（SenseVoice/Paraformer/Fun-ASR-Nano）==========")
    print(f"📈 cluster1（低音质高错误簇）：")
    print(f"   SenseVoice平均CER：0.31-0.37")
    print(f"   Paraformer平均CER：（请替换为实际值）")
    print(f"   Fun-ASR-Nano平均CER：{cluster1_results['overall_metrics']['average']['cer']:.4f}")

    print(f"\n📈 cluster2（高音质中等错误簇）：")
    print(f"   SenseVoice平均CER：0.3787")
    print(f"   Paraformer平均CER：（请替换为实际值）")
    print(f"   Fun-ASR-Nano平均CER：{cluster2_results['overall_metrics']['average']['cer']:.4f}")


# ====================== 7. 运行入口 ======================
if __name__ == "__main__":
    # 【关键修改】替换为你的CSV文件实际路径（二选一）
    # 方式1：CSV文件在脚本同目录，无需修改（已适配_full_sample后缀）
    # 方式2：使用绝对路径（推荐，避免路径问题）
    # 示例：
    # CLUSTER_GROUPS["cluster1"] = [
    #     "G:/your_path/cluster_0_full_sample.csv",
    #     "G:/your_path/cluster_1_full_sample.csv",
    #     "G:/your_path/cluster_2_full_sample.csv",
    #     "G:/your_path/cluster_3_full_sample.csv",
    #     "G:/your_path/cluster_4_full_sample.csv",
    #     "G:/your_path/cluster_5_full_sample.csv",
    #     "G:/your_path/cluster_7_full_sample.csv",
    #     "G:/your_path/cluster_8_full_sample.csv"
    # ]
    # CLUSTER_GROUPS["cluster2"] = ["G:/your_path/cluster_6_full_sample.csv"]

    # 运行主函数
    main()