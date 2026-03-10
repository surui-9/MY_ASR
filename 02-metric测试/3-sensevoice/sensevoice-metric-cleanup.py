import json
import os
import re
import jiwer
import numpy as np
from funasr import AutoModel
from tqdm import tqdm
from typing import Dict, List

model_dir = "iic/SenseVoiceSmall"

# ========== 配置项 ==========
JSONL_PATH = "../sichuan_asr_manifest_dialect.jsonl"  # 清洗后的标注文件路径（若未替换，需改为清洗后文件路径）
AUDIO_ROOT = "../sichuan/wav"  # 音频根目录
DEVICE = "cuda:0"  # 保持你的GPU配置


# 关键：取消MAX_SAMPLES限制，评测全部有效样本
# MAX_SAMPLES = 1000

# ========== 核心清洗函数（重点处理SenseVoice标签） ==========
def clean_text(text: str) -> str:
    """
    彻底清洗文本：去除SenseVoice特殊标签+标点+空格，适配四川方言场景
    - 去除：<<|zh|>、<<|NEUTRAL|>等模型标签，中英文标点，空格/不可见字符
    - 保留：中文汉字、方言原生写法（gai/切等）、数字
    """
    if not text:
        return ""

    # 1. 优先去除SenseVoice的特殊标签（核心！解决标签干扰问题）
    # 匹配<<|xxx|>格式的所有标签，彻底删除
    text = re.sub(r"<\|.*?\|>", "", text)

    # 2. 去除中英文标点（避免格式干扰CER计算）
    punctuation_pattern = r'[,，.。!！?？;；:："\"\'\'‘’“”、()（）\[\]【】《》～·@#￥%^&*-=<>_/]'
    text = re.sub(punctuation_pattern, "", text)

    # 3. 去除空格、全角空格、换行、制表符
    text = re.sub(r"[\s\n\t\u3000]", "", text)

    # 4. 去除不可见字符（如控制字符），避免隐性干扰
    text = re.sub(r"[\x00-\x1F\x7F]", "", text)

    # 5. 收尾：去除首尾空字符（兜底）
    return text.strip()


# ========== 核心Metric计算函数（基于清洗后文本） ==========
def compute_metrics(pred_text: str, ref_text: str) -> Dict:
    """统一计算逻辑：先清洗，再计算CER/WER，确保结果准确"""
    # 关键：预测文本（识别结果）和参考文本（标注）都用同一函数清洗
    pred_clean = clean_text(pred_text)
    ref_clean = clean_text(ref_text)

    # 极端情况处理
    if not ref_clean:
        return {
            "cer": 1.0,
            "wer": 1.0,
            "acc": 0.0,
            "pred_text_original": pred_text,
            "ref_text_original": ref_text,
            "pred_text_cleaned": pred_clean,
            "ref_text_cleaned": ref_clean
        }
    if not pred_clean:
        return {
            "cer": 1.0,
            "wer": 1.0,
            "acc": 0.0,
            "pred_text_original": pred_text,
            "ref_text_original": ref_text,
            "pred_text_cleaned": pred_clean,
            "ref_text_cleaned": ref_clean
        }

    # 计算CER（字符错误率）- 基于纯清洗文本
    cer = jiwer.cer(ref_clean, pred_clean)
    acc = 1 - cer  # 字符准确率

    # 计算WER（词错误率，中文按单字分词）
    ref_words = " ".join(list(ref_clean))
    pred_words = " ".join(list(pred_clean))
    wer = jiwer.wer(ref_words, pred_words)

    return {
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "acc": round(acc, 4),
        "pred_text_original": pred_text,  # 保留原始识别结果（便于查错）
        "ref_text_original": ref_text,  # 保留原始标注（便于对比）
        "pred_text_cleaned": pred_clean,  # 清洗后识别文本（用于计算）
        "ref_text_cleaned": ref_clean  # 清洗后标注文本（用于计算）
    }


# ========== 加载全量评测数据 ==========
def load_eval_data(jsonl_path: str) -> List[Dict]:
    """加载JSONL数据，拼接绝对音频路径，保留全部有效样本"""
    eval_data = []
    if not os.path.exists(jsonl_path):
        print(f"❌ 错误：标注文件不存在 → {jsonl_path}")
        return eval_data

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # 拼接绝对音频路径，避免相对路径错误
                data["audio_filepath"] = os.path.abspath(
                    os.path.join(os.path.dirname(jsonl_path), data["audio_filepath"])
                )
                eval_data.append(data)
            except json.JSONDecodeError as e:
                print(f"⚠️ 第{line_num}行JSON格式错误，跳过 → {e}")
                continue

    print(f"✅ 共加载标注样本：{len(eval_data)} 条")
    return eval_data


def sensevoice_inference(model, audio_path: str) -> str:
    """
    SenseVoiceSmall 四川方言短音频（10秒内）最优推理参数
    核心优化：适配方言口语化、短音频低切分、解码偏向方言词汇
    """
    try:
        result = model.generate(
            input=audio_path,
            cache={},
            batch_size=1,  # 短音频单条推理效率最高
            language="zh",  # 强制中文，避免识别成其他语言 "zh-cn"效果不如zh
            detect_language=False,
            use_pun=False,     # 关闭标点（减少清洗压力，匹配标注格式）
            use_itn=False,      # 关闭逆文本规范化
            # ========== VAD优化（10秒短音频专属） ==========
            vad_kwargs={
                "max_single_segment_time": 10000,  # 匹配音频最长10秒，避免过度切分
                "vad_merge_silence": 800,          # 短音频减小静音合并，保留口语停顿
                "vad_onset": 0.3,                  # 降低触发阈值，避免漏检方言轻读音
                "vad_offset": 0.1,
                "vad_pad_ms": 300,                 # 前后补300ms，避免切丢开头/结尾方言词汇
            },
            # ========== 解码策略（方言优先） ==========
            beam_size=6,  # Small模型最优值（5→6，覆盖更多方言候选词，不显著变慢）
            decoding_chunk_size=[-1, -1, -1],  # 全量解码（短音频无需分块，准确率更高）
            alpha=0.8,    # 新增：语言模型权重，降低随机性，偏向常见方言词汇
            beta=0.8,     # 新增：长度惩罚，避免识别结果过短/过长
            # ========== 其他适配参数 ==========
            temperature=0.6,  # 降低温度（0.7→0.6），减少随机错误
            max_new_tokens=200,  # 匹配10秒音频最大字符数（四川方言语速≈20字/秒）
            param_dict={"dialect": "sichuan"},
        )
        return result[0]["text"].strip() if result and len(result) > 0 else ""
    except Exception as e:
        print(f"\n❌ 音频{os.path.basename(audio_path)}识别失败：{str(e)[:50]}...")
        return ""


# ========== 主评测流程（全量样本） ==========
def main():
    # 1. 加载全量评测数据
    print(f"📥 加载全量评测数据：{JSONL_PATH}")
    eval_data = load_eval_data(JSONL_PATH)
    if not eval_data:
        print("❌ 无有效标注样本，终止评测！")
        return

    # 筛选有效音频样本（仅跳过路径不存在的音频，不限制数量）
    valid_samples = [d for d in eval_data if os.path.exists(d["audio_filepath"])]
    invalid_samples = [d for d in eval_data if not os.path.exists(d["audio_filepath"])]

    print(f"✅ 有效音频样本：{len(valid_samples)} 条（全部评测）")
    if invalid_samples:
        print(f"⚠️ 无效音频样本（路径不存在）：{len(invalid_samples)} 条，已跳过")

    # 2. 加载SenseVoice模型（保持你的配置）
    print(f"\n🔄 加载FunASR-SenseVoice模型（{DEVICE}）...")
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        device=DEVICE,
        language="zh",
        disable_update=True
    )
    print(f"✅ {model_dir}模型加载完成！")

    # 3. 逐条推理并计算Metric（全量样本，无数量限制）
    all_results = []
    cer_list = []
    wer_list = []
    acc_list = []

    # 进度条：显示全量样本评测进度
    pbar = tqdm(valid_samples, desc=f"{model_dir}全量样本评测中")
    for sample in pbar:
        try:
            audio_path = sample["audio_filepath"]
            ref_text = sample["text"]  # 原始标注文本（后续会清洗）

            # 模型推理（获取带标签的原始识别结果）
            pred_text = sensevoice_inference(model, audio_path)

            # 计算Metric（自动清洗识别文本和标注文本）
            metrics = compute_metrics(pred_text, ref_text)

            # 保存完整结果（含原始+清洗后文本，便于分析）
            result = {
                "audio_filepath": audio_path,
                "speaker_id": sample.get("speaker_id", ""),
                "ref_text_original": ref_text,
                "pred_text_original": pred_text,
                "ref_text_cleaned": metrics["ref_text_cleaned"],
                "pred_text_cleaned": metrics["pred_text_cleaned"],
                "cer": metrics["cer"],
                "wer": metrics["wer"],
                "acc": metrics["acc"]
            }
            all_results.append(result)

            # 累计指标用于统计
            cer_list.append(metrics["cer"])
            wer_list.append(metrics["wer"])
            acc_list.append(metrics["acc"])

            # 进度条实时显示平均指标
            pbar.set_postfix({
                "平均CER": f"{np.mean(cer_list):.4f}",
                "平均WER": f"{np.mean(wer_list):.4f}",
                "平均ACC": f"{np.mean(acc_list):.4f}"
            })
        except Exception as e:
            # 单样本出错跳过，不中断全量评测
            print(f"\n⚠️ 样本{os.path.basename(sample['audio_filepath'])}处理失败：{str(e)[:50]}...")
            continue

    # 4. 生成全量样本评测报告（含清洗说明）
    if not all_results:
        print("\n❌ 无有效评测结果！")
        return

    # 计算整体指标（全量样本）
    avg_cer = round(np.mean(cer_list), 4)
    avg_wer = round(np.mean(wer_list), 4)
    avg_acc = round(np.mean(acc_list), 4)
    med_cer = round(np.median(cer_list), 4)
    med_wer = round(np.median(wer_list), 4)
    med_acc = round(np.median(acc_list), 4)

    # 找出Top10坏例（CER最高，便于后续分析）
    worst_cases = sorted(all_results, key=lambda x: x["cer"], reverse=True)[:10]

    # 保存JSON详细报告（含全量样本结果）
    json_report_path = "测评结果9-保留音译字.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        report = {
            "model": model_dir,
            "device": DEVICE,
            "eval_scope": "全量有效样本",
            "total_labeled_samples": len(eval_data),
            "total_valid_samples": len(valid_samples),
            "total_invalid_samples": len(invalid_samples),
            "text_cleaning_rules": [
                "1. 去除SenseVoice特殊标签（<<|zh|>、<<|NEUTRAL|>等）",
                "2. 去除中英文标点（，。！？等）",
                "3. 去除空格、全角空格、换行、不可见字符",
                "4. 保留中文、方言原生写法（gai/切等）、数字"
            ],
            "overall_metrics": {
                "average": {"cer": avg_cer, "wer": avg_wer, "acc": avg_acc},
                "median": {"cer": med_cer, "wer": med_wer, "acc": med_acc}
            },
            "top10_worst_cases": worst_cases,
            "all_samples_results": all_results
        }
        json.dump(report, f, ensure_ascii=False, indent=4)

    # 保存简易文本报告（便于快速查看）
    txt_report_path = "测评结果9-保留音译字.txt"
    with open(txt_report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"{model_dir} 四川方言评测报告（全量有效样本）\n")
        f.write("=" * 80 + "\n")
        f.write(f"标注样本总数：{len(eval_data)} 条\n")
        f.write(f"有效音频样本：{len(valid_samples)} 条\n")
        f.write(f"无效音频样本：{len(invalid_samples)} 条\n")
        f.write(f"运行设备：{DEVICE}\n")
        f.write("\n【文本清洗规则】\n")
        f.write("1. 彻底去除SenseVoice特殊标签（<<|zh|>、<<|NEUTRAL|>等）\n")
        f.write("2. 去除所有中英文标点，避免格式干扰\n")
        f.write("3. 去除空格、换行、不可见字符\n")
        f.write("4. 保留方言原生写法（如gai、切）和数字\n")
        f.write("\n【核心指标（基于清洗后文本）】\n")
        f.write(f"平均CER（字符错误率）：{avg_cer}\n")
        f.write(f"平均WER（词错误率）：{avg_wer}\n")
        f.write(f"平均ACC（字符准确率）：{avg_acc}\n")
        f.write(f"中位数CER：{med_cer}\n")
        f.write("\n【Top10坏例（CER最高）】\n")
        for i, case in enumerate(worst_cases, 1):
            f.write(f"\n{i}. 音频：{os.path.basename(case['audio_filepath'])}\n")
            f.write(f"   原始标注：{case['ref_text_original']}\n")
            f.write(f"   清洗后标注：{case['ref_text_cleaned']}\n")
            f.write(f"   原始识别（带标签）：{case['pred_text_original']}\n")
            f.write(f"   清洗后识别：{case['pred_text_cleaned']}\n")
            f.write(f"   CER：{case['cer']}\n")

    # 5. 打印最终结果
    print("\n" + "=" * 80)
    print(f"✅ {model_dir} 全量样本评测完成！")
    print(f"📊 评测范围：{len(valid_samples)} 条有效样本（无数量限制）")
    print(f"📊 核心指标（基于清洗后文本）：")
    print(f"   平均CER：{avg_cer}（越小越好）")
    print(f"   平均WER：{avg_wer}（越小越好）")
    print(f"   平均ACC：{avg_acc}（越大越好）")
    print(f"\n📄 评测报告已保存：")
    print(f"   JSON详情（全量结果）：{json_report_path}")
    print(f"   文本摘要（坏例分析）：{txt_report_path}")
    print("=" * 80)


# ========== 依赖安装（若未安装） ==========
# pip install funasr jiwer tqdm numpy torch

if __name__ == "__main__":
    main()