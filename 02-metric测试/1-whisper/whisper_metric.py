import json
import os
import jiwer
import re
import numpy as np
from whisper import load_model
from tqdm import tqdm
from typing import Dict, List

# ========== 配置项（重点：新增样本数量限制） ==========
JSONL_PATH = "../sichuan_asr_manifest_dialect.jsonl"  # 你的评测JSONL文件路径
AUDIO_ROOT = "./sichuan/wav"  # 音频根目录
DEVICE = "cuda:0"
WHISPER_MODEL_SIZE = "large-v3"
MAX_SAMPLES = 1000  # 仅评测前1000条有效样本


# ========== 核心工具函数（无修改） ==========
def compute_metrics(pred_text: str, ref_text: str) -> Dict:
  """修复后的Metric计算函数"""
  pred = pred_text.strip().replace(" ", "").lower()
  ref = ref_text.strip().replace(" ", "").lower()

  if not ref:
    return {
      "cer": 1.0,
      "wer": 1.0,
      "acc": 0.0,
      "pred_text": pred_text,
      "ref_text": ref_text
    }
  if not pred:
    return {
      "cer": 1.0,
      "wer": 1.0,
      "acc": 0.0,
      "pred_text": pred_text,
      "ref_text": ref_text
    }

  # CER计算
  cer = jiwer.cer(ref, pred)
  acc = 1 - cer

  # WER计算（空格分隔单字）
  ref_words = " ".join(list(ref))
  pred_words = " ".join(list(pred))
  wer = jiwer.wer(ref_words, pred_words)

  return {
    "cer": round(cer, 4),
    "wer": round(wer, 4),
    "acc": round(acc, 4),
    "pred_text": pred_text,
    "ref_text": ref_text
  }


def load_eval_data(jsonl_path: str) -> List[Dict]:
  """加载JSONL评测数据，拼接完整音频路径"""
  eval_data = []
  with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      data = json.loads(line)
      data["audio_filepath"] = os.path.abspath(
          os.path.join(os.path.dirname(jsonl_path), data["audio_filepath"])
      )
      eval_data.append(data)
  return eval_data


def whisper_inference(model, audio_path: str) -> str:
    try:
        transcribe_kwargs = {
            "audio": audio_path,
            # "language": "zh",
            "language": "Chinese",
            "fp16": True,
            "verbose": False,
            # 核心修复：降低beam_size，避免base模型过载
            "beam_size": 5,
            "best_of": 5,
            "temperature": 0.0,  # 温度设为0，强制确定性生成，杜绝随机重复
            "suppress_tokens":"-1"
        }

        # 步骤2：兼容initial_prompt（部分低版本Whisper叫"prompt"）
        try:
            # 优先用initial_prompt（新版Whisper）
            transcribe_kwargs["initial_prompt"] = "四川方言，口语化表达，包含：啥子、啷个、要得、巴适、莫得、耙耳朵、摆龙门阵"
        except:
            # 低版本Whisper用prompt
            transcribe_kwargs["prompt"] = "四川方言，口语化表达，包含：啥子、啷个、要得、巴适、莫得、耙耳朵、摆龙门阵"

        # 执行转录
        result = model.transcribe(**transcribe_kwargs)
        pred_text = result["text"].strip()

        # 步骤3：后处理（清洗+方言纠错，和标注格式对齐）
        # 1. 去除标点（和标注文本清洗规则一致）
        pred_text = re.sub(r'[,，.。!！?？;；:："\"\'\'‘’“”、()（）\[\]【】《》～·]', "", pred_text)
        # 2. 去除多余空格（Whisper默认加空格）
        pred_text = re.sub(r"\s+", "", pred_text)
        # 3. 方言词汇纠错（解决核心谐音/用词错误）
        dialect_correction = {
            "什么": "啥子",
            "怎么": "啷个",
            "没有": "莫得",
            "舒服": "巴适",
            "好的": "要得",
            "是绵": "石棉",
            "和母金": "护目镜",
            "要拥抱": "有用不",
            "海鲜": "海神"
        }
        for wrong, correct in dialect_correction.items():
            pred_text = pred_text.replace(wrong, correct)

        return pred_text

    except Exception as e:
        print(f"\n❌ 音频{os.path.basename(audio_path)}识别失败：{str(e)[:50]}...")
        return ""


# ========== 主评测流程（核心修改：限制样本数量） ==========
def main():
  # 1. 加载评测数据
  print(f"📥 加载评测数据：{JSONL_PATH}")
  eval_data = load_eval_data(JSONL_PATH)
  valid_samples = [d for d in eval_data if os.path.exists(d["audio_filepath"])]
  invalid_samples = [d for d in eval_data if not os.path.exists(d["audio_filepath"])]

  print(f"✅ 共加载 {len(eval_data)} 条样本")
  print(f"✅ 有效音频样本：{len(valid_samples)} 条")
  if invalid_samples:
    print(f"⚠️ 无效音频样本（路径不存在）：{len(invalid_samples)} 条，已跳过")

  # ========== 核心修改：截取前1000条有效样本 ==========
  if len(valid_samples) > MAX_SAMPLES:
    valid_samples = valid_samples[:MAX_SAMPLES]
    print(f"\n🔍 已截取前 {MAX_SAMPLES} 条有效样本进行评测（缩短耗时）")
  else:
    print(f"\n🔍 有效样本数不足{MAX_SAMPLES}条，将评测全部{len(valid_samples)}条")

  # 2. 加载Whisper模型
  print(f"\n🔄 加载Whisper-{WHISPER_MODEL_SIZE}模型（{DEVICE}）...")
  model = load_model(WHISPER_MODEL_SIZE, device=DEVICE)
  print(f"✅ Whisper模型加载完成！")

  # 3. 逐条推理并计算Metric
  all_results = []
  cer_list = []
  wer_list = []
  acc_list = []

  # 进度条显示：基于截取后的样本数量
  pbar = tqdm(valid_samples, desc="Whisper评测中")
  for sample in pbar:
    try:
      audio_path = sample["audio_filepath"]
      ref_text = sample["text"]

      # 模型推理
      pred_text = whisper_inference(model, audio_path)

      # 计算Metric
      metrics = compute_metrics(pred_text, ref_text)

      # 保存结果
      result = {
        "audio_filepath": audio_path,
        "speaker_id": sample["speaker_id"],
        "ref_text": ref_text,
        "pred_text": pred_text,
        "cer": metrics["cer"],
        "wer": metrics["wer"],
        "acc": metrics["acc"]
      }
      all_results.append(result)

      # 累计Metric
      cer_list.append(metrics["cer"])
      wer_list.append(metrics["wer"])
      acc_list.append(metrics["acc"])

      # 更新进度条
      pbar.set_postfix({
        "平均CER": f"{np.mean(cer_list):.4f}",
        "平均WER": f"{np.mean(wer_list):.4f}",
        "平均ACC": f"{np.mean(acc_list):.4f}"
      })
    except Exception as e:
      print(f"\n⚠️ 样本{os.path.basename(sample['audio_filepath'])}处理失败：{str(e)[:50]}...")
      continue

  # 4. 生成评测报告（无修改，适配截取后的样本）
  if not all_results:
    print("\n❌ 无有效评测结果！")
    return

  avg_cer = round(np.mean(cer_list), 4)
  avg_wer = round(np.mean(wer_list), 4)
  avg_acc = round(np.mean(acc_list), 4)
  med_cer = round(np.median(cer_list), 4)
  med_wer = round(np.median(wer_list), 4)
  med_acc = round(np.median(acc_list), 4)

  worst_cases = sorted(all_results, key=lambda x: x["cer"], reverse=True)[:5]

  # 保存JSON报告
  report = {
    "model": f"whisper-{WHISPER_MODEL_SIZE}",
    "device": DEVICE,
    "actual_sample_count": len(all_results),
    "overall_metrics": {
      "average": {"cer": avg_cer, "wer": avg_wer, "acc": avg_acc},
      "median": {"cer": med_cer, "wer": med_wer, "acc": med_acc}
    },
    "worst_cases": worst_cases,
    "all_samples": all_results
  }

  json_report_path = "whisper_sichuan_eval_report_1000samples.json"
  with open(json_report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=4)

  # 保存文本报告
  txt_report_path = "whisper_sichuan_eval_report_1000samples.txt"
  with open(txt_report_path, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write(f"Whisper-{WHISPER_MODEL_SIZE} 四川方言评测报告\n")
    f.write("=" * 60 + "\n")
    f.write(f"评测样本数：{len(all_results)} 条\n")
    f.write(f"运行设备：{DEVICE}\n")
    f.write("\n【核心指标】\n")
    f.write(f"平均CER（字符错误率）：{avg_cer}\n")
    f.write(f"平均WER（词错误率）：{avg_wer}\n")
    f.write(f"平均ACC（字符准确率）：{avg_acc}\n")
    f.write(f"中位数CER：{med_cer}\n")
    f.write("\n【Top5坏例（CER最高）】\n")
    for i, case in enumerate(worst_cases, 1):
      f.write(f"\n{i}. 音频：{os.path.basename(case['audio_filepath'])}\n")
      f.write(f"   标注文本：{case['ref_text']}\n")
      f.write(f"   识别文本：{case['pred_text']}\n")
      f.write(f"   CER：{case['cer']}\n")

  # 5. 打印最终结果
  print("\n" + "=" * 60)
  print(f"✅ Whisper-{WHISPER_MODEL_SIZE} 评测完成！")
  print(f"📊 核心指标：")
  print(f"   平均CER：{avg_cer}（越小越好）")
  print(f"   平均WER：{avg_wer}（越小越好）")
  print(f"   平均ACC：{avg_acc}（越大越好）")
  print(f"\n📄 评测报告已保存：")
  print(f"   JSON详情：{json_report_path}")
  print(f"   文本摘要：{txt_report_path}")
  print("=" * 60)


if __name__ == "__main__":
  main()