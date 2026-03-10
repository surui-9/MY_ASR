import json
import os
import jiwer
import numpy as np
from funasr import AutoModel
from tqdm import tqdm
from typing import Dict, List
import librosa
import soundfile as sf

# ========== 配置项（和之前保持一致，保证对比公平） ==========
JSONL_PATH = "../sichuan_asr_manifest_dialect.jsonl"  # 你的评测JSONL文件路径
AUDIO_ROOT = "../sichuan/wav"  # 音频根目录
DEVICE = "cuda:0"  # 有GPU改为"cuda"（速度提升10倍+）
# MAX_SAMPLES = 1000  # 仅评测前1000条有效样本


# ========== 核心工具函数（和Whisper/SenseVoice完全一致） ==========
def compute_metrics(pred_text: str, ref_text: str) -> Dict:
  """统一的Metric计算逻辑，确保三者评测结果可对比"""
  pred = pred_text.strip().replace(" ", "").lower()
  ref = ref_text.strip().replace(" ", "").lower()

  # 极端情况处理
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

  # 计算CER（字符错误率）
  cer = jiwer.cer(ref, pred)
  acc = 1 - cer  # 字符准确率

  # 计算WER（词错误率，中文按单字分词）
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
      # 拼接绝对音频路径，避免相对路径错误
      data["audio_filepath"] = os.path.abspath(
          os.path.join(os.path.dirname(jsonl_path), data["audio_filepath"])
      )
      eval_data.append(data)
  return eval_data


def resample_audio(input_path, output_path="./temp_16k.wav"):
  """将任意音频转为 Paraformer 要求的 16kHz 单声道 16bit WAV"""
  y, sr = librosa.load(input_path, sr=None)
  # 转单声道
  if len(y.shape) > 1:
    y = librosa.to_mono(y)
  # 重采样到 16kHz（Paraformer强制要求）
  if sr != 16000:
    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
  # 保存为 16bit WAV（避免格式不兼容）
  sf.write(output_path, y, 16000, subtype="PCM_16")
  return output_path


def paraformer_inference(model, audio_path: str) -> str:
  """Paraformer模型推理封装，适配你的专属配置"""
  try:
    # 第一步：统一音频格式（Paraformer对格式要求严格）
    audio_16k = resample_audio(audio_path)

    # 第二步：Paraformer推理（按你的配置）
    result = model.generate(
        input=audio_16k,
        cache={},
        batch_size=1,
        # 保持和你的配置一致的参数
        disable_pbar=True  # 关闭内部进度条，避免干扰
    )

    # 第三步：提取识别结果（去除标点，保证和标注文本格式一致）
    pred_text = result[0]["text"].strip()
    # 移除标点符号（标注文本无标点，避免Metric计算干扰）
    punc_list = ['，', '。', '！', '？', '；', '：', '、', '“', '”', '（', '）']
    for punc in punc_list:
      pred_text = pred_text.replace(punc, "")

    # 删除临时文件
    os.remove(audio_16k)

    return pred_text
  except Exception as e:
    print(f"\n❌ 音频{os.path.basename(audio_path)}识别失败：{str(e)[:50]}...")
    return ""


# ========== 主评测流程（前1000条样本） ==========
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

  # # 截取前1000条有效样本
  # if len(valid_samples) > MAX_SAMPLES:
  #   valid_samples = valid_samples[:MAX_SAMPLES]
  #   print(f"\n🔍 已截取前 {MAX_SAMPLES} 条有效样本进行评测")
  # else:
  #   print(f"\n🔍 有效样本数不足{MAX_SAMPLES}条，将评测全部{len(valid_samples)}条")

  # 2. 加载Paraformer模型（完全按你的配置）
  print(f"\n🔄 加载Paraformer-large-vad-punc模型（{DEVICE}）...")
  model = AutoModel(
      model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
      vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
      vad_kwargs={"max_single_segment_time": 60000},  # 你的60秒分段配置
      punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
      disable_update=True
      # spk_model="iic/speech_campplus_sv_zh-cn_16k-common",  # 注释保持和你的配置一致
  )
  print(f"✅ Paraformer模型加载完成！")

  # 3. 逐条推理并计算Metric
  all_results = []
  cer_list = []
  wer_list = []
  acc_list = []

  pbar = tqdm(valid_samples, desc="Paraformer评测中")
  for sample in pbar:
    try:
      audio_path = sample["audio_filepath"]
      ref_text = sample["text"]

      # 模型推理
      pred_text = paraformer_inference(model, audio_path)

      # 计算Metric（和Whisper/SenseVoice用同一套逻辑）
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

      # 累计Metric用于统计
      cer_list.append(metrics["cer"])
      wer_list.append(metrics["wer"])
      acc_list.append(metrics["acc"])

      # 更新进度条（实时显示平均指标）
      pbar.set_postfix({
        "平均CER": f"{np.mean(cer_list):.4f}",
        "平均WER": f"{np.mean(wer_list):.4f}",
        "平均ACC": f"{np.mean(acc_list):.4f}"
      })
    except Exception as e:
      # 单样本出错时跳过，不中断整体评测
      print(f"\n⚠️ 样本{os.path.basename(sample['audio_filepath'])}处理失败：{str(e)[:50]}...")
      continue

  # 4. 生成评测报告（和前两者格式完全一致，便于对比）
  if not all_results:
    print("\n❌ 无有效评测结果！")
    return

  # 计算整体指标
  avg_cer = round(np.mean(cer_list), 4)
  avg_wer = round(np.mean(wer_list), 4)
  avg_acc = round(np.mean(acc_list), 4)
  med_cer = round(np.median(cer_list), 4)
  med_wer = round(np.median(wer_list), 4)
  med_acc = round(np.median(acc_list), 4)

  # 找出Top5坏例
  worst_cases = sorted(all_results, key=lambda x: x["cer"], reverse=True)[:5]

  # 保存JSON详细报告
  report = {
    "model": "Paraformer-large-vad-punc",
    "device": DEVICE,
    # "max_samples_limit": MAX_SAMPLES,
    "actual_sample_count": len(all_results),
    "overall_metrics": {
      "average": {"cer": avg_cer, "wer": avg_wer, "acc": avg_acc},
      "median": {"cer": med_cer, "wer": med_wer, "acc": med_acc}
    },
    "worst_cases": worst_cases,
    "all_samples": all_results
  }

  json_report_path = "./paraformer_sichuan_eval_dialect.json"
  with open(json_report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=4)

  # 保存简易文本报告（便于阅读和对比）
  txt_report_path = "./paraformer_sichuan_eval_dialect .txt"
  with open(txt_report_path, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write(f"Paraformer-large-vad-punc 四川方言评测报告\n")
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
  print(f"✅ Paraformer-large-vad-punc 评测完成！")
  print(f"📊 核心指标：")
  print(f"   平均CER：{avg_cer}（越小越好）")
  print(f"   平均WER：{avg_wer}（越小越好）")
  print(f"   平均ACC：{avg_acc}（越大越好）")
  print(f"\n📄 评测报告已保存：")
  print(f"   JSON详情：{json_report_path}")
  print(f"   文本摘要：{txt_report_path}")
  print("=" * 60)


# ========== 依赖安装（终端执行） ==========
# pip install funasr jiwer tqdm numpy torch librosa soundfile

if __name__ == "__main__":
  main()