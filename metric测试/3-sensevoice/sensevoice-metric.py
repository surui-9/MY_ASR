import json
import os
import jiwer
import numpy as np
from funasr import AutoModel
from tqdm import tqdm
from typing import Dict, List

model_dir="iic/SenseVoiceSmall"

# ========== 配置项 ==========
JSONL_PATH = "../sichuan_eval.jsonl"  # 你的评测JSONL文件路径
AUDIO_ROOT = "../sichuan/wav"  # 音频根目录
DEVICE = "cpu"
MAX_SAMPLES = 1000  # 仅评测前1000条有效样本



# ========== 核心Metric计算函数（和Whisper完全一致，保证对比公平） ==========
def compute_metrics(pred_text: str, ref_text: str) -> Dict:
  """统一的Metric计算逻辑，确保和Whisper评测结果可对比"""
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


def sensevoice_inference(model, audio_path: str) -> str:
  """SenseVoice模型推理封装，适配中文方言"""
  try:
    # SenseVoice推理（针对中文/方言优化）
    result = model.generate(
        input=audio_path,
        cache={},
        batch_size=1,
        # 方言识别优化参数
        # language="zh",
        language="zh-cn",
        use_pun=False,  # 关闭标点符号预测（减少干扰，评测更干净）
        use_itn=True,  # 开启逆文本规范化（数字/符号转中文）
        vad_kwargs={"max_single_segment_time": 30000,
                    "vad_merge_silence": 1000  # 合并短静音，避免音频被切得太碎
                    },  # 适配长音频
        # ========== 解码优化 ==========
        beam_size=5,  # 增大beam_size，提升方言识别准确率（牺牲一点速度）
        decoding_chunk_size=[-1, -1, -1]  # 全量解码，适合短音频
    )
    return result[0]["text"].strip()
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

  # 截取前1000条有效样本
  if len(valid_samples) > MAX_SAMPLES:
    valid_samples = valid_samples[:MAX_SAMPLES]
    print(f"\n🔍 已截取前 {MAX_SAMPLES} 条有效样本进行评测")
  else:
    print(f"\n🔍 有效样本数不足{MAX_SAMPLES}条，将评测全部{len(valid_samples)}条")

  # 2. 加载SenseVoice模型（中文方言最优模型）
  print(f"\n🔄 加载FunASR-SenseVoice模型（{DEVICE}）...")

  model = AutoModel(
      model=model_dir,
      trust_remote_code=True,
      device=DEVICE,
      language="zh",
      # vad_model="fsmn-vad",  # 开启VAD，提升长音频识别准确率
      # vad_kwargs={"max_single_segment_time": 30000},
      disable_update=True
  )
  print(f"✅ {model_dir}模型加载完成！")

  # 3. 逐条推理并计算Metric
  all_results = []
  cer_list = []
  wer_list = []
  acc_list = []

  pbar = tqdm(valid_samples, desc=f"{model_dir}评测中")
  for sample in pbar:
    try:
      audio_path = sample["audio_filepath"]
      ref_text = sample["text"]

      # 模型推理
      pred_text = sensevoice_inference(model, audio_path)

      # 计算Metric（和Whisper用同一套逻辑）
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

  # 4. 生成评测报告（和Whisper报告格式一致，便于对比）
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
    "model": model_dir,
    "device": DEVICE,
    "max_samples_limit": MAX_SAMPLES,
    "actual_sample_count": len(all_results),
    "overall_metrics": {
      "average": {"cer": avg_cer, "wer": avg_wer, "acc": avg_acc},
      "median": {"cer": med_cer, "wer": med_wer, "acc": med_acc}
    },
    "worst_cases": worst_cases,
    "all_samples": all_results
  }

  json_report_path = "./sensevoice_sichuan_eval_report_1000samples.json"
  with open(json_report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=4)

  # 保存简易文本报告（便于阅读和对比）
  txt_report_path = "./sensevoice_sichuan_eval_report_1000samples.txt"
  with open(txt_report_path, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write(f"{model_dir} 四川方言评测报告（前{MAX_SAMPLES}条）\n")
    f.write("=" * 60 + "\n")
    f.write(f"评测样本数：{len(all_results)} 条（限制最多{MAX_SAMPLES}条）\n")
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
  print(f"✅ {model_dir} 评测完成（前{MAX_SAMPLES}条）！")
  print(f"📊 核心指标：")
  print(f"   平均CER：{avg_cer}（越小越好）")
  print(f"   平均WER：{avg_wer}（越小越好）")
  print(f"   平均ACC：{avg_acc}（越大越好）")
  print(f"\n📄 评测报告已保存：")
  print(f"   JSON详情：{json_report_path}")
  print(f"   文本摘要：{txt_report_path}")
  print("=" * 60)


# ========== 依赖安装（终端执行） ==========
# pip install funasr jiwer tqdm numpy torch

if __name__ == "__main__":
  main()
