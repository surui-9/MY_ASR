import os
import json

# 设置根目录（你解压后的文件夹）
root_dir = "./sichuan"  # 请根据实际情况修改

list_file = os.path.join(root_dir, "list.txt")
output_jsonl = "sichuan_eval.jsonl"

samples = []
with open(list_file, "r", encoding="utf-8") as f:
  for line in f:
    parts = line.strip().split("\t")
    if len(parts) < 5:
      continue
    _, audio_filename, speaker_id, _, text = parts

    # ✅ 关键：构建真实音频路径
    audio_path = os.path.join(root_dir, "wav", speaker_id, audio_filename)

    # 检查文件是否存在（避免路径错误）
    if os.path.exists(audio_path):
      samples.append({
        "audio_filepath": audio_path,
        "text": text.strip(),
        "speaker_id": speaker_id
      })
    else:
      print(f"⚠️ 文件不存在: {audio_path}")

# 保存为 JSONL（通用评测格式）
with open(output_jsonl, "w", encoding="utf-8") as fout:
  for sample in samples:
    fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"✅ 成功处理 {len(samples)} 条有效样本，保存至 {output_jsonl}")