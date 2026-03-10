import os
import re
import json


def clean_text(text):
    """
    清洗四川方言标注文本：
    1. 将 "xxx"【yyy】 替换为 yyy
    2. 删除孤立的 "xxx"
    3. 去除所有标点符号，只保留中文、英文、数字、空格
    """
    # 步骤1: 匹配 “xxx”【yyy】 或 "xxx"【yyy】 并替换为 yyy
    pattern = r'[“"]([^”"]*)[”"]\s*【([^】]*)】'
    text = re.sub(pattern, r'\2', text)

    # 步骤2: 删除剩余的孤立引号内容
    text = re.sub(r'[“"][^”"]*[”"]', '', text)

    # 步骤3: 只保留汉字、英文字母、数字、空格
    cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def main():
    input_list_file = "./sichuan/list.txt"
    output_jsonl_file = "sichuan_asr_manifest.jsonl"
    base_audio_dir = "./sichuan/wav"

    with open(input_list_file, "r", encoding="utf-8") as fin, \
            open(output_jsonl_file, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            # ✅ 关键修复：使用 split(maxsplit=3) 自动处理 Tab/空格
            parts = line.split(maxsplit=3)
            if len(parts) != 4:
                print(f"Skipping invalid line: {line}")
                continue

            _, wav_name, speaker_id, raw_text = parts

            audio_path = os.path.join(base_audio_dir, speaker_id, wav_name).replace("\\", "/")
            clean_txt = clean_text(raw_text)

            if not clean_txt:
                print(f"Warning: empty text after cleaning for {wav_name}")
                continue

            entry = {
                "audio_filepath": audio_path,
                "text": clean_txt,
                "speaker_id": speaker_id
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ 清洗完成！输出文件：{output_jsonl_file}")


if __name__ == "__main__":
    main()