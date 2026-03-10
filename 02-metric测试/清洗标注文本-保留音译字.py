import os
import re
import json

def clean_text_keep_dialect(text):
    # 1. 删除【...】注释
    text = re.sub(r'【[^】]*】', '', text)
    # 2. 去掉“”或""，保留内容
    text = re.sub(r'[“"]([^”"]*)[”"]', r'\1', text)
    # 3. 只保留汉字、字母、数字、空格
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
    # 4. 清理多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    input_list_file = "./sichuan/list.txt"
    output_jsonl_file = "sichuan_asr_manifest_dialect.jsonl"
    base_audio_dir = "./sichuan/wav"

    with open(input_list_file, "r", encoding="utf-8") as fin, \
         open(output_jsonl_file, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=3)
            if len(parts) != 4:
                print(f"Skipping invalid line: {line}")
                continue

            _, wav_name, speaker_id, raw_text = parts
            audio_path = os.path.join(base_audio_dir, speaker_id, wav_name).replace("\\", "/")
            clean_txt = clean_text_keep_dialect(raw_text)

            if not clean_txt:
                continue

            entry = {
                "audio_filepath": audio_path,
                "text": clean_txt,
                "speaker_id": speaker_id
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ 清洗完成！文件：{output_jsonl_file}")

if __name__ == "__main__":
    main()