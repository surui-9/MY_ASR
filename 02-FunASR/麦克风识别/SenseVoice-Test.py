import os
import tempfile
import sounddevice as sd
import soundfile as sf
from funasr import AutoModel

# 设置环境变量（避免某些 M1 兼容性警告）
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 加载本地或自动下载的 SenseVoiceSmall 模型（首次运行会自动下载到 ～/.cache/modelscope/）
print("正在加载语音识别模型（首次运行需下载，约200MB）...")
model = AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    device="cpu",  # M1 使用 CPU（Metal 加速暂未完全支持，CPU 已足够快）
    disable_pbar=True,  # 关闭进度条更干净
)

def record_audio(duration=5, sample_rate=16000):
    """录制音频，返回音频文件路径"""
    print(f"请开始说话（最多 {duration} 秒）...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # 等待录音结束
    print("录音结束，正在识别...")

    # 保存为临时 wav 文件
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_data, sample_rate)
        return tmp.name

def main():
    try:
        while True:
            input("按回车键开始录音（输入 'q' 并回车退出）: ")
            user_input = input().strip()
            if user_input.lower() == 'q':
                print("退出程序。")
                break

            # 录音
            audio_file = record_audio(duration=5)

            # 语音识别
            result = model.generate(input=audio_file)
            text = result[0]["text"] if isinstance(result, list) and len(result) > 0 else ""

            print(f"识别结果: {text}")

            # 删除临时文件
            os.unlink(audio_file)

    except KeyboardInterrupt:
        print("\n程序被中断。")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()