import os
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
from funasr import AutoModel

# ========== 全局统一配置（M1 Mac适配） ==========
# 音频参数（两个模型都兼容16k采样率）
SAMPLE_RATE = 16000
DURATION = 10  # 每次录音10秒
MIC_DEVICE_ID = 0  # 替换成你的麦克风ID（可通过python -m sounddevice.query_devices查询）

# 模型缓存路径
os.environ["MODELSCOPE_CACHE"] = os.path.expanduser("~/.cache/modelscope")
os.environ["WHISPER_CACHE"] = os.path.expanduser("~/.cache/whisper")


# ========== 初始化两个模型（02-FunASR + 03-Whisper） ==========
def init_models():
  """初始化所有测试模型，确保加载成功"""
  models = {}

  # 1. 初始化FunASR nano模型（适配1.3.0版本）
  print("🔄 加载FunASR nano模型...")
  try:
    models["funasr"] = AutoModel(
        model="damo/speech_paraformer-large-vad-punc_asr_nano-zh",
        model_revision="v2.0.4",
        device="cpu",
        disable_update=True
    )
    print("✅ 02-FunASR nano模型加载成功！")
  except Exception as e:
    print(f"❌ FunASR加载失败：{str(e)}")
    print("💡 请检查模型ID或手动下载模型后指定本地路径")

  # 2. 初始化Whisper模型（base版本，平衡速度和效果）
  print("\n🔄 加载Whisper base模型...")
  try:
    models["1-whisper"] = whisper.load_model("base", device="cpu")
    print("✅ 03-Whisper base模型加载成功！")
  except Exception as e:
    print(f"❌ Whisper加载失败：{str(e)}")

  return models


# ========== 通用录音函数（所有模型共用） ==========
def record_audio():
  """录制音频，返回两个模型都兼容的格式"""
  print(f"\n🎤 请说话（{DURATION}秒）...")
  # 录制float32格式音频（Whisper原生支持）
  audio_float32 = sd.rec(
      int(DURATION * SAMPLE_RATE),
      samplerate=SAMPLE_RATE,
      channels=1,
      dtype=np.float32,
      device=MIC_DEVICE_ID
  )
  sd.wait()  # 等待录音完成
  audio_float32 = np.squeeze(audio_float32)

  # 转换为int16格式（FunASR要求）
  audio_int16 = (audio_float32 * 32767).astype(np.int16)

  # 保存临时文件（可选）
  write("../../03-Whisper/asr_test_temp.wav", SAMPLE_RATE, audio_int16)

  return audio_float32, audio_int16


# ========== 模型识别函数（分别适配） ==========
def recognize_funasr(model, audio_int16):
  """FunASR识别逻辑"""
  result = model.generate(
      input=audio_int16,
      cache={},
      is_final=True,
      language="zh"
  )
  if result and len(result) > 0:
    return result[0].get("text", "未识别")
  return "未识别"


def recognize_whisper(model, audio_float32):
  """Whisper识别逻辑"""
  audio = whisper.pad_or_trim(audio_float32)
  mel = whisper.log_mel_spectrogram(audio).to(model.device)
  result = model.decode(mel, language="zh")
  return result.text


# ========== 主测试逻辑 ==========
if __name__ == "__main__":
  # 初始化模型
  models = init_models()
  if not models:
    print("❌ 无可用模型，退出测试")
    exit(1)

  print("\n💡 所有模型加载完成，按Ctrl+C退出测试\n")

  try:
    while True:
      # 录制音频
      audio_float32, audio_int16 = record_audio()

      # 分别调用模型识别
      print("\n📝 识别结果对比：")
      if "funasr" in models:
        funasr_text = recognize_funasr(models["funasr"], audio_int16)
        print(f"   02-FunASR nano：{funasr_text}")
      if "1-whisper" in models:
        whisper_text = recognize_whisper(models["1-whisper"], audio_float32)
        print(f"   03-Whisper base：{whisper_text}")
      print("-" * 50)

  except KeyboardInterrupt:
    print("\n👋 测试结束！")
  except Exception as e:
    print(f"\n❌ 运行出错：{str(e)}")