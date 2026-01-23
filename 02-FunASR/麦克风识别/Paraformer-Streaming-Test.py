import os
import numpy as np
import sounddevice as sd
from funasr import AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("正在加载流式 Paraformer 模型...")
model = AutoModel(
    model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    model_revision="v2.0.5",
    trust_remote_code=True,
    device="cpu",
    disable_pbar=True,
    disable_update=True,
)

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.3
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# 音频增益（关键！）
GAIN = 8.0  # 尝试 5～10，根据你的麦克风调整
SILENCE_THRESHOLD = 0.01  # 幅度低于此值视为静音

print("\n🎤 开始录音（请靠近麦克风，正常音量说话）...")
print("提示：如果仍识别为'嗯'，请尝试调大 GAIN 值（如 10 或 15）\n")

try:
    cache = {}
    speech_started = False  # 是否已经开始说话

    def audio_callback(indata, frames, time, status):
        global cache, speech_started
        if status:
            print(status)

        audio_chunk = indata[:, 0].astype(np.float32)
        # 计算当前块的 RMS 能量
        rms = np.sqrt(np.mean(audio_chunk ** 2))

        # 应用增益
        audio_chunk = audio_chunk * GAIN

        # 防止溢出 [-1, 1]
        audio_chunk = np.clip(audio_chunk, -1.0, 1.0)

        # 如果能量太低，跳过推理（减少“嗯”）
        if rms < SILENCE_THRESHOLD:
            if speech_started:
                # 可能说话结束，但为了流式体验，仍送入空 chunk 不现实
                # 我们只跳过推理，不清空 cache
                pass
            return

        # 有语音，开始/继续识别
        speech_started = True

        result = model.generate(input=audio_chunk, cache=cache, is_final=False)
        if result and isinstance(result, list) and len(result) > 0:
            output = result[0]
            cache = output.get("cache", cache)
            text = output.get("text", "").strip()
            if text:
                print(f"\r识别中: {text}", end="", flush=True)

    with sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        dtype='float32',
        blocksize=CHUNK_SIZE,
        callback=audio_callback
    ):
        input("▶️ 请开始说话（说完按回车结束）\n")

    # Final
    final = model.generate(input=np.array([]), cache=cache, is_final=True)
    if final and isinstance(final, list) and len(final) > 0:
        final_text = final[0].get("text", "").strip()
        print(f"\n✅ 最终结果: {final_text}")

except KeyboardInterrupt:
    print("\n⏹️ 中断")
except Exception as e:
    import traceback
    print(f"\n❌ 错误: {e}")
    traceback.print_exc()
finally:
    print("\n🔚 结束")