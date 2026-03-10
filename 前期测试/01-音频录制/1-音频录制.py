import sounddevice as sd
import soundfile as sf

# 设置录制时间
duration = 20
sample_rate = 16000

print(f"开始录音 {duration} 秒...")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()  # 等待录音完成
file_name= "音频5-五台山论道.wav"
sf.write(file_name, audio, sample_rate)
print(f"录音已保存为 {file_name}")