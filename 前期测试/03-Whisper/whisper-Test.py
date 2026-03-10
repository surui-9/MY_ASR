import whisper

# 1. 加载模型（首次运行会自动下载 ～1.5GB）
# 可选: tiny, base, small, medium, large, large-v2, large-v3, large-v3-turbo
model = whisper.load_model("base", device="cpu")  # M1 用 cpu

# 2. 指定音频文件（支持 .wav, .mp3, .m4a 等常见格式）
# audio_path = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
audio_path = "/01-音频录制/音频1-新闻.wav"

# 3. 执行识别（自动处理采样率、声道等）
result = model.transcribe(
    audio_path,
    language="zh",      # 强制中文，提升准确率 & 速度
    fp16=False,
    verbose=False        # 显示进度（可选）
)

# 4. 打印结果
print("\n 识别结果:")
print(result["text"].strip())


audio_path = "/01-音频录制/音频2-会议.wav"

# 3. 执行识别（自动处理采样率、声道等）
result = model.transcribe(
    audio_path,
    language="zh",      # 强制中文，提升准确率 & 速度
    fp16=False,
    verbose=False        # 显示进度（可选）
)

# 4. 打印结果
print("\n 识别结果:")
print(result["text"].strip())


audio_path = "/01-音频录制/音频3-口音.wav"

# 3. 执行识别（自动处理采样率、声道等）
result = model.transcribe(
    audio_path,
    language="zh",      # 强制中文，提升准确率 & 速度
    fp16=False,
    verbose=False        # 显示进度（可选）
)

# 4. 打印结果
print("\n 识别结果:")
print(result["text"].strip())


audio_path = "/01-音频录制/音频4-绕口令.wav"

# 3. 执行识别（自动处理采样率、声道等）
result = model.transcribe(
    audio_path,
    language="zh",      # 强制中文，提升准确率 & 速度
    fp16=False,
    verbose=False        # 显示进度（可选）
)

# 4. 打印结果
print("\n 识别结果:")
print(result["text"].strip())


audio_path = "/01-音频录制/音频5-五台山论道.wav"

# 3. 执行识别（自动处理采样率、声道等）
result = model.transcribe(
    audio_path,
    language="zh",      # 强制中文，提升准确率 & 速度
    fp16=False,
    verbose=False        # 显示进度（可选）
)

# 4. 打印结果
print("\n 识别结果:")
print(result["text"].strip())