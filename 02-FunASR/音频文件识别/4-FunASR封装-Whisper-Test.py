from funasr import AutoModel

model = AutoModel(
    model="Whisper-large-v3-turbo",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    vad_kwargs={"max_single_segment_time": 30000},
    disable_update=True
)

DecodingOptions = {
  "task": "transcribe",
  # "language": None, 识别出来有些会自动转成英语
  "language": 'zh',
  "beam_size": None,
  "fp16": False,
  "without_timestamps": False,
  "prompt": None,
}
res = model.generate(
    DecodingOptions=DecodingOptions,
    batch_size_s=0,
    # input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
    input="/Users/surui/PycharmProjects/MY_ASR/01-音频录制/音频1-新闻.wav",
)
print(res)

res = model.generate(
    DecodingOptions=DecodingOptions,
    batch_size_s=0,
    # input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
    input="/Users/surui/PycharmProjects/MY_ASR/01-音频录制/音频2-会议.wav",
)
print(res)

res = model.generate(
    DecodingOptions=DecodingOptions,
    batch_size_s=0,
    input="/Users/surui/PycharmProjects/MY_ASR/01-音频录制/音频3-口音.wav",
)
print(res)

res = model.generate(
    DecodingOptions=DecodingOptions,
    batch_size_s=0,
    input="/Users/surui/PycharmProjects/MY_ASR/01-音频录制/音频4-绕口令.wav",
)
print(res)

res = model.generate(
    DecodingOptions=DecodingOptions,
    batch_size_s=0,
    input="/Users/surui/PycharmProjects/MY_ASR/01-音频录制/音频5-五台山论道.wav",
)
print(res)
