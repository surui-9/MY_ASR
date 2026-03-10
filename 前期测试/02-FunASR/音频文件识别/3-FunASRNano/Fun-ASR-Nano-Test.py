from funasr import AutoModel


#例1
model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
model = AutoModel(
    model=model_dir,
    disable_update=True,
    trust_remote_code=True,
    remote_code="./model.py",
    device="cpu"
)

wav_path = r"/01-前期测试/01-音频录制/音频1-新闻.wav"
print(wav_path)
res = model.generate(input=[wav_path], cache={}, batch_size=1)
text = res[0]["text"]
print(text)



# 例2
# vad_model：表示开启 VAD(指定VAD模型，更适合中文的VAD模型)，VAD 的作用是区分语音和非语音
# 有专门做vad的模型，也有集成到大模型中的作为一个模块的vad
# vad_kwargs：表示 VAD 模型配置,max_single_segment_time: 表示vad_model最大切割音频时长, 单位是毫秒 ms。
# batch_size_s 表示采用动态 batch，batch 中总音频时长，单位为秒 s。
# model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
# model = AutoModel(
#     model=model_dir,
#     trust_remote_code=True,
#     vad_model="fsmn-vad",
#     vad_kwargs={"max_single_segment_time": 30000},
#     remote_code="./model.py",
#     device="cuda:0",
# )
# # # wav_path = f"{model.model_path}/example/test.wav"
# wav_path = r"G:\ruc\MY_ASR\01-音频录制\音频1-新闻.wav"
# res = model.generate(input=[wav_path], cache={}, batch_size=1)
# text = res[0]["text"]
# print(text)








