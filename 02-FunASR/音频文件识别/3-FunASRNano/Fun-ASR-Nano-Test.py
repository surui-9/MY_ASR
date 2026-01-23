from funasr import AutoModel


#例1
# model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
# model = AutoModel(
#     model=model_dir,
#     disable_update=True,
#     trust_remote_code=True,
#     remote_code="./model.py",
#     device="cpu"
# )
#
# wav_path = f"{model.model_path}/example/zh.mp3"
# # wav_path = f"{model.model_path}/example/test.wav"
# print(wav_path)
# res = model.generate(input=[wav_path], cache={}, batch_size=1)
# text = res[0]["text"]
# print(text)



# 例2
# vad_model：表示开启 VAD(指定VAD模型，更适合中文的VAD模型)，VAD 的作用是区分语音和非语音
# 有专门做vad的模型，也有集成到大模型中的作为一个模块的vad
# vad_kwargs：表示 VAD 模型配置,max_single_segment_time: 表示vad_model最大切割音频时长, 单位是毫秒 ms。
# batch_size_s 表示采用动态 batch，batch 中总音频时长，单位为秒 s。
model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    remote_code="./model.py",
    device="cpu",
)
# wav_path = f"{model.model_path}/example/test.wav"
wav_path = "/Users/surui/PycharmProjects/MY_ASR/01-音频录制/音频1-新闻.wav"
res = model.generate(input=[wav_path], cache={}, batch_size=1)
text = res[0]["text"]
print(text)


wav_path = "/Users/surui/PycharmProjects/MY_ASR/01-音频录制/音频2-会议.wav"
res = model.generate(input=[wav_path], cache={}, batch_size=1)
text = res[0]["text"]
print(text)


wav_path = "/Users/surui/PycharmProjects/MY_ASR/01-音频录制/音频3-口音.wav"
res = model.generate(input=[wav_path], cache={}, batch_size=1)
text = res[0]["text"]
print(text)

wav_path = "/Users/surui/PycharmProjects/MY_ASR/01-音频录制/音频4-绕口令.wav"
res = model.generate(input=[wav_path], cache={}, batch_size=1)
text = res[0]["text"]
print(text)

wav_path = "/Users/surui/PycharmProjects/MY_ASR/01-音频录制/音频5-五台山论道.wav"
res = model.generate(input=[wav_path], cache={}, batch_size=1)
text = res[0]["text"]
print(text)

















# 不通过FunASR
# 参数说明
# model_dir：模型名称，或本地磁盘中的模型路径。
# trust_remote_code：是否信任远程代码，用于加载自定义模型实现。
# remote_code：指定模型具体代码的位置（例如，当前目录下的 model.py），支持绝对路径与相对路径。
# device：指定使用的设备，如 "cuda:0" 或 "cpu"。

# from model import 4-FunASRNano

# model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
# m, kwargs = 4-FunASRNano.from_pretrained(model=model_dir, device="cpu")
# m.eval()
#
# # wav_path = f"{kwargs['model_path']}/example/zh.mp3"
# # wav_path = f"{kwargs['model_path']}/example/test.wav"
# wav_path = "/Users/surui/PycharmProjects/MY_ASR/01-音频录制/wutaishan.wav"
# res = m.inference(data_in=[wav_path], **kwargs)
# text = res[0][0]["text"]
# print(text)
