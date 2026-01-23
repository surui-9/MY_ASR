from funasr import AutoModel

# 例1 (带vad的paraformer)
model = AutoModel(
    model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    vad_kwargs={"max_single_segment_time": 60000},
    punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    disable_update=True
    # spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
)

res1 = model.generate(
    # input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
    input="/Users/surui/PycharmProjects/MY_ASR/01-音频录制/音频1-新闻.wav",
    cache={},
)
print(res1)

res2 = model.generate(
    input="/Users/surui/PycharmProjects/MY_ASR/01-音频录制/音频2-会议.wav",
    cache={},
)
print(res2)

res3 = model.generate(
    input="/Users/surui/PycharmProjects/MY_ASR/01-音频录制/音频3-口音.wav",
    cache={},
)
print(res3)

res4 = model.generate(
    input="/Users/surui/PycharmProjects/MY_ASR/01-音频录制/音频4-绕口令.wav",
    cache={},
)
print(res4)

res5 = model.generate(
    input="/Users/surui/PycharmProjects/MY_ASR/01-音频录制/音频5-五台山论道.wav",
    cache={},
)
print(res5)

