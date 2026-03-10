from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "iic/SenseVoiceSmall"


model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    # remote_code="./model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    # device="cuda:0",
    device="cpu",
)


# zh
res1 = model.generate(
    # input=f"{model.model_path}/example/zh.mp3",
    input="/Users/surui/PycharmProjects/MY_ASR/01-音频录制/音频1-新闻.wav",
    cache={},
    language="zh",  # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(res1[0]["text"])
print(text)


res2 = model.generate(
    input="/Users/surui/PycharmProjects/MY_ASR/01-音频录制/音频2-会议.wav",
    cache={},
    language="zh",  # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(res2[0]["text"])
print(text)


res3 = model.generate(
    input="/Users/surui/PycharmProjects/MY_ASR/01-音频录制/音频3-口音.wav",
    cache={},
    language="zh",  # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(res3[0]["text"])
print(text)


res4 = model.generate(
    # input=f"{model.model_path}/example/zh.mp3",
    input="/Users/surui/PycharmProjects/MY_ASR/01-音频录制/音频4-绕口令.wav",
    cache={},
    language="zh",  # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(res4[0]["text"])
print(text)


res5 = model.generate(
    # input=f"{model.model_path}/example/zh.mp3",
    input="/Users/surui/PycharmProjects/MY_ASR/01-音频录制/音频5-五台山论道.wav",
    cache={},
    language="zh",  # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(res5[0]["text"])
print(text)