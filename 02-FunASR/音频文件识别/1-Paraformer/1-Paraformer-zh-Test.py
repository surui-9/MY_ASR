from funasr import AutoModel



# 2-paraformer-zh
# 2-paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need
model = AutoModel(model="2-paraformer-zh",
                  vad_model="fsmn-vad",
                  vad_kwargs={"max_single_segment_time": 60000},
                  punc_model="ct-punc",
                  disable_update=True
                  # spk_model="cam++"
                  )
# wav_file = f"{model.model_path}/example/asr_example.wav"
wav_file = "/Users/surui/PycharmProjects/MY_ASR/01-音频录制/音频1-新闻.wav"
res1 = model.generate(input=wav_file, batch_size_s=300, batch_size_threshold_s=60, hotword='魔搭')
print(res1)

wav_file = "/Users/surui/PycharmProjects/MY_ASR/01-音频录制/音频2-会议.wav"
res2 = model.generate(input=wav_file, batch_size_s=300, batch_size_threshold_s=60, hotword='魔搭')
print(res2)

wav_file = "/Users/surui/PycharmProjects/MY_ASR/01-音频录制/音频3-口音.wav"
res3 = model.generate(input=wav_file, batch_size_s=300, batch_size_threshold_s=60, hotword='魔搭')
print(res3)

wav_file = "/Users/surui/PycharmProjects/MY_ASR/01-音频录制/音频4-绕口令.wav"
res4 = model.generate(input=wav_file, batch_size_s=300, batch_size_threshold_s=60, hotword='魔搭')
print(res4)

wav_file = "/Users/surui/PycharmProjects/MY_ASR/01-音频录制/音频5-五台山论道.wav"
res5 = model.generate(input=wav_file, batch_size_s=300, batch_size_threshold_s=60, hotword='魔搭')
print(res5)




# 2-paraformer-zh-streaming
# chunk_size = [0, 10, 5] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
# encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
# decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention
#
# model = AutoModel(model="2-paraformer-zh-streaming")
#
# import soundfile
# import os
#
# wav_file = os.path.join(model.model_path, "example/asr_example.wav")
# speech, sample_rate = soundfile.read(wav_file)
# chunk_stride = chunk_size[1] * 960 # 600ms
#
# cache = {}
# total_chunk_num = int(len((speech)-1)/chunk_stride+1)
# for i in range(total_chunk_num):
#     speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
#     is_final = i == total_chunk_num - 1
#     res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size, encoder_chunk_look_back=encoder_chunk_look_back, decoder_chunk_look_back=decoder_chunk_look_back)
#     print(res)
