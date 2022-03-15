# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
''' Logging
Version     Date    Change_by   Description
#00     2022-Feb-28     ZhaoTongtong,ZengRui   Initial version  


'''
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


import os
from datetime import datetime
import pandas as pd
import subprocess
import re

# import filetype # to check file type

import malaya_speech
from pyannote.audio import Pipeline as pa_Pipeline
from pydub import AudioSegment


import uob_audiosegmentation, uob_noisereduce, uob_speakerdiarization, uob_mainprocess
from init import (
    pretrained_model_path,
    AUDIO_NAME,
    AUDIO_PATH,
    AUDIO_FILE,
    SAMPLE_RATE,
    sd_global_starttime
)


# TODO: get uploaded file name from Front-End




#### TODO: Convert to .wav / Resampling
#### * Check if an Audio file
# kind = filetype.guess(AUDIO_FILE)
# if kind is None:
#     print('Cannot guess file type!')

# print('File extension: %s' % kind.extension)
# print('File MIME type: %s' % kind.mime)


#### * Convert to .wav
# audioname1='speech.mp3'
# audiopath_from1='uob/from'
# audiopath_to1='uob/to'
# audiofile1 = os.path.join(audiopath_from1,audioname1)
# AudioSegment.from_file(audiofile1).export(os.path.join(audiopath_to1,audioname1),format='wav')  # TODO: param setup - codec 



#### * Declare variables --> Declare in <init.py>
# AUDIO_NAME = 'The-Singaporean-White-Boy.wav' #'Bdb001_interaction_first60s.wav' #'The-Singaporean-White-Boy.wav'
# AUDIO_PATH = './wav/'
# AUDIO_FILE = os.path.join(AUDIO_PATH,AUDIO_NAME)
# SAMPLE_RATE = 44100

num_audio_files = 1

starttime = datetime.now()




#### * Load audio file
y, sr = malaya_speech.load(AUDIO_FILE, SAMPLE_RATE)
audio_duration = len(y) / sr
print('*' * 30)
print('length:',len(y), 'sample rate:', sr, 'duration(s):',audio_duration)
print('*' * 30)



#### * Load models
## Noise reduce models
# nr_model = uob_noisereduce.load_noisereduce_model(quantized=False)
nr_model = uob_noisereduce.load_noisereduce_model_local(quantized=False)
## Load malaya vad model
# vad_model_vggvox2 = uob_speakerdiarization.load_vad_model(quantized=False)
vad_model_vggvox2 = uob_speakerdiarization.load_vad_model_local(quantized=False)
## Load malaya speaker vector model 
# sv_model_speakernet, sv_model_vggvox2 = uob_speakerdiarization.load_speaker_vector_model(quantized=False)
sv_model_speakernet, sv_model_vggvox2 = uob_speakerdiarization.load_speaker_vector_model_local(quantized=False)
## Load pyannote.audio pipeline for sd
pa_pipeline = None
pa_pipeline = pa_Pipeline.from_pretrained('pyannote/speaker-diarization')  # TODO: uncomment for Pyannote.audio model. !!specturalcluster package needs to be updated.
print('Pretrained Models Loading Done!!!')
print('*' * 30)


#### * Segmentation
chunksfolder = ''
if audio_duration > 300:  # segment if longer than 5 min=300s
    totalchunks, nowtime = uob_audiosegmentation.audio_segmentation(name=AUDIO_NAME,file=AUDIO_FILE)
    print('  Segmentation Done!!!\n','*' * 30)
    chunksfolder = 'chunks_'+AUDIO_NAME[:5]+'_'+nowtime  #'./chunks'+nowtime 
    chunksfolder = os.path.join(AUDIO_PATH, chunksfolder)
    print('chunksfolder: ', chunksfolder)
    # num_audio_files = len([n for n in os.listdir(chunksfolder+"/") if n.endswith(".wav")])
    num_audio_files = totalchunks

    
print('Number of audio files to process: ', num_audio_files)
print('*' * 30)


#### * Process SD
# chunksfolder = './wav/chunks_The-S_20220309_185316' #'chunks_The-S_20220228_162841'   # * for test
tem_sd_result = []
tem_sd_index = 0
if chunksfolder != '':
    for filename in os.listdir(chunksfolder+"/"):
        if filename.endswith(".wav"): 
            # print(os.path.join(chunksfolder, filename))
            ### * Load chunk file
            file = os.path.join(chunksfolder, filename)
            y, sr = malaya_speech.load(file, SAMPLE_RATE)
                       
            ### * Process: reduce noise + vad + scd + ovl + sd
            ## return list[index, start, end, duration, speaker_label]
            sd_result = uob_mainprocess.sd_process(y, sr, 
                                                audioname=filename,
                                                audiopath=chunksfolder,
                                                audiofile=file,
                                                nr_model=nr_model,   # ?: [nr_model, nr_quantized_model]
                                                vad_model=vad_model_vggvox2,
                                                sv_model=sv_model_speakernet,    # ?: sv_model_speakernet, sv_model_vggvox2
                                                pipeline=pa_pipeline,
                                                chunks=True, #fixed
                                                reducenoise=False,
                                                sd_proc='malaya')  # ?: [pyannoteaudio, malaya]
            
            
            # ### * Cut audio by SD result
            # slices_path = chunksfolder + '/slices'
            # if not os.path.exists(slices_path):
            #     os.mkdir(slices_path)
            # uob_mainprocess.cut_audio_by_timestamps(start_end_list=sd_result, audioname=filename, audiofile=file, part_path=slices_path)
            # print('*'*30, 'Cut')
                    
            for row in sd_result[1:]:
                if 'not' not in row[4].lower():
                    tem_sd_result.append( [tem_sd_index + 1,
                                            row[1]+sd_global_starttime, 
                                            row[1]+sd_global_starttime+row[3],
                                            row[3],
                                            row[4]])


else:
    ### * Load single file
    y, sr = malaya_speech.load(AUDIO_FILE, SAMPLE_RATE)

    ### * Process: reduce noise + vad + scd + ovl + sd
    ## return list[index, start, end, duration, speaker_label]
    sd_result = uob_mainprocess.sd_process(y, sr, 
                                        audioname=AUDIO_NAME,
                                        audiopath=AUDIO_PATH,
                                        audiofile=AUDIO_FILE,
                                        nr_model=nr_model,   # ?: [nr_model, nr_quantized_model]
                                        vad_model=vad_model_vggvox2,
                                        sv_model=sv_model_speakernet,    # ?: sv_model_speakernet, sv_model_vggvox2
                                        pipeline=pa_pipeline,
                                        chunks=False, #fixed
                                        reducenoise=False, 
                                        sd_proc='malaya')  # ?: [pyannoteaudio, malaya]
    
    # ### * Cut audio by SD result
    # namef, namec = os.path.splitext(AUDIO_NAME)
    # slices_path = AUDIO_PATH+namef+'_slices'
    # # slices_path = os.path.join(AUDIO_PATH, namef, 'slices').replace('\\','/')
    # if not os.path.exists(slices_path):
    #     os.mkdir(slices_path)
    # uob_mainprocess.cut_audio_by_timestamps(start_end_list=sd_result, audioname=AUDIO_NAME, audiofile=AUDIO_FILE, part_path=slices_path)
    # print('*'*30, 'Cut')

    for row in sd_result[1:]:
        if 'not' not in row[4].lower():
            tem_sd_result.append([tem_sd_index + row[0],
                                    row[1],
                                    row[2],
                                    row[3],
                                    row[4]])


# TODO: to move to the end after STT is done
# Clear variables
sd_global_starttime = 0.0

endtime = datetime.now()

print('*' * 30,'\n  Finished!!',)
print('start from:', starttime) 
print('end at:', endtime) 
print('duration: ', endtime-starttime)








#### * Process STT

final_sd_result = pd.DataFrame(tem_sd_result, columns=['index','starttime','endtime','duration','speaker_label'])
print(final_sd_result)
# quit()  # TODO: comment after STT section is done completely

###  Cut audio by SD result
namef, namec = os.path.splitext(AUDIO_NAME)
slices_path = AUDIO_PATH + namef + '_slices'  # TODO: where to save slices?
if not os.path.exists(slices_path):
    os.mkdir(slices_path)
# TODO: next line, audio file is noisereduced or not?
uob_mainprocess.cut_audio_by_timestamps(start_end_list=tem_sd_result, audioname=AUDIO_NAME, audiofile=AUDIO_FILE, part_path=slices_path)
print('*'*30, 'Cut Slices Done')

###  Speech to Text Conversion and get the output csv file

subprocess.call(["python","stt.py"],cwd= '.pretrained-model/stt')
stt = pd.read_csv ('.pretrained-model/stt/output.csv')

# # Load VOSK model
# from vosk import Model, KaldiRecognizer, SetLogLevel
# print('*'*30, 'Loading Model')
# sample_rate=16000
# model = Model("model")
# rec = KaldiRecognizer(model, sample_rate)

# # STT Start
# import wave
# import json

# print('*'*30, 'STT Start')
# transcription = []

# stt = pd.DataFrame(columns=['index', 'text'])
# for filename in os.listdir("./audio/"): #(slices_path+"/"):
#     if filename.endswith(".wav"): 
#         namef, namec = os.path.splitext(filename)
#         namef_other, namef_index = namef.rsplit("_", 1)
#         namef_index = int(namef_index)
        
#         wf = wave.open("./audio/"+filename, "rb")   # Change when confirm
#         # file = slices_path+"/"+filename
#         while True:
#             data = wf.readframes(10000000)
#             if len(data) == 0:
#                 break
#             if rec.AcceptWaveform(data):
#                 # Convert json output to dict
#                 result_dict = json.loads(rec.Result())
#                 # Extract text values and append them to transcription list
#                 transcription.append(result_dict.get("text", ""))

#         # Get final bits of audio and flush the pipeline
#         final_result = json.loads(rec.FinalResult())
#         transcription.append(final_result.get("text", ""))

#         # merge or join all list elements to one big string
#         transcription_text = ' '.join(transcription)
#         print(transcription_text)  # For output check only, remove after confirm.

#         stt.loc[len(stt)] = [namef_index, transcription_text]
#         transcription_text = ""
#         transcription = []

#         # text = subprocess.check_output(["python","ffmpeg.py",filename],cwd= './stt').decode("utf-8")
#         # textNew = re.findall('"([^"]*)"', text)
#         # index = namef_index  # index = re.findall(r'\d+',filename)[-1]
#         # stt.loc[len(stt)] = [index, textNew[1]]
        
# # stt['index'] = stt['index'].astype(int)
print(stt)
print(final_sd_result)
final = pd.merge(left = final_sd_result, right = stt, on="index",how='left')
final.to_csv('output.csv')




