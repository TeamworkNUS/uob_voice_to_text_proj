# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
''' Logging
Version     Date    Change_by   Description
#00     2022-Feb-28     ZhaoTongtong,ZengRui   Initial version  


'''
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


import os
from datetime import datetime
import pandas as pd
import shutil
import subprocess
import re


import librosa
import soundfile as sf
import malaya_speech
from pyannote.audio import Pipeline as pa_Pipeline
from pydub import AudioSegment


import uob_audiosegmentation, uob_noisereduce, uob_speechenhancement, uob_speakerdiarization, uob_stt, uob_mainprocess, uob_utils, uob_label
from init import (
    pretrained_model_path,
    AUDIO_NAME,
    AUDIO_PATH,
    AUDIO_FILE,
    SAMPLE_RATE,
    STT_SAMPLERATE,
    FLG_REDUCE_NOISE,
    FLG_SPEECH_ENHANCE,
    sd_global_starttime
)


# TODO: get uploaded file name from Front-End
# upload_AUDIO_NAME = AUDIO_NAME
# upload_AUDIO_PATH = AUDIO_PATH



#### TODO: Convert to .wav / Resampling
#### * Check if an Audio/wav file & convert to WAV file
# file_extension, file_mime = uob_utils.check_file_type(upload_AUDIO_NAME, upload_AUDIO_PATH)
# if file_extension == 'wav' and 'audio/' in file_mime and 'wav' in file_mime:
#     AUDIO_NAME = upload_AUDIO_NAME
#     AUDIO_PATH = upload_AUDIO_PATH
# else:
#     namef, namec = os.path.splitext(upload_AUDIO_NAME)
#     AUDIO_NAME = namef + '.wav'
#     uob_utils.audio2wav(from_audioname=upload_AUDIO_NAME, from_audiopath=upload_AUDIO_PATH, to_audioname=AUDIO_NAME, to_audiopath=AUDIO_PATH)

#### * Standardize Audio -->No need, malaya_speech.read() func can resample
# uob_utils.get_audio_params(AUDIO_NAME, AUDIO_PATH)
# uob_utils.standardize_audio(from_audioname=AUDIO_NAME, from_audiopath=AUDIO_PATH, to_audioname=AUDIO_NAME, to_audiopath=AUDIO_PATH, sample_rate=44100, no_of_channel=1)
# uob_utils.get_audio_params(AUDIO_NAME, AUDIO_PATH)


#### * Declare variables --> Declare in <init.py>
# AUDIO_NAME = 'The-Singaporean-White-Boy.wav' #'Bdb001_interaction_first60s.wav' #'The-Singaporean-White-Boy.wav'
# AUDIO_PATH = './wav/'
# AUDIO_FILE = os.path.join(AUDIO_PATH,AUDIO_NAME)
# SAMPLE_RATE = 44100

num_audio_files = 1

starttime = datetime.now()




#### * Load audio file
# y, sr = malaya_speech.load(AUDIO_FILE, SAMPLE_RATE)
# audio_duration = len(y) / sr
# print('*' * 30)
# print('length:',len(y), 'sample rate:', sr, 'duration(s):',audio_duration)
# print('*' * 30)

orig_params, orig_channels, orig_samplerate, orig_sampwidth, orig_nframes = uob_utils.get_audio_params(AUDIO_NAME,AUDIO_PATH)
audio_duration = orig_nframes / orig_samplerate
print('*' * 30)
print('length:',orig_nframes, 'sample rate:', orig_samplerate, 'duration(s):',audio_duration)
print('*' * 30)



#### * Load SD models
## Noise reduce models
# nr_model = uob_noisereduce.load_noisereduce_model(quantized=False)
nr_model = uob_noisereduce.load_noisereduce_model_local(quantized=False)
## Speech Reduce models
# se_model = uob_speechenhancement.load_speechenhancement_model(model='unet',quantized=True)
se_model = uob_speechenhancement.load_speechenhancement_model_local(quantized=False)
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
if audio_duration > 3600:  # segment if longer than 5 min=300s
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
            # y, sr = malaya_speech.load(file, SAMPLE_RATE)
            y, sr = librosa.load(file, sr=None, mono=True)  #-->mono: y = y/2

                       
            ### * Process: reduce noise + vad + scd + ovl + sd
            ## return list[index, start, end, duration, speaker_label]
            sd_result = uob_mainprocess.sd_process(y, sr, 
                                                audioname=filename,
                                                audiopath=chunksfolder,
                                                audiofile=file,
                                                nr_model=nr_model,   # ?: [nr_model, nr_quantized_model]
                                                se_model=se_model,
                                                vad_model=vad_model_vggvox2,
                                                sv_model=sv_model_speakernet,    # ?: sv_model_speakernet, sv_model_vggvox2
                                                pipeline=pa_pipeline,
                                                chunks=True, #fixed
                                                reducenoise=FLG_REDUCE_NOISE,
                                                speechenhance=FLG_SPEECH_ENHANCE,
                                                sd_proc='resemblyzer')  # ?: [pyannoteaudio, malaya, resemblyzer]
            
            
            # ### * Cut audio by SD result
            # slices_path = chunksfolder + '/slices'
            # if not os.path.exists(slices_path):
            #     os.mkdir(slices_path)
            # uob_mainprocess.cut_audio_by_timestamps(start_end_list=sd_result, audioname=filename, audiofile=file, part_path=slices_path)
            # print('*'*30, 'Cut')
                    
            for row in sd_result[1:]:
                if 'not' not in row[4].lower():
                    tem_sd_index += 1
                    tem_sd_result.append( [tem_sd_index,
                                            row[1]+sd_global_starttime, 
                                            row[2]+sd_global_starttime,
                                            row[3],
                                            row[4]])
                    if row[0] == len(sd_result[1:]):
                        sd_global_starttime += row[2]
                    
            


else:
    ### * Load single file
    # y, sr = malaya_speech.load(AUDIO_FILE, SAMPLE_RATE)
    y, sr = librosa.load(AUDIO_FILE, sr=None, mono=True)  #-->mono: y = y/2
    # audioname = str(sr)+"_"+AUDIO_NAME
    # sf.write(os.path.join(AUDIO_PATH,audioname), y, sr)

    ### * Process: reduce noise + vad + scd + ovl + sd
    ## return list[index, start, end, duration, speaker_label]
    sd_result = uob_mainprocess.sd_process(y, sr, 
                                        audioname=AUDIO_NAME,
                                        audiopath=AUDIO_PATH,
                                        audiofile=AUDIO_FILE,
                                        nr_model=nr_model,   # ?: [nr_model, nr_quantized_model]
                                        se_model=se_model,
                                        vad_model=vad_model_vggvox2,
                                        sv_model=sv_model_speakernet,    # ?: sv_model_speakernet, sv_model_vggvox2
                                        pipeline=pa_pipeline,
                                        chunks=False, #fixed
                                        reducenoise=FLG_REDUCE_NOISE, 
                                        speechenhance=FLG_SPEECH_ENHANCE,
                                        sd_proc='resemblyzer')  # ?: [pyannoteaudio, malaya, resemblyzer]
    
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
            tem_sd_index += 1
            tem_sd_result.append([tem_sd_index,
                                    row[1],
                                    row[2],
                                    row[3],
                                    row[4]])



#### * Process STT

final_sd_result = pd.DataFrame(tem_sd_result, columns=['index','starttime','endtime','duration','speaker_label'])

###  Cut audio by SD result
namef, namec = os.path.splitext(AUDIO_NAME)
slices_path = AUDIO_PATH + namef + '_slices'  # TODO: where to save slices?
try:
    if not os.path.exists(slices_path):
        os.mkdir(slices_path)
    elif os.path.exists(slices_path):
        shutil.rmtree(slices_path) # delete whole folder
        os.mkdir(slices_path)
except Exception as e:
    print('Failed to delete or create slices folder %s. Reason: %s' % (slices_path, e))

if FLG_REDUCE_NOISE == False:
    if chunksfolder != '':
        for filename in os.listdir(chunksfolder+"/"):
            if filename.endswith(".wav"): 
                uob_mainprocess.cut_audio_by_timestamps(start_end_list=tem_sd_result, audioname=filename, audiofile=os.path.join(chunksfolder,filename), part_path=slices_path)
    else:
        uob_mainprocess.cut_audio_by_timestamps(start_end_list=tem_sd_result, audioname=AUDIO_NAME, audiofile=AUDIO_FILE, part_path=slices_path)
elif FLG_REDUCE_NOISE == True and FLG_SPEECH_ENHANCE == False:
    if chunksfolder != '':
        for filename in os.listdir(chunksfolder+"/"):
            if filename.endswith("_nr.wav"): 
                uob_mainprocess.cut_audio_by_timestamps(start_end_list=tem_sd_result, audioname=filename, audiofile=os.path.join(chunksfolder,filename), part_path=slices_path)
    else:
        filename = namef + "_nr.wav"
        file = os.path.join(AUDIO_PATH, filename)
        uob_mainprocess.cut_audio_by_timestamps(start_end_list=tem_sd_result, audioname=filename, audiofile=file, part_path=slices_path)
elif FLG_REDUCE_NOISE == True and FLG_SPEECH_ENHANCE == True:
    if chunksfolder != '':
        for filename in os.listdir(chunksfolder+"/"):
            if filename.endswith("_nr_se.wav"): 
                uob_mainprocess.cut_audio_by_timestamps(start_end_list=tem_sd_result, audioname=filename, audiofile=os.path.join(chunksfolder,filename), part_path=slices_path)
    else:
        filename = namef + "_nr_se.wav"
        file = os.path.join(AUDIO_PATH, filename)
        uob_mainprocess.cut_audio_by_timestamps(start_end_list=tem_sd_result, audioname=filename, audiofile=file, part_path=slices_path)
else:
     raise Exception(
                f'There is an exception when cutting audio into slices.'
            )

print('*'*30, 'Cut Slices Done')

###  Speech to Text Conversion
## Load VOSK model
stt_model_vosk_rec = uob_stt.load_stt_model(stt_model='vosk',pretrained_model_path=os.path.join(pretrained_model_path,'stt/model'), sr=STT_SAMPLERATE)
## STT start
print('*'*30)
print('STT Conversion Start')
stt = uob_mainprocess.stt_process(slices_path=slices_path, rec=stt_model_vosk_rec, sr = STT_SAMPLERATE)
print('*'*30)
print('STT Conversion Done')

### merge SD and STT
transactionDf = pd.merge(left = final_sd_result, right = stt, on="index",how='left')


###  Speaker Labelling
print('*'*30)
print("Speaker Labelling Start")
final = uob_mainprocess.speaker_label_func(transactionDf,pretrained_model_path=os.path.join(pretrained_model_path,'label/label_wordvector_model'),checklist_path=os.path.join(pretrained_model_path,'label/'))
print('*'*30)
print("Speaker Labelling Done")


# print(stt)
# print(final_sd_result)
print(final)
final.to_csv(os.path.splitext(AUDIO_NAME)[0] + '_output.csv')

### Store output to database
print('*'*30)
print("Insert Output to Database Start")
# uob_mainprocess.dbInsertSTT_func(finalDf=final, orig_path=AUDIO_PATH, processed_path=AUDIO_PATH, slices_path=slices_path) #TODO: change path to path+name
print('*'*30)
print("Insert Output to Database Done")

# Clear variables
sd_global_starttime = 0.0

endtime = datetime.now()

print('*' * 30,'\n  Finished!!',)
print('start from:', starttime) 
print('end at:', endtime) 
print('duration: ', endtime-starttime)