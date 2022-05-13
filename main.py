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
import json


import librosa
import soundfile as sf
import malaya_speech
from pyannote.audio import Pipeline as pa_Pipeline
from pydub import AudioSegment


import uob_audiosegmentation, uob_noisereduce, uob_speechenhancement, uob_superresolution, uob_speakerdiarization, uob_stt, uob_mainprocess, uob_utils, uob_label, uob_storage, uob_speechenhancement_new
from init import (
    pretrained_model_path,
    AUDIO_NAME,
    AUDIO_PATH,
    AUDIO_FILE,
    SAMPLE_RATE,
    STT_SAMPLERATE,
    FLG_REDUCE_NOISE,
    FLG_SPEECH_ENHANCE,
    FLG_SPEECH_ENHANCE_NEW,
    FLG_SUPER_RES,
    sd_global_starttime,
    sttModel,
    sdModel,
    flg_slice_orig
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
se_model = uob_speechenhancement.load_speechenhancement_model_local(quantized=False) if FLG_SPEECH_ENHANCE == True else None
se_model_new = uob_speechenhancement_new.load_speechenhancement_model_local() if FLG_SPEECH_ENHANCE_NEW == True else None
## Super Resolution models
# sr_model = uob_superresolution.load_superresolution_model(quantized=False)
sr_model = uob_superresolution.load_superresolution_model_local(quantized=False)
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
if audio_duration > 1800:  # segment if longer than 5 min=300s
    totalchunks, nowtime, chunksfolder = uob_audiosegmentation.audio_segmentation(name=AUDIO_NAME,file=AUDIO_FILE,savetime=starttime)
    print('  Segmentation Done!!!\n','*' * 30)
    # chunksfolder = 'chunks_'+AUDIO_NAME[:5]+'_'+nowtime  #'./chunks'+nowtime 
    # chunksfolder = os.path.join(AUDIO_PATH, chunksfolder)
    print('chunksfolder: ', chunksfolder)
    # num_audio_files = len([n for n in os.listdir(chunksfolder+"/") if n.endswith(".wav")])
    num_audio_files = totalchunks

    
print('Number of audio files to process: ', num_audio_files)
print('*' * 30)


#### * Process SD
# chunksfolder = '.\\wav\\chunks_Bdb00_20220330_114451' #'./wav/chunks_The-S_20220309_185316' #'chunks_The-S_20220228_162841'   # * for test
tem_sd_result = []
tem_sd_index = 0
if chunksfolder != '':
    for filename in os.listdir(chunksfolder+"/"):
        if filename.endswith(".wav"): 
            print(os.path.join(chunksfolder, filename))
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
                                                sr_model=sr_model,
                                                vad_model=vad_model_vggvox2,
                                                sv_model=sv_model_speakernet,    # ?: sv_model_speakernet, sv_model_vggvox2
                                                pipeline=pa_pipeline,
                                                chunks=True, #fixed
                                                reducenoise=FLG_REDUCE_NOISE,
                                                speechenhance=FLG_SPEECH_ENHANCE,
                                                superresolution=FLG_SUPER_RES,
                                                speechenhance_new=FLG_SPEECH_ENHANCE_NEW,
                                                se_model_new = se_model_new,
                                                sd_proc=sdModel)  # ?: [pyannoteaudio, malaya, resemblyzer]
            
            
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
                                            row[4],
                                            row[5],
                                            row[1],
                                            row[2]])
                    if row[0] == len(sd_result[1:]):
                        sd_global_starttime += row[2]
                        print(sd_global_starttime)
                    
            


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
                                        sr_model=sr_model,
                                        vad_model=vad_model_vggvox2,
                                        sv_model=sv_model_speakernet,    # ?: sv_model_speakernet, sv_model_vggvox2
                                        pipeline=pa_pipeline,
                                        chunks=False, #fixed
                                        reducenoise=FLG_REDUCE_NOISE, 
                                        speechenhance=FLG_SPEECH_ENHANCE,
                                        superresolution=FLG_SUPER_RES,
                                        speechenhance_new=FLG_SPEECH_ENHANCE_NEW,
                                        se_model_new = se_model_new,
                                        sd_proc=sdModel)  # ?: [pyannoteaudio, malaya, resemblyzer]
    
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
                                    row[4],
                                    row[5],
                                    0,
                                    0])



#### * Process STT

final_sd_result = pd.DataFrame(tem_sd_result, columns=['index','starttime','endtime','duration','speaker_label','chunk_filename','chunk_starttime','chunk_endtime'])

###  Cut audio by SD result
namef, namec = os.path.splitext(AUDIO_NAME)
slices_path = AUDIO_PATH + namef + '_slices_' + starttime.strftime("%Y%m%d_%H%M%S")

try:
    if not os.path.exists(slices_path):
        os.mkdir(slices_path)
    elif os.path.exists(slices_path):
        shutil.rmtree(slices_path) # delete whole folder
        os.mkdir(slices_path)
except Exception as e:
    print('Failed to delete or create slices folder %s. Reason: %s' % (slices_path, e))

# Start to cut
if chunksfolder != '':
    if flg_slice_orig == False:
        for filename in os.listdir(chunksfolder+"_processed/"):
            tem_sd_result_forSlices = final_sd_result[final_sd_result['chunk_filename']==filename]
            tem_sd_result_forSlices['index'] = tem_sd_result_forSlices['index'].astype(str)
            tem_sd_result_forSlices = tem_sd_result_forSlices[['index','chunk_starttime','chunk_endtime']].values.tolist()
            uob_mainprocess.cut_audio_by_timestamps(start_end_list=tem_sd_result_forSlices, audioname=filename, audiofile=os.path.join(chunksfolder+"_processed/",filename), part_path=slices_path)
    else:
        for filename in os.listdir(chunksfolder+"/"):
            if re.match('^(%s_)(\d{4})'%(namef), filename):
                tem_sd_result_forSlices = final_sd_result[final_sd_result['chunk_filename'].str[-8:]==filename[-8:]]
                tem_sd_result_forSlices['index'] = tem_sd_result_forSlices['index'].astype(str)
                tem_sd_result_forSlices = tem_sd_result_forSlices[['index','chunk_starttime','chunk_endtime']].values.tolist()
                print(tem_sd_result_forSlices)
                uob_mainprocess.cut_audio_by_timestamps(start_end_list=tem_sd_result_forSlices, audioname=filename, audiofile=os.path.join(chunksfolder,filename), part_path=slices_path)

        
else:
# Get audio to cut into slices
    if FLG_SUPER_RES == True:
        if FLG_REDUCE_NOISE == False:
            filename_forSlices = namef + "_sr.wav"
        elif FLG_REDUCE_NOISE == True and FLG_SPEECH_ENHANCE == False:
            filename_forSlices = namef + "_nr_sr.wav"
        elif FLG_REDUCE_NOISE == True and FLG_SPEECH_ENHANCE == True:
            filename_forSlices = namef + "_nr_se_sr.wav"
        else:
            raise Exception(
                        f'There is an exception when searching for .wav file to cut into slices.'
                    )
    else:
        if FLG_REDUCE_NOISE == False:
            filename_forSlices = namef + ".wav"
        elif FLG_REDUCE_NOISE == True and FLG_SPEECH_ENHANCE == False:
            filename_forSlices = namef + "_nr.wav"
        elif FLG_REDUCE_NOISE == True and FLG_SPEECH_ENHANCE == True:
            filename_forSlices = namef + "_nr_se.wav"
        else:
            raise Exception(
                        f'There is an exception when searching for .wav file to cut into slices.'
                    )

    if flg_slice_orig == True:
        file_forSlices = AUDIO_FILE
    else:
        file_forSlices = os.path.join(AUDIO_PATH, filename_forSlices)
    uob_mainprocess.cut_audio_by_timestamps(start_end_list=tem_sd_result, audioname=filename_forSlices, audiofile=file_forSlices, part_path=slices_path)

print('*'*30, 'Cut Slices Done')

###  Speech to Text Conversion
## Load stt model
stt_model = uob_stt.load_stt_model(stt_model=sttModel,pretrained_model_path=pretrained_model_path, sr=STT_SAMPLERATE)
## STT start
print('*'*30)
print('STT Conversion Start')
stt = uob_mainprocess.stt_process(sttModel = sttModel, slices_path=slices_path, rec=stt_model, sr = STT_SAMPLERATE)
print('*'*30)
print('STT Conversion Done')

### merge SD and STT
transactionDf = pd.merge(left = final_sd_result, right = stt, on="index",how='left')
transactionDf.to_csv('transactionDf.csv')

###  Speaker Labelling
print('*'*30)
print("Speaker Labelling Start")
final = uob_mainprocess.speaker_label_func(transactionDf,pretrained_model_path=os.path.join(pretrained_model_path,'label/label_wordvector_model'),checklist_path=os.path.join(pretrained_model_path,'label/'))
print('*'*30)
print("Speaker Labelling Done")


# print(stt)
# print(final_sd_result)
print(final)
# final = final.sort_values('index')
final.to_csv(os.path.splitext(AUDIO_NAME)[0] + '_output.csv')

### Store output to database
print('*'*30)
print("Insert Output to Database Start")
audio_id = uob_storage.dbInsertAudio(upload_filename=AUDIO_NAME, upload_filepath=AUDIO_PATH, audioname=AUDIO_NAME, audiopath=AUDIO_PATH)
uob_storage.dbInsertSTT(finalDf=final, audio_id=audio_id, slices_path=slices_path)

print('*'*30)
print("Insert Output to Database Done")

# Clear variables
sd_global_starttime = 0.0

endtime = datetime.now()

print('*' * 30,'\n  Finished!!',)
print('start from:', starttime) 
print('end at:', endtime) 
print('duration: ', endtime-starttime)


### Save to Log Table
params = json.dumps({"NR":FLG_REDUCE_NOISE, "SE":FLG_SPEECH_ENHANCE, "SE_NEW":FLG_SPEECH_ENHANCE_NEW, "SR":FLG_SUPER_RES, "SD":sdModel, "STT":sttModel})
analysis_name = json.dumps({"0":"SD", "1":"STT"})
message = ''
# process_time = json.dumps({"starttime":starttime, "endtime":endtime, "duration":endtime-starttime})
process_time = '{"starttime":"%s", "endtime":"%s", "duration":"%s"}'%(starttime,endtime,endtime-starttime)

uob_storage.dbInsertLog(audio_id=audio_id, params=params, analysis_name=analysis_name, process_time=process_time)