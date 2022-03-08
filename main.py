# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
''' Logging
Version     Date    Change_by   Description
#00     2022-Feb-28     Zhao Tongtong   Initial version  


'''
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


import os
from datetime import datetime

# import filetype # to check file type

import malaya_speech
from pyannote.audio import Pipeline as pa_Pipeline
from pydub import AudioSegment


import uob_audiosegmentation, uob_noisereduce, uob_speakerdiarization, uob_mainprocess


# TODO: get uploaded file name from Front-End




#### TODO: Convert to .wav / Resampling
#### * Check if an Audio file
# kind = filetype.guess('Bdb001_interaction_first60s.wav')
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



#### * Declare variables
AUDIO_NAME = 'Bdb001_interaction_first60s.wav' #'Bdb001_interaction_first60s.wav' #'The-Singaporean-White-Boy.wav'
AUDIO_PATH = './'
AUDIO_FILE = os.path.join(AUDIO_PATH,AUDIO_NAME)
SAMPLE_RATE = 44100
num_audio_files = 1

starttime = datetime.now()




#### * Load audio file
y, sr = malaya_speech.load(AUDIO_FILE, SAMPLE_RATE)
audio_duration = len(y) / sr
print('*' * 30)
print('length:',len(y), 'sample rate:', sr, 'duration(s):',audio_duration)
print('*' * 30)

###choose your model here
sd_proc='pyannoteaudio' # [pyannoteaudio, malaya]

#### * Load models
## Noise reduce models
nr_model, nr_quantized_model = uob_noisereduce.load_noisereduce_model(modelname='resnet-unet')
## Load malaya speaker vector model 
sv_model_speakernet, sv_model_vggvox2 = uob_speakerdiarization.load_speaker_vector()
## Load pyannote.audio pipeline for sd
if if sd_proc == 'malaya':
    pa_pipeline = None
else:
    pa_pipeline = pa_Pipeline.from_pretrained('pyannote/speaker-diarization') 
#!!specturalcluster package needs to be updated.
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


#### * Process
# chunksfolder = 'chunks_The-S_20220228_162841'   # * for test
if chunksfolder != '':
    for filename in os.listdir(chunksfolder+"/"):
        if filename.endswith(".wav"): 
            # print(os.path.join(chunksfolder, filename))
            ### * Load chunk file
            file = os.path.join(chunksfolder, filename)
            y, sr = malaya_speech.load(file, SAMPLE_RATE)
                       
            ### * Process: reduce noise + vad + scd + ovl + sd
            sd_result = uob_mainprocess.process(y, sr, 
                                                audioname=filename,
                                                audiopath=chunksfolder,
                                                audiofile=file,
                                                nr_model=nr_model,   # ?: [nr_model, nr_quantized_model]
                                                sv_model=sv_model_speakernet,    # ?: sv_model_speakernet, sv_model_vggvox2
                                                pipeline=pa_pipeline,
                                                reducenoise=False,
                                                sd_proc=sd_proc)  
            
            
            # TODO: ....................... STT .........................
            
            # TODO: .......concatenation if breakdown into chunks........
    


else:
    ### * Load single file
    y, sr = malaya_speech.load(AUDIO_FILE, SAMPLE_RATE)

    ### * Process: reduce noise + vad + scd + ovl + sd
    sd_result = uob_mainprocess.process(y, sr, 
                                        audioname=AUDIO_NAME,
                                        audiopath=AUDIO_PATH,
                                        audiofile=AUDIO_FILE,
                                        nr_model=nr_model,   # ?: [nr_model, nr_quantized_model]
                                        sv_model=sv_model_speakernet,    # ?: sv_model_speakernet, sv_model_vggvox2
                                        pipeline=pa_pipeline,
                                        reducenoise=False, 
                                        sd_proc=sd_proc)  
    
    
    
    # TODO: ....................... STT .........................








endtime = datetime.now()

print('*' * 30,'\n  Finished!!',)
print('start from:', starttime) 
print('end at:', endtime) 
print('duration: ', endtime-starttime)



