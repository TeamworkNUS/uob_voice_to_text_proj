import logging
import os

pretrained_model_path = 'D:/EBAC/Internship/UOB/Projects/pretrained-model/'

SAMPLE_RATE = 44100
STT_SAMPLERATE = 16000
FLG_REDUCE_NOISE:bool = True
FLG_SPEECH_ENHANCE:bool = False
FLG_SPEECH_ENHANCE_NEW:bool = False
FLG_SUPER_RES:bool = False

label_stop_words = 'nan'
label_checklist = 'checklist.txt'

segmentation_threshold = 1800  #if audio_duration > segmentation_threshold(seconds), then segment
sd_global_starttime = 0.0
sttModel = 'vosk' # ? ['malaya-speech', 'vosk']
sdModel = 'resemblyzer' # ? [pyannoteaudio, malaya, resemblyzer]

flg_slice_orig = False
stt_replace_template = './analysis/utils/stt_replace_template.csv'
profanity_list = './analysis/utils/swear_words_list.txt'

userguide_path = './analysis/utils/Voice_To_Text_Analysis_Web_UserGuide.pptx'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                              Initialization  Processes                              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# * Audo Convert FLG_SPEECH_ENHANCE for better effect
if FLG_REDUCE_NOISE == False and FLG_SPEECH_ENHANCE==True:
    logging.warning('Auto turn off Speech Enhancement for better effect since Noise Reduction is off.')
    FLG_SPEECH_ENHANCE = False

# * Audo Convert FLG_SPEECH_ENHANCE for better effect
if FLG_SPEECH_ENHANCE == True and FLG_SPEECH_ENHANCE_NEW==True:
    logging.warning('Only keep Speech_Enhancement_New on if both SE are set True')
    FLG_SPEECH_ENHANCE = False