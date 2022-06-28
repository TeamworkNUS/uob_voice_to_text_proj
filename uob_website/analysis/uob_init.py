import logging
import os

pretrained_model_path = 'D:/EBAC/Internship/UOB/Projects/pretrained-model/'

FLG_REDUCE_NOISE:bool = True

label_stop_words = 'nan'
label_checklist = 'checklist.txt'

segmentation_threshold = 600  #if audio_duration > segmentation_threshold(seconds), then segment

sttModel = 'malaya-speech' # ? ['malaya-speech', 'malaya-wav2vec2']
sdModel = 'resemblyzer' # ? [resemblyzer]

stt_replace_template = './analysis/utils/stt_replace_template.csv'
profanity_list = './analysis/utils/swear_words_list.txt'
kyc_product_list = './analysis/utils/items_verify_checklist.csv'
userguide_path = './analysis/utils/Voice_To_Text_Analysis_Web_UserGuide.pptx'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                              Initialization  Processes                              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

### Switch off GPU usage. To turn on, simply comment below statement.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"