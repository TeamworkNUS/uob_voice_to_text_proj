import os

# pretrained_model_path = 'E:\\Python\\Python37\\Lib\\site-packages\\malaya_speech\\pretrained-model\\'
pretrained_model_path = '.\\pretrained-model\\'

AUDIO_NAME = 'Bdb001_interaction_first30s.wav' #'LE_listening_B1_A_phone_call_from_a_customer.wav' #'009NTWY_U3_CL_Shopping.wav' #'Bdb001_interaction_first60s.wav' #
AUDIO_PATH = './wav/'
AUDIO_FILE = os.path.join(AUDIO_PATH,AUDIO_NAME)
SAMPLE_RATE = 44100

sd_global_starttime = 0.0