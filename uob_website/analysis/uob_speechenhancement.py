import malaya_speech
import numpy as np
from malaya_speech import Pipeline

from analysis import uob_extractmodel


def load_speechenhancement_model(model='unet', quantized = False): 
    ## List available deep model:
    # malaya_speech.speech_enhancement.available_deep_enhance()
    #? model=['unet','resnet-unet']
    
    ## Load deep model
    model = malaya_speech.speech_enhancement.deep_enhance(model = model, quantized = quantized)
    # !Deep Enhance model trained on 22k sample rate, so make sure load the audio with 22k sample rate.
    return model

def load_speechenhancement_model_local(model = 'unet', quantized = False): 
    model = uob_extractmodel.get_model_speechenhancement(model = model, quantized = quantized)
    return model


def get_se_output(y, sr, model):
    # Speech Enhancement model only accept 1 audio for single feed-forward, output 'logits'
    output = model.predict(y)
    return output

