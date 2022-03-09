# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
''' Logging
Version     Date    Change_by   Description
#00     2022-Feb-28     Zhao Tongtong   Initial version  


'''
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import numpy as np
import malaya_speech
from malaya_speech import Pipeline

import uob_extractmodel


def load_noisereduce_model(quantized = False): 
    ## List available deep model:
    # malaya_speech.noise_reduction.available_model()
    
    ## Load deep model
    model = malaya_speech.noise_reduction.deep_model(model = 'resnet-unet', quantized=quantized)
    #! Noise Reduction model trained on 44k sample rate, so make sure load the audio with 44k sample rate.
    return model

def load_noisereduce_model_local(quantized = False): 

    model = uob_extractmodel.get_model_noisereduce(model = 'resnet-unet', quantized=quantized)
    #! Noise Reduction model trained on 44k sample rate, so make sure load the audio with 44k sample rate.
    return model


def get_splitted_output(y, sr, model):
    output = model(y)
    voice = output['voice']
    noise = output['noise']
    return voice, noise


def output_voice_pipeline(y, sr, model, frame_duration_ms=15000):
    p = Pipeline()
    pipeline = (
        p.map(malaya_speech.generator.frames, 
              frame_duration_ms = frame_duration_ms,
              sample_rate = sr)
        .foreach_map(model)
        .foreach_map(lambda x: x['voice'])
        .map(np.concatenate)
    )
    
    results = p.emit(y)
    
    print(results.keys())
    # dict_keys(['frames', 'noise-reduction', '<lambda>', 'concatenate'])

    return results['concatenate']  #-->y


    
    