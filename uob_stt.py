import os
import pandas as pd
import wave
import json

import uob_extractmodel
from init import (
    SAMPLE_RATE,
    pretrained_model_path
)

def load_stt_model(stt_model='vosk', pretrained_model_path=pretrained_model_path, sr=SAMPLE_RATE):
    if stt_model=='vosk':
        model, rec = uob_extractmodel.get_model_stt_vosk(pretrained_model_path=pretrained_model_path, sr=sr)
        return rec


def stt_conversion_vosk(slices_path, rec):
    transcription = []

    stt = pd.DataFrame(columns=['index', 'text'])
    for filename in os.listdir(slices_path+"/"): #(slices_path+"/"):
        if filename.endswith(".wav"): 
            namef, namec = os.path.splitext(filename)
            namef_other, namef_index = namef.rsplit("_", 1)
            namef_index = int(namef_index)
            
            wf = wave.open(slices_path+"/"+filename, "rb")   # Change when confirm
            # file = slices_path+"/"+filename
            while True:
                data = wf.readframes(10000000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    # Convert json output to dict
                    result_dict = json.loads(rec.Result())
                    # Extract text values and append them to transcription list
                    transcription.append(result_dict.get("text", ""))

            # Get final bits of audio and flush the pipeline
            final_result = json.loads(rec.FinalResult())
            transcription.append(final_result.get("text", ""))

            # merge or join all list elements to one big string
            transcription_text = ' '.join(transcription)
            print(transcription_text)  # For output check only, remove after confirm.

            stt.loc[len(stt)] = [namef_index, transcription_text]
            transcription_text = ""
            transcription = []

           
    # print(stt)
    # stt.to_csv('output.csv')
    return stt