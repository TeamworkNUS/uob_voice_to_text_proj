import os
import pandas as pd
import wave
import json
import subprocess
import librosa
import uob_extractmodel
from init import (
    SAMPLE_RATE,
    pretrained_model_path,
    STT_SAMPLERATE
)

def load_stt_model(stt_model='vosk', pretrained_model_path=pretrained_model_path, sr=STT_SAMPLERATE):
    if stt_model=='vosk':
        model, rec = uob_extractmodel.get_model_stt_vosk(pretrained_model_path=pretrained_model_path, sr=sr)
        return rec
    elif stt_model=='malaya-speech': 
        model = uob_extractmodel.get_model_stt_malaya_speech(pretrained_model_path=pretrained_model_path)
        return model
    # elif stt_model=='deepspeech':
    #     model = uob_extractmodel.get_model_stt_deepspeech()
    #     return model


def stt_conversion_vosk(slices_path, rec, sr = STT_SAMPLERATE):
    transcription = []

    stt = pd.DataFrame(columns=['index', 'text','slice_wav_file_name'])
    for filename in os.listdir(slices_path+"/"): #(slices_path+"/"):
        if filename.endswith(".wav"): 
            namef, namec = os.path.splitext(filename)
            namef_other, namef_index = namef.rsplit("_", 1)
            namef_index = int(namef_index)

            inputFile = slices_path+"/"+filename
            process = subprocess.Popen(['ffmpeg', '-loglevel', 'quiet', '-i',
                            # os.listdir("./audio/")[0],
                            inputFile,
                            '-ar', str(sr) , '-ac', '1', '-f', 's16le', '-'],
                            stdout=subprocess.PIPE)
            
            # wf = wave.open(slices_path+"/"+filename, "rb")   # Change when confirm
            # file = slices_path+"/"+filename
            while True:
                data = process.stdout.read(10000000)
                # data = wf.readframes(10000000)
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
            # print(transcription_text)  # For output check only, remove after confirm.

            stt.loc[len(stt)] = [namef_index, transcription_text,filename]
            transcription_text = ""
            transcription = []

           
    return stt


def stt_conversion_malaya_speech(slices_path, rec):
    stt = pd.DataFrame(columns=['index', 'text','slice_wav_file_name'])
    for filename in os.listdir(slices_path+"/"): 
        if filename.endswith(".wav"): 
            namef, namec = os.path.splitext(filename)
            namef_other, namef_index = namef.rsplit("_", 1)
            namef_index = int(namef_index)
            inputFile = slices_path+"/"+filename
            # singlish, sr = malaya_speech.load(inputFile)
            singlish, sr = librosa.load(inputFile,sr= None, mono= True)
            # greedy_decoder
            transcription = rec.greedy_decoder([singlish])
            # beam_decoder
            # transcription = rec.beam_decoder([singlish])
            transcription_text = ' '.join(transcription)
            stt.loc[len(stt)] = [namef_index, transcription_text,filename]

           
    return stt