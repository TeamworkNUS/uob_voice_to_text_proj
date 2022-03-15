import pandas as pd
import os
# Load VOSK model
from vosk import Model, KaldiRecognizer, SetLogLevel
print('*'*30, 'Loading Model')
sample_rate=16000
model = Model("model")
rec = KaldiRecognizer(model, sample_rate)

# STT Start
import wave
import json

print('*'*30, 'STT Start')
transcription = []

stt = pd.DataFrame(columns=['index', 'text'])
for filename in os.listdir("./audio/"): #(slices_path+"/"):
    if filename.endswith(".wav"): 
        namef, namec = os.path.splitext(filename)
        namef_other, namef_index = namef.rsplit("_", 1)
        namef_index = int(namef_index)
        
        wf = wave.open("./audio/"+filename, "rb")   # Change when confirm
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

        # text = subprocess.check_output(["python","ffmpeg.py",filename],cwd= './stt').decode("utf-8")
        # textNew = re.findall('"([^"]*)"', text)
        # index = namef_index  # index = re.findall(r'\d+',filename)[-1]
        # stt.loc[len(stt)] = [index, textNew[1]]
        
# stt['index'] = stt['index'].astype(int)
print(stt)
stt.to_csv('output.csv')
# print(final_sd_result)
# final = pd.merge(left = final_sd_result, right = stt, on="index",how='left')
# final.to_csv('output.csv')
