import os
import re
import pandas as pd
import wave
import json
import subprocess
import librosa
import uob_extractmodel
from nltk import word_tokenize, pos_tag
import pandas as pd
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
            y, sr = librosa.load(inputFile,sr= None, mono= True)
            if librosa.get_duration(y=y, sr=sr) > 0.5:
                singlish, sr = librosa.load(inputFile,sr= 16000, mono= True)
                # greedy_decoder
                transcription = rec.greedy_decoder([singlish])
                # beam_decoder
                # transcription = rec.beam_decoder([singlish])
                transcription_text = ' '.join(transcription)
                # transcription_text = transcription_text if transcription_text.strip() != 'ya' else ''
                if transcription_text.strip() != 'ya':
                    stt.loc[len(stt)] = [namef_index, transcription_text,filename]
                else:
                    stt.loc[len(stt)] = [namef_index, '',filename]
            else:
                stt.loc[len(stt)] = [namef_index, '',filename]
                
    return stt

def pos_replace(l, template):
    '''
    l: the stt sentence
    template: DataFrame, read as csv from stt_replace_template.txt
    pos_tag reference: https://universaldependencies.org/u/pos/all.html#al-u-pos/
    '''
    flag_change = ""
    words = word_tokenize(l)
    # l_r = TreebankWordDetokenizer().detokenize(words)
    l_r = " ".join(words)
    for i in range(len(template)):
        row = template.iloc[i]
        if str(row['replace'])[-1]==';':
            row['replace'] = str(row['replace'])[:-1]
        pos_before = str(row['pos_before']).split(';')
        pos_after = str(row['pos_after']).split(';')
        replace_list = str(row['replace']).split(';')
        flag_direct_replace = str(row['flag_direct_replace'])
        key = word_tokenize(row['key'])           
        for p in replace_list:
            p_w = word_tokenize(p)
            # p_s = TreebankWordDetokenizer().detokenize(p_w)
            p_s = " ".join(p_w)
            if p_s in l_r:
                # p_w = word_tokenize(p)
                lenth_p = len(p_w)
                words = word_tokenize(l_r)
                words_pos = pos_tag(words, tagset='universal')
                lenth_w = len(words)
                # r_len = lenth_w-lenth_p+1
                if flag_direct_replace.lower() == "x":
                    i = 0
                    # for i in range(lenth_w-lenth_p+1):
                    while i < lenth_w-lenth_p+1:
                        # print(i)
                        if words[i:i+lenth_p]==p_w:
                            print('directly replace', p, 'with',row[0])
                            words[i:i+lenth_p] = key
                            words_pos = pos_tag(words, tagset='universal')
                            lenth_w = len(words)
                            # r_len = lenth_w-lenth_p+1
                            flag_change = "x"
                        i += 1
                else:
                    # for i in range(lenth_w-lenth_p+1):
                    i = 0
                    while i < lenth_w-lenth_p+1:
                        # print(i)
                        if words[i:i+lenth_p]==p_w:
                            if i-1>=0 and words_pos[i-1][1] in pos_before:
                                print('the pos of the word before', p, 'is',words_pos[i-1][1])
                                words[i:i+lenth_p] = key
                                words_pos = pos_tag(words, tagset='universal')
                                lenth_w = len(words)
                                flag_change = "x"

                            elif i+lenth_p<len(words) and words_pos[i+lenth_p][1] in pos_after:
                                print('the pos of the word after', p, 'is',words_pos[i+lenth_p][1])
                                words[i:i+lenth_p] = key
                                words_pos = pos_tag(words, tagset='universal')
                                lenth_w = len(words)
                                flag_change = "x"
                        i += 1
                        
                if flag_change == "x":
                    print(l_r)
                    l_r = " ".join(words)
                    # l_r = TreebankWordDetokenizer().detokenize(words)
                    print(l_r)
                    flag_change = ""

    return l_r


def profanity_handling(l, p_pattern):
    '''
    l: the stt sentence
    '''
    p_once = re.sub(p_pattern, " *** ", ' '+l+' ')
    p_twice = re.sub(p_pattern, " *** ", p_once)
    l_p = p_twice.strip(' ')
        
    return l_p