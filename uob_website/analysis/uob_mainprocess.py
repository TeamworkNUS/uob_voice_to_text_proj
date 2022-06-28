# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
''' Logging
Version     Date    Change_by   Description
#00     2022-Feb-28     Zhao Tongtong   Initial version  


'''
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import os
import re
import numpy as np
import pandas as pd
import shutil
import zipfile, rarfile, py7zr
import soundfile as sf
import json
import librosa
import struct
from typing import Union, List, Optional
from pathlib import Path
from scipy.ndimage.morphology import binary_dilation


from analysis.resemblyzer.hparams import *
from analysis import webrtcvad

from analysis import uob_noisereduce, uob_audiosegmentation, uob_speakerdiarization, uob_stt, uob_label, uob_storage, uob_utils, uob_personalInfo
from analysis.uob_init import(
    stt_replace_template,
    profanity_list
)


def process_upload_file(upload_filename, upload_filepath):
    '''
    RETURN
        upload_file_count, upload_filetype_general, filename, filepath
    '''
    upload_file = os.path.join(upload_filepath, upload_filename)
    upload_purename = os.path.splitext(upload_filename)[0]
    
    upload_extension, upload_mime = uob_utils.check_file_type(upload_filename, upload_filepath)
    if upload_extension in ['wav','mp3'] and upload_mime.startswith('audio/'):
        to_audioname = os.path.splitext(upload_filename)[0]+'.wav'
        to_audiopath = upload_filepath
        if upload_extension != 'wav' or upload_mime != 'audio/x-wav':
            uob_utils.audio2wav(from_audioname=upload_filename, from_audiopath=upload_filepath, to_audioname=to_audioname, to_audiopath=to_audiopath)
        return 1, 'audio', to_audioname, to_audiopath
    
    
    elif upload_extension == 'zip' and upload_mime == 'application/zip':
        zip_folder = os.path.join(upload_filepath, upload_purename).replace('\\','/')
        if not os.path.exists(zip_folder): 
            os.mkdir(zip_folder)
            
        zip = zipfile.ZipFile(upload_file)
        zip.extractall(path=zip_folder)
        zip.close()
        ## Delete irregular files
        namelist_toDel = [name for name in os.listdir(zip_folder) if not os.path.isfile(os.path.join(zip_folder,name))]
        for name in namelist_toDel:
            shutil.rmtree(os.path.join(zip_folder, name))

        ## Process regular file
        namelist = [name for name in os.listdir(zip_folder) if os.path.isfile(os.path.join(zip_folder,name))]
        for name in namelist:
            # zip.extract(name, zip_folder)  #zip.namelist()
            unzip_extension, unzip_mime = uob_utils.check_file_type(name, zip_folder)
            to_audioname = os.path.splitext(name)[0]+'.wav'
            to_audiopath = zip_folder
            print(name, unzip_extension, unzip_mime)
            if unzip_extension == None or unzip_mime == None:
                os.remove(os.path.join(zip_folder, name))
                print('Cannot get file extension or mime! The file will be removed.')
                continue
            
            if (unzip_extension != 'wav' or unzip_mime != 'audio/x-wav') and unzip_mime.startswith('audio'):
                uob_utils.audio2wav(from_audioname=name, from_audiopath=zip_folder, to_audioname=to_audioname, to_audiopath=to_audiopath)
                os.remove(os.path.join(zip_folder, name))
            elif not unzip_mime.startswith('audio'):
                os.remove(os.path.join(zip_folder, name))
                print('remove non-audio file:',name)
        return len(os.listdir(zip_folder)), 'compressed', '*', zip_folder
        
        
    elif upload_extension == 'rar' and upload_mime == 'application/x-rar-compressed':
        zip_folder = os.path.join(upload_filepath, upload_purename).replace('\\','/')
        if not os.path.exists(zip_folder): 
            os.mkdir(zip_folder)
        
        rar = rarfile.RarFile(upload_file)
        rar.extractall(path=zip_folder)
        rar.close()
        ## Delete irregular files
        namelist_toDel = [name for name in os.listdir(zip_folder) if not os.path.isfile(os.path.join(zip_folder,name))]
        for name in namelist_toDel:
            shutil.rmtree(os.path.join(zip_folder, name))
        
        ## Process regular file
        namelist = [name for name in os.listdir(zip_folder) if os.path.isfile(os.path.join(zip_folder,name))]
        for name in namelist:
            # rar.extract(name, zip_folder)  # rar.namelist()
            unzip_extension, unzip_mime = uob_utils.check_file_type(name, zip_folder)
            to_audioname = os.path.splitext(name)[0]+'.wav'
            to_audiopath = zip_folder
            print(name, unzip_extension, unzip_mime)
            if unzip_extension == None or unzip_mime == None:
                os.remove(os.path.join(zip_folder, name))
                print('Cannot get file extension or mime! The file will be removed.')
                continue
            
            if (unzip_extension != 'wav' or unzip_mime != 'audio/x-wav') and unzip_mime.startswith('audio'):
                uob_utils.audio2wav(from_audioname=name, from_audiopath=zip_folder, to_audioname=to_audioname, to_audiopath=to_audiopath)
                os.remove(os.path.join(zip_folder, name))
            elif not unzip_mime.startswith('audio'):
                os.remove(os.path.join(zip_folder, name))
                print('remove non-audio file:',name)
        return len(os.listdir(zip_folder)), 'compressed', '*', zip_folder
    
    
    elif upload_extension == '7z' and upload_mime == 'application/x-7z-compressed':
        zip_folder = os.path.join(upload_filepath, upload_purename).replace('\\','/')
        if not os.path.exists(zip_folder): 
            os.mkdir(zip_folder)
        
        sevenZip = py7zr.SevenZipFile(upload_file)
        sevenZip.extractall(path=zip_folder)
        sevenZip.close()
        ## Delete irregular files
        namelist_toDel = [name for name in os.listdir(zip_folder) if not os.path.isfile(os.path.join(zip_folder,name))]
        for name in namelist_toDel:
            shutil.rmtree(os.path.join(zip_folder, name))
        
        ## Process regular file
        namelist = [name for name in os.listdir(zip_folder) if os.path.isfile(os.path.join(zip_folder,name))]
        for name in namelist:        
            unzip_extension, unzip_mime = uob_utils.check_file_type(name, zip_folder)
            to_audioname = os.path.splitext(name)[0]+'.wav'
            to_audiopath = zip_folder
            print(name, unzip_extension, unzip_mime)
            if unzip_extension == None or unzip_mime == None:
                os.remove(os.path.join(zip_folder, name))
                print('Cannot get file extension or mime! The file will be removed.')
                continue
            
            if (unzip_extension != 'wav' or unzip_mime != 'audio/x-wav') and unzip_mime.startswith('audio'):
                uob_utils.audio2wav(from_audioname=name, from_audiopath=zip_folder, to_audioname=to_audioname, to_audiopath=to_audiopath)
                os.remove(os.path.join(zip_folder, name))
            elif not unzip_mime.startswith('audio'):
                os.remove(os.path.join(zip_folder, name))
        return len(os.listdir(zip_folder)), 'compressed', '*', zip_folder
    
    
    else:
        try:
            os.remove(upload_file)
        except Exception as e:
            print('Remove file', upload_file, 'failed because of Exception', type(e), e)
        
        raise ValueError(
            "Upload file type is not supported. Please upload pure audio file (wav, mp3) or compressed file (zip, rar)"
        )


def sd_process(y, sr, audioname, audiopath, audiofile, nr_model=None, chunks:bool=True, reducenoise:bool=False, sd_proc='resemblyzer'):
    ## Reduce noise
    if reducenoise == True:
        ## load nr models
        # nr_model, nr_quantized_model = uob_noisereduce.load_noisereduce_model(modelname='resnet-unet')
        # start to process
        y = malaya_reduce_noise(y, sr, nr_model=nr_model)
        
        if chunks:
            namef, namec = os.path.splitext(audioname)
            namef_other, namef_index = namef.rsplit("_", 1)
            namef_index = int(namef_index)
            namec = namec[1:]
            audioname = '%s_%04d.%s'%(namef_other+'_nr',namef_index,namec)
            sf.write(os.path.join(audiopath,audioname), y, sr) # TODO: how to save wav? delete the file after done?
        else:
            namef, namec = os.path.splitext(audioname)
            namec = namec[1:]
            audioname = '%s.%s'%(namef+'_nr',namec)
            sf.write(os.path.join(audiopath,audioname), y, sr) # TODO: how to save wav? delete the file after done?
    
    
    if chunks:
        audiopath = audiopath + '_processed'
        if not os.path.exists(audiopath): 
            os.mkdir(audiopath)
        sf.write(os.path.join(audiopath,audioname), y, sr)
    else:
        sf.write(os.path.join(audiopath,audioname), y, sr)
    print('audiopath: ',audiopath)
    
    ## Speaker Diarization
    if sd_proc == 'resemblyzer':
        sd_result = resemblyzer_sd(audioname, audiopath, audiofile)
    else:
        raise Exception('!!! Please input correct SD model: [pyannoteaudio, malaya, resemblyzer]')
    

    ## Add chunk filename
    if chunks == True:
        for row in sd_result:
            row.append(audioname)
    else:
        for row in sd_result:
            row.append("")


    return sd_result, audioname, audiopath


def stt_process(sttModel, slices_path, rec, **kwargs):

    if sttModel == 'malaya-wav2vec2':
        model = kwargs.get('wav2vec2_model')
        stt_result = uob_stt.stt_conversion_malaya_wav2vec2(slices_path, rec, model)
    elif sttModel == 'malaya-speech':
        stt_result = uob_stt.stt_conversion_malaya_speech(slices_path, rec)


    ### Extra Steps
    # 1. word replacement: pos_tagger & direct replace
    # 2. profanity handling: masking
    
    ## POS Tagger manually replace words
    if os.path.exists(path=stt_replace_template):
        print('Go to pos tagger replace...')
        template = pd.read_csv(stt_replace_template)
        text_replace = []
        for index, row in stt_result.iterrows():
            l = str(row["text"])
            l_r = uob_stt.pos_replace(l = l, template = template)
            text_replace.append(str(l_r))
        stt_result['text'] = text_replace
    
    ## Profanity handling
    if os.path.exists(path=profanity_list):
        print('Go to profanity handling...')
        swear_corpus=[]
        with open(profanity_list,"r") as f:
            for word in f.readlines():
                word = word.strip()
                if len(word) !=0:
                    swear_corpus.append(' '+word+' ')
        p_pattern = re.compile("|".join(swear_corpus))
        text_replace = []
        for index, row in stt_result.iterrows():
            l = str(row["text"])
            l_p = uob_stt.profanity_handling(l = l, p_pattern = p_pattern)
            text_replace.append(str(l_p))
        stt_result['text'] = text_replace
        
    return stt_result


def speaker_label_process(transactionDf, label_model, checklist_path):
    label_result = uob_label.speaker_label_func(transactionDf, label_model, checklist_path)
    return label_result


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                       Functions                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def malaya_reduce_noise(y, sr, nr_model):
    '''
    INPUT
        y: waveform of audio
        sr: sample rate
        audiopath: the path for saving sd result in csv
        nr_model: noise reduce model
    OUTPUT
        y: waveform of audio whose noise has been reduced
    '''
    ### * Reduce Noise
    # nr_model, nr_quantized_model = uob_noisereduce.load_noisereduce_model(modelname='resnet-unet')  
    noisereduced_audio = uob_noisereduce.output_voice_pipeline(y, sr, nr_model, frame_duration_ms=15000)
    y = noisereduced_audio
    return y



def resemblyzer_sd(audioname, audiopath, audiofile):
    ## Process (output->pd.Dataframe)
    from analysis.uob_extractmodel import resemblyzer_VoiceEncoder
    
    encoder = resemblyzer_VoiceEncoder("cpu")
    int16_max = (2 ** 15) - 1

    def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray], source_sr: Optional[int] = None):

        wav, source_sr = librosa.load(fpath_or_wav)

        # Resample the wav
        if source_sr is not None:
            wav = librosa.resample(wav, source_sr, sampling_rate)

        # Apply the preprocessing: normalize volume and shorten long silences
        wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
        # wav = trim_long_silences(wav)

        return wav

    def wav_to_mel_spectrogram(wav):

        frames = librosa.feature.melspectrogram(
            wav,
            sampling_rate,
            n_fft=int(sampling_rate * mel_window_length / 1000),
            hop_length=int(sampling_rate * mel_window_step / 1000),
            n_mels=mel_n_channels
        )
        return frames.astype(np.float32).T

    def trim_long_silences(wav):

        # Compute the voice detection window size
        samples_per_window = (vad_window_length * sampling_rate) // 1000

        # Trim the end of the audio to have a multiple of the window size
        wav = wav[:len(wav) - (len(wav) % samples_per_window)]

        # Convert the float waveform to 16-bit mono PCM
        pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))

        # Perform voice activation detection
        voice_flags = []
        vad = webrtcvad.Vad(mode=3)
        for window_start in range(0, len(wav), samples_per_window):
            window_end = window_start + samples_per_window
            voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                             sample_rate=sampling_rate))
        voice_flags = np.array(voice_flags)

        # Smooth the voice detection with a moving average
        def moving_average(array, width):
            array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
            ret = np.cumsum(array_padded, dtype=float)
            ret[width:] = ret[width:] - ret[:-width]
            return ret[width - 1:] / width

        audio_mask = moving_average(voice_flags, vad_moving_average_width)
        audio_mask = np.round(audio_mask).astype(np.bool)

        # Dilate the voiced regions
        audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
        audio_mask = np.repeat(audio_mask, samples_per_window)

        return wav[audio_mask == True]

    def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
        if increase_only and decrease_only:
            raise ValueError("Both increase only and decrease only are set")
        rms = np.sqrt(np.mean((wav * int16_max) ** 2))
        wave_dBFS = 20 * np.log10(rms / int16_max)
        dBFS_change = target_dBFS - wave_dBFS
        if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
            return wav
        return wav * (10 ** (dBFS_change / 20))

    ####### Calling the Preprocessing and Embedding function

    wav = preprocess_wav(audiofile)
    _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)
    print(cont_embeds.shape)
    
    ####### Speaker Diarization
    df = uob_speakerdiarization.resemblyzer_speaker_diarization(cont_embeds, wav_splits)
    
    ####### Modify Output df Format
    df.insert(0, 'index', df.index+1, allow_duplicates=False)
    df['duration'] = df['endtime']-df['starttime']
    
    df['starttime'] = df['starttime'].astype(float)
    df['endtime'] = df['endtime'].astype(float)
    df['duration'] = df['duration'].astype(float)
    df['speaker_label'] = df['speaker_label'].astype(str)

    df = df[['index','starttime','endtime','duration','speaker_label']]

    diarization_result = df.values.tolist()
    
    ## Print & Save time period & speaker label
    result_timestamps = []
    result_timestamp = ['index','starttime','endtime','duration','speaker_label']
    result_timestamps.append(result_timestamp)
    print('Index\tStart\tEnd\tDuration\tSpeaker')

    for row in diarization_result:
        # index += 1
        print(row[0], row[1], row[2], row[3], row[4])
        result_timestamp = [int(row[0]), float(row[1]), float(row[2]), float(row[3]), str(row[4])]
        result_timestamps.append(result_timestamp)
    
    return result_timestamps



def cut_audio_by_timestamps(start_end_list:list, audioname, audiofile, part_path):
    for row in start_end_list:
        index=row[0]
        start=row[1]
        end=row[2]
        # dur=float(row[3])
        # label=row[4]
        
        namef, namec = os.path.splitext(audioname)
        namec = namec[1:]
        part_name = '%s_%s.%s'%(namef, str(index), namec)
        part_file = part_path + '/' + part_name
        
        uob_audiosegmentation.get_second_part_wav(audiofile, start, end, part_file)