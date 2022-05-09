# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
''' Logging
Version     Date    Change_by   Description
#00     2022-Feb-28     Zhao Tongtong   Initial version  


'''
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import soundfile as sf
import wave
import json
import malaya_speech
from pyannote.audio import Pipeline

import webrtcvad
import librosa
import struct
from typing import Union, List, Optional
from pathlib import Path
from scipy.ndimage.morphology import binary_dilation
from resemblyzer.hparams import *

import os
import numpy as np
from datetime import datetime

import uob_noisereduce, uob_speakerdiarization, uob_audiosegmentation, uob_stt, uob_speechenhancement, uob_label, uob_storage, uob_superresolution, uob_speechenhancement_new

def sd_process(y, sr, audioname, audiopath, audiofile, nr_model=None, se_model=None, sr_model=None, vad_model=None, sv_model=None, pipeline=None, chunks:bool=True, reducenoise:bool=False, speechenhance:bool=False, superresolution:bool=False, speechenhance_new:bool=False, se_model_new=None, sd_proc='pyannoteaudio'):
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
    
    
    ## Speech Enhancement
    if speechenhance == True:
        y = malaya_speech_enhance(y, sr, se_model=se_model)
        
        if chunks:
            namef, namec = os.path.splitext(audioname)
            namef_other, namef_index = namef.rsplit("_", 1)
            namef_index = int(namef_index)
            namec = namec[1:]
            audioname = '%s_%04d.%s'%(namef_other+'_se',namef_index,namec)
            sf.write(os.path.join(audiopath,audioname), y, sr) # TODO: how to save wav? delete the file after done?
        else:
            namef, namec = os.path.splitext(audioname)
            namec = namec[1:]
            audioname = '%s.%s'%(namef+'_se',namec)
            sf.write(os.path.join(audiopath,audioname), y, sr) # TODO: how to save wav? delete the file after done?
        
        ## Speech Enhancement
    if speechenhance_new == True:
        y = malaya_speech_enhance_new(y, sr, se_model_new = se_model_new)
        
        if chunks:
            namef, namec = os.path.splitext(audioname)
            namef_other, namef_index = namef.rsplit("_", 1)
            namef_index = int(namef_index)
            namec = namec[1:]
            audioname = '%s_%04d.%s'%(namef_other+'_se',namef_index,namec)
            sf.write(os.path.join(audiopath,audioname), y, sr) # TODO: how to save wav? delete the file after done?
        else:
            namef, namec = os.path.splitext(audioname)
            namec = namec[1:]
            audioname = '%s.%s'%(namef+'_se',namec)
            sf.write(os.path.join(audiopath,audioname), y, sr) # TODO: how to save wav? delete the file after done?

    ## Super Resolution
    if superresolution == True:
        #! 1.Currently we only supported 4x super resolution, if input sample rate is 16k, output will become 16k * 4.
        #! 2.We trained on 11025 for input sample rate, 44100 for output sample rate.
        reduction_factor = 4
        y_, sr_ = malaya_speech.load(os.path.join(audiopath,audioname), sr = sr // reduction_factor)
        y = malaya_super_resolution(y_, sr_, sr_model=sr_model)
        # y = malaya_super_resolution(y, sr, sr_model=sr_model)
        
        if chunks:
            namef, namec = os.path.splitext(audioname)
            namef_other, namef_index = namef.rsplit("_", 1)
            namef_index = int(namef_index)
            namec = namec[1:]
            audioname = '%s_%04d.%s'%(namef_other+'_sr',namef_index,namec)
            sf.write(os.path.join(audiopath,audioname), y, sr) # TODO: how to save wav? delete the file after done?
        else:
            namef, namec = os.path.splitext(audioname)
            namec = namec[1:]
            audioname = '%s.%s'%(namef+'_sr',namec)
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
    if sd_proc == 'malaya':
        sd_result = malaya_sd(y, sr, audioname, audiopath, vad_model, sv_model)
    elif sd_proc == 'pyannoteaudio':
        sd_result = pyannoteaudio_sd(audioname, audiopath, audiofile, pipeline)
    elif sd_proc == 'resemblyzer':
        sd_result = resemblyzer_sd(audioname, audiopath, audiofile)
    else:
        raise Exception('!!! Please input correct SD model: [pyannoteaudio, malaya, resemblyzer]')
    
    
    # ## Delete reduced noise temp file
    # if os.path(audiopath+'/noisereduced.wav'):
    #     os.remove(audiopath+'/noisereduced.wav')

    ## Add chunk filename
    if chunks == True:
        for row in sd_result:
            row.append(audioname)
    else:
        for row in sd_result:
            row.append("")


    return sd_result

def stt_process(sttModel, slices_path, rec, sr):
    if sttModel == 'vosk':
        stt_result = uob_stt.stt_conversion_vosk(slices_path, rec, sr)
    elif sttModel == 'malaya-speech':
        stt_result = uob_stt.stt_conversion_malaya_speech(slices_path, rec)
    return stt_result

def speaker_label_func(transactionDf, pretrained_model_path, checklist_path):
    label_result = uob_label.speaker_label_func(transactionDf, pretrained_model_path, checklist_path)
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


def malaya_speech_enhance(y, sr, se_model):
    ### * Enhance the Speech
    speechenhanced_audio =uob_speechenhancement.get_se_output(y, sr, se_model)
    y = speechenhanced_audio
    return y
    
def malaya_speech_enhance_new(y, sr, se_model_new):
    ### * Enhance the Speech
    speechenhanced_audio =uob_speechenhancement_new.get_se_output(y, sr, se_model_new)
    y = speechenhanced_audio
    return y
    
def malaya_super_resolution(y, sr, sr_model):
    ### * Convert to Super Resolution Audio
    superresolution_audio =uob_superresolution.get_sr_output(y, sr, sr_model)
    y = superresolution_audio
    return y    


def malaya_sd(y, sr, audioname, audiopath, vad_model, sv_model):  
    '''
    INPUT
        y: waveform of audio
        sr: sample rate
        audiopath: the path for saving sd result in csv
        sv_model: speaker vector model
    OUTPUT
        diarization_result-->List[(Tuple, str)]: timestamp, duration, cluster label
    '''
    ### * Speaker Diarization
    ## Load SD model + VAD
    # model_speakernet, model_vggvox2 = uob_speakerdiarization.load_speaker_vector()
    grouped_vad = uob_speakerdiarization.load_vad(y, sr, vad_model = vad_model, frame_duration_ms=30, threshold_to_stop=0.5)
    speaker_vector = sv_model #model_speakernet

    
    ## Diarization
    # ?: choose a SD function below, comment others
    # result_sd = uob_speakerdiarization.speaker_similarity(speaker_vector, grouped_vad)
    # result_sd = uob_speakerdiarization.affinity_propagation(speaker_vector, grouped_vad)
    result_sd = uob_speakerdiarization.spectral_clustering(speaker_vector, grouped_vad, min_clusters=2, max_clusters=2) #!p_percentile issue
    # result_sd = uob_speakerdiarization.n_speakers_clustering(speaker_vector, grouped_vad, n_speakers=2, model='kmeans') #['spectralcluster','kmeans']
    # result_sd = uob_speakerdiarization.speaker_change_detection(speaker_vector, grouped_vad, y, sr,frame_duration_ms=500, 
    #                                                           min_clusters = 2, max_clusters = 2) #!p_percentile issue
    
    ## Visualization #TODO: to comment for production
    uob_speakerdiarization.visualization_sd(y, grouped_vad, sr, result_sd)
    
    
    ## Get timestamp
    grouped_result = uob_speakerdiarization.get_timestamp(result_diarization = result_sd)
    diarization_result = grouped_result
    
    ## Print & Save result to csv
    result_timestamps = []
    result_timestamp = ['index','starttime','endtime','duration','speaker_label']
    result_timestamps.append(result_timestamp)
    print('Index\tStart\tEnd\tDuration\tSpeaker')
    for i in grouped_result:
        index = grouped_result.index(i) + 1
        end = i[0].timestamp+i[0].duration
        print(str(index)+'\t'+str(i[0].timestamp)+'\t'+str(end)+'\t'+str(i[0].duration)+'\t'+str(i[1]))

        result_timestamp = [int(index), float(i[0].timestamp), float(end), float(i[0].duration),str(i[1])]
        result_timestamps.append(result_timestamp)

    # # Remove "Save" after integration
    # namef, namec = os.path.splitext(audioname)
    # namec = namec[1:]
    # save_name = '%s_%s.%s'%(namef, datetime.now().strftime('%y%m%d-%H%M%S'), 'csv')
    # np.savetxt(os.path.join(audiopath,save_name), result_timestamps,
    #            delimiter=',', fmt='% s')
    
    # return diarization_result
    return result_timestamps



def pyannoteaudio_sd(audioname, audiopath, audiofile, pa_pipeline):
    '''
    INPUT
        audioname: name.wav
        audiofile: path/name.wav
        pa_pipeline: pyannote.audio pretrained pipeline for handling speaker diarization
    OUTPUT
        diarization_result: speaker diarization result, including timestamp, duration and cluster label
    '''
    ## Process
    diarization_result = uob_speakerdiarization.pyannoteaudio_speaker_diarization(audiofile, pa_pipeline)
    
    ## Print & Save time period & speaker label
    result_timestamps = []
    result_timestamp = ['index','starttime','endtime','duration','speaker_label']
    result_timestamps.append(result_timestamp)
    index = 0
    print('Index\tStart\tEnd\tDuration\tSpeaker')
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        # speaker speaks between turn.start and turn.end
        index += 1
        print(index, turn.start, turn.end, turn.end-turn.start, speaker)
        result_timestamp = [int(index), float(turn.start), float(turn.end), float(turn.end-turn.start), str(speaker)]
        result_timestamps.append(result_timestamp)

    # # Remove "Save" after integration
    # namef, namec = os.path.splitext(audioname)
    # namec = namec[1:]
    # save_name = '%s_%s.%s'%(namef, datetime.now().strftime('%y%m%d-%H%M%S'), 'csv')
    # np.savetxt(os.path.join(audiopath,save_name), result_timestamps,
    #            delimiter=',', fmt='% s')
    
    # return diarization_result
    return result_timestamps



def resemblyzer_sd(audioname, audiopath, audiofile):
    ## Process (output->pd.Dataframe)
    from uob_extractmodel import resemblyzer_VoiceEncoder
    
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
    # index = 0
    for row in start_end_list:
        # print(', '.join(row))
        index=row[0]
        start=row[1]
        end=row[2]
        # dur=float(row[3])
        # label=row[4]
        
        # index += 1
        namef, namec = os.path.splitext(audioname)
        namec = namec[1:]
        part_name = '%s_%s.%s'%(namef, str(index), namec)
        part_file = part_path + '/' + part_name
        
        uob_audiosegmentation.get_second_part_wav(audiofile, start, end, part_file)