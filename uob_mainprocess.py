# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
''' Logging
Version     Date    Change_by   Description
#00     2022-Feb-28     Zhao Tongtong   Initial version  


'''
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

from datetime import datetime
import soundfile as sf
import malaya_speech
from pyannote.audio import Pipeline

import os
import numpy as np

import uob_noisereduce, uob_speakerdiarization

def process(y, sr, audioname, audiopath, audiofile, nr_model=None, vad_model=None, sv_model=None, pipeline=None, chunks:bool=True, reducenoise:bool=False, sd_proc='pyannoteaudio'):
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
    
    
    ## Speaker Diarization
    if sd_proc == 'malaya':
        sd_result = malaya_sd(y, sr, audioname, audiopath, vad_model, sv_model)
    elif sd_proc == 'pyannoteaudio':
        sd_result = pyannoteaudio_sd(audioname, audiopath, audiofile, pipeline)
    
    
    # ## Delete reduced noise temp file
    # if os.path(audiopath+'/noisereduced.wav'):
    #     os.remove(audiopath+'/noisereduced.wav')
    

    return sd_result





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
    grouped_vad = uob_speakerdiarization.load_vad(y, sr, vad_model = vad_model, frame_duration_ms=30, threshold_to_stop=0.3)
    speaker_vector = sv_model #model_speakernet

    
    ## Diarization
    # ?: choose a SD function below, comment others
    # result_sd = uob_speakerdiarization.speaker_similarity(speaker_vector, grouped_vad)
    # result_sd = uob_speakerdiarization.affinity_propagation(speaker_vector, grouped_vad)
    # result_sd = uob_speakerdiarization.spectral_clustering(speaker_vector, grouped_vad, min_clusters=2, max_clusters=3) #!p_percentile issue
    result_sd = uob_speakerdiarization.n_speakers_clustering(speaker_vector, grouped_vad, n_speakers=2, model='kmeans') #['spectralcluster','kmeans']
    # result_sd = uob_speakerdiarization.speaker_change_detection(grouped_vad, y, sr,frame_duration_ms=500, 
    #                                                           min_clusters = 2, max_clusters = 3) #!p_percentile issue
    
    
    ## Get timestamp
    grouped_result = uob_speakerdiarization.get_timestamp(result_diarization = result_sd)
    diarization_result = grouped_result
    
    ## Print & Save result to csv
    result_timestamps = []
    result_timestamp = ['starttime','endtime','duration','speaker_label']
    result_timestamps.append(result_timestamp)
    print('start\tend\tduration\tlabel')
    for i in grouped_result:
        end = i[0].timestamp+i[0].duration
        print(str(i[0].timestamp)+'\t'+str(end)+'\t'+str(i[0].duration)+'\t'+str(i[1]))

        result_timestamp = [str(i[0].timestamp),str(end),str(i[0].duration),str(i[1])]
        result_timestamps.append(result_timestamp)

    namef, namec = os.path.splitext(audioname)
    # namef_other, namef_index = namef.rsplit("_", 1)
    # save_name = '%s_%04d.%s'%(namef_other,namef_index,'csv')
    namec = namec[1:]
    save_name = '%s_%s.%s'%(namef, datetime.now().strftime('%y%m%d-%H%M%S'), 'csv')
    np.savetxt(os.path.join(audiopath,save_name), result_timestamps,
               delimiter=',', fmt='% s')
    
    # TODO: convert to common format
    
    return diarization_result



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
    result_timestamp = ['starttime','endtime','duration','speaker_label']
    result_timestamps.append(result_timestamp)
    print('Start\tEnd\tPoint\tSpeaker')
    for turn, xxx, speaker in diarization_result.itertracks(yield_label=True):
        # speaker speaks between turn.start and turn.end
        print(turn.start, turn.end, xxx, speaker)
        result_timestamp = [str(turn.start),str(turn.end),str(turn.end-turn.start),str(speaker)]
        result_timestamps.append(result_timestamp)

    namef, namec = os.path.splitext(audioname)
    # namef_other, namef_index = namef.rsplit("_", 1)
    # save_name = '%s_%04d.%s'%(namef_other,namef_index,'csv')
    namec = namec[1:]
    save_name = '%s_%s.%s'%(namef, datetime.now().strftime('%y%m%d-%H%M%S'), 'csv')
    np.savetxt(os.path.join(audiopath,save_name), result_timestamps,
               delimiter=',', fmt='% s')
    
    # TODO: convert to common format
    
    return diarization_result