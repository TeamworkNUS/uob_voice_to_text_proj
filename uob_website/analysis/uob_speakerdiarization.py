# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
''' Logging
Version     Date    Change_by   Description
#00     2022-Feb-28     Zhao Tongtong   Initial version  


'''
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

from malaya_speech import Pipeline
import malaya_speech
import numpy as np
import matplotlib.pyplot as plt
from pyannote.audio import Pipeline as pa_Pipeline
from sklearn.cluster import KMeans, SpectralClustering
import speechbrain
import os
import pandas as pd
from analysis.resemblyzer.hparams import *

from analysis import uob_extractmodel

'''
Note:
    malaya-speech ：spectralcluster-0.1.0
    pyannote-audio 0.0.1 : spectralcluster<0.3,>=0.2.4
'''


def load_vad_model(quantized=False):
    model_vggvoxv2 = malaya_speech.vad.deep_model(model = 'vggvox-v2',quantized=quantized)
    return model_vggvoxv2

def load_vad_model_local(quantized=False):
    model_vggvoxv2 = uob_extractmodel.get_model_vad(model = 'vggvox-v2',quantized=quantized)
    return model_vggvoxv2


def load_speaker_vector_model(quantized=False):
    model_speakernet = malaya_speech.speaker_vector.deep_model('speakernet',quantized=quantized)
    model_vggvox2 = malaya_speech.speaker_vector.deep_model('vggvox-v2',quantized=quantized)
    return model_speakernet, model_vggvox2

def load_speaker_vector_model_local(quantized=False):
    model_speakernet = uob_extractmodel.get_model_speaker_vector(model='speakernet',quantized=quantized)
    model_vggvox2 = uob_extractmodel.get_model_speaker_vector(model='vggvox-v2',quantized=quantized)
    return model_speakernet, model_vggvox2


def load_audio_file(audio_file):
    y, sr = malaya_speech.load(audio_file)
    print(len(y), sr)
    return y, sr



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                   Malaya-Speech                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def load_vad(y, sr, vad_model, frame_duration_ms=30, threshold_to_stop=0.3):
    # vad = malaya_speech.vad.deep_model(model = 'vggvox-v2') 
    vad = vad_model
    frames = list(malaya_speech.utils.generator.frames(y, frame_duration_ms, sr))
    p = Pipeline()
    pipeline = (
        p.batching(5)
        .foreach_map(vad.predict)
        .flatten()
    )
    
    result = p.emit(frames)
    print(result.keys())
    
    frames_vad = [(frame, result['flatten'][no]) for no, frame in enumerate(frames)]
    grouped_vad = malaya_speech.utils.group.group_frames(frames_vad)  #Group multiple frames based on label.
    grouped_vad = malaya_speech.utils.group.group_frames_threshold(grouped_vad, threshold_to_stop = threshold_to_stop) # Group multiple frames based on label and threshold to stop.
    
    # visualize Amptitude-Time(s)  # TODO: comment the visualization
    # malaya_speech.extra.visualization.visualize_vad(y, grouped_vad, sr, figsize = (15, 3))
    
    return grouped_vad


## Speaker Similarity
def speaker_similarity(speaker_vector, grouped_vad):

    result_diarization_model = malaya_speech.diarization.speaker_similarity(grouped_vad, speaker_vector)
    
    return result_diarization_model


## Affinity Propagation
def affinity_propagation(speaker_vector, grouped_vad):
    
    result_diarization_ap_model = malaya_speech.diarization.affinity_propagation(grouped_vad, speaker_vector)
    
    return result_diarization_ap_model


## Spectral Clustering
def spectral_clustering(speaker_vector, grouped_vad, min_clusters=2, max_clusters=5):
    result_diarization_sc_model = malaya_speech.diarization.spectral_cluster(grouped_vad, speaker_vector,
                                                            min_clusters = min_clusters,
                                                            max_clusters = max_clusters)
       
    return result_diarization_sc_model


## Static N speakers using sklearn clustering
def n_speakers_clustering(speaker_vector, grouped_vad, n_speakers=2, model='kmeans'): #['kmeans','spectralcluster']
    
    if model == 'kmeans': 
        ## kmeans
        kmeans = KMeans(n_clusters = n_speakers)
        result_diarization_kmeans_model = malaya_speech.diarization.n_clustering(grouped_vad, speaker_vector,
                                                                                model = kmeans,
                                                                                norm_function = lambda x: x)
        return result_diarization_kmeans_model
        
    elif model == 'spectralcluster': 
        ## spectral clustering
        spectralclustering = SpectralClustering(n_clusters = n_speakers)
        result_diarization_spectralclustering_model = malaya_speech.diarization.n_clustering(grouped_vad, speaker_vector,
                                                                                model = spectralclustering,
                                                                                norm_function = lambda x: x)
        return result_diarization_spectralclustering_model


## Use speaker change detection
def speaker_change_detection( speaker_vector, grouped_vad, y, sr, 
                             frame_duration_ms=500, min_clusters = 2, max_clusters = 5):
        
    # speakernet = malaya_speech.speaker_change.deep_model('speakernet')
    speakernet = uob_extractmodel.get_model_speaker_change(model='speakernet') #TODO: to move to main.py?
    frames_speaker_change = list(malaya_speech.utils.generator.frames(y, frame_duration_ms, sr))
    probs_speakernet = [(frame, speakernet.predict_proba([frame])[0, 1]) for frame in frames_speaker_change] # !: check function 'predict_proba' for all models.-->'Speaker2Vec' object has no attribute 'predict_proba'
    
    nested_grouped_vad = malaya_speech.utils.group.group_frames(grouped_vad)
    splitted_speakervec = malaya_speech.speaker_change.split_activities(nested_grouped_vad, probs_speakernet)
    
    result_diarization_sc_splitted_model = malaya_speech.diarization.spectral_cluster(splitted_speakervec, 
                                                                                       speaker_vector,
                                                                                       min_clusters = min_clusters,
                                                                                       max_clusters = max_clusters)
    
    return result_diarization_sc_splitted_model


## Get timestamp
def get_timestamp(result_diarization):
    grouped = malaya_speech.group.group_frames(result_diarization)
    
    ## Output timestamp, duration, cluster label
    # grouped[1][0].timestamp, grouped[1][0].duration, grouped[1][1]
   
    return grouped


## Visualization
def visualization_sd(y, grouped_vad, sr, result_diarization):
    nrows = 2
    fig, ax = plt.subplots(nrows = nrows, ncols = 1)
    fig.set_figwidth(20)
    fig.set_figheight(nrows * 3)
    malaya_speech.extra.visualization.visualize_vad(y, grouped_vad, sr, ax = ax[0])
    malaya_speech.extra.visualization.plot_classification(result_diarization, 
                                    'diarization', ax = ax[1],
                                    x_text = 0.01, y_text = 0.5)
    fig.tight_layout()
    plt.show()



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                  Pyannote-Audio                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 
def pyannoteaudio_speaker_diarization(audiofile, pipeline):
    # pipeline = pa_Pipeline.from_pretrained('pyannote/speaker-diarization')
    own_file = {'audio': audiofile}
    diarization_result = pipeline(own_file, num_speakers=2)
    # print(diarization_result)  # TODO: comment
    # diarization_result #--> Can only visualize in notebook
    # for turn, xxx, speaker in diarization_result.itertracks(yield_label=True):
    #     # speaker speaks between turn.start and turn.end
    #     print(turn.start,turn.end,xxx,speaker)
    return diarization_result


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                    Resemblyzer                                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def resemblyzer_speaker_diarization(cont_embeds, wav_splits) -> pd.DataFrame:
    ###### Spectral Clustering

    from spectralcluster import SpectralClusterer

    clusterer = SpectralClusterer(
        min_clusters=2,
        max_clusters=2,
        # p_percentile=0.85,   #Latest SpectralCluster package removes this two params.
        # gaussian_blur_sigma=1
        )

    labels = clusterer.predict(cont_embeds)

    ##### Arranging the vectors as per the time windows and labels

    def create_labelling(labels, wav_splits):
        # from resemblyzer import sampling_rate
        times = [((s.start + s.stop) / 2) / sampling_rate for s in wav_splits]
        labelling = []
        start_time = 0

        for i, time in enumerate(times):
            if i > 0 and labels[i] != labels[i - 1]:
                temp = [str(labels[i - 1]), start_time, time]
                labelling.append(tuple(temp))
                start_time = time
            if i == len(times) - 1:
                temp = [str(labels[i]), start_time, time]
                labelling.append(tuple(temp))

        return labelling

    labelling = create_labelling(labels, wav_splits)

    ###### Creating Data Frame for Speaker Identification

    # df = pd.DataFrame(labelling, columns=['Speaker', 'Start_Time', 'End_Time'])
    df = pd.DataFrame(labelling, columns=['speaker_label', 'starttime', 'endtime'])

    return df
