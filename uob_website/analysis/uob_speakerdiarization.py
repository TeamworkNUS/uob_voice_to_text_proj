# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
''' Logging
Version     Date    Change_by   Description
#00     2022-Feb-28     Zhao Tongtong   Initial version  


'''
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import numpy as np
import pandas as pd
from analysis.resemblyzer.hparams import *

from analysis import uob_extractmodel



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
    df = pd.DataFrame(labelling, columns=['speaker_label', 'starttime', 'endtime'])

    return df
