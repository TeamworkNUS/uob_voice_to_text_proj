import json
import os
import re
import shutil
import librosa
import pandas as pd

from pyannote.audio import Pipeline as pa_Pipeline
from sqlalchemy import null

from analysis import (uob_audiosegmentation, uob_mainprocess, uob_noisereduce, uob_speechenhancement, uob_speechenhancement_new, uob_superresolution,
                      uob_speakerdiarization, uob_stt, uob_storage)
from .uob_init import (
    pretrained_model_path,
    segmentation_threshold,
    FLG_REDUCE_NOISE,
    FLG_SPEECH_ENHANCE,
    FLG_SPEECH_ENHANCE_NEW,
    FLG_SUPER_RES,
    sttModel,
    sdModel,
    flg_slice_orig,
    STT_SAMPLERATE
)


def sd_and_stt(audio, starttime, analysis_name, username):
    #### * Load Models
    print('Pretrained Models Loading Start')
    ## audio preprocessing
    nr_model = uob_noisereduce.load_noisereduce_model_local(quantized=False) if FLG_REDUCE_NOISE else None
    se_model = uob_speechenhancement.load_speechenhancement_model_local(quantized=False) if FLG_SPEECH_ENHANCE else None
    se_model_new = uob_speechenhancement_new.load_speechenhancement_model_local() if FLG_SPEECH_ENHANCE_NEW == True else None
    sr_model = uob_superresolution.load_superresolution_model_local(quantized=False) if FLG_SUPER_RES else None
    
    ## SD
    vad_model_vggvox2 = uob_speakerdiarization.load_vad_model_local(quantized=False) if sdModel=='malaya' else None
    sv_model_speakernet, sv_model_vggvox2 = uob_speakerdiarization.load_speaker_vector_model_local(quantized=False) if sdModel=='malaya' else None, None
    pa_pipeline = pa_Pipeline.from_pretrained('pyannote/speaker-diarization') if sdModel=='pyannoteaudio' else None
    
    print('Pretrained Models Loading Done!!!')
    print('*' * 30)
    
    #### * Initialize Variables
    num_audio_files = 1
    savetime = starttime.strftime("%Y%m%d_%H%M%S")
    audio_file = os.path.join(audio.path_orig, audio.audio_name).replace("\\","/")
    if 'sd_starttime_{}'.format(audio.audio_id) not in globals():
        globals()['sd_starttime_{}'.format(audio.audio_id)] = 0.0
    
    #### * Segmentation
    chunksfolder = ''
    audio_duration = json.loads(audio.audio_meta)["duration"]
    if audio_duration > segmentation_threshold:  # segment if longer than 5 min=300s
        totalchunks, chunksfolder = uob_audiosegmentation.audio_segmentation(name=audio.audio_name,path=audio.path_orig,savetime=savetime)
        num_audio_files = totalchunks
        print('chunksfolder: ', chunksfolder)
        print('Segmentation Done!!!\n','*' * 30)

    print('Number of audio files to process: ', num_audio_files)
    print('*' * 30)
    
    
    #### * Process SD
    tem_sd_result = []
    tem_sd_index = 0
    if chunksfolder != '':
        for filename in os.listdir(chunksfolder+"/"):
            if filename.endswith(".wav"): 
                ### * Load chunk file
                file = os.path.join(chunksfolder, filename).replace("\\","/")
                y, sr = librosa.load(file, sr=None, mono=True)  #-->mono: y = y/2

                        
                ### * Process
                ## return list ['index','starttime','endtime','duration','speaker_label','chunk_filename']
                sd_result, audioname, audiopath = uob_mainprocess.sd_process(y, sr, 
                                                        audioname=filename,
                                                        audiopath=chunksfolder,
                                                        audiofile=file,
                                                        nr_model=nr_model,   # ?: [nr_model, nr_quantized_model]
                                                        se_model=se_model,
                                                        se_model_new=se_model_new,
                                                        sr_model=sr_model,
                                                        vad_model=vad_model_vggvox2,
                                                        sv_model=sv_model_speakernet,    # ?: sv_model_speakernet, sv_model_vggvox2
                                                        pipeline=pa_pipeline,
                                                        chunks=True, #fixed
                                                        reducenoise=FLG_REDUCE_NOISE,
                                                        speechenhance=FLG_SPEECH_ENHANCE,
                                                        speechenhance_new=FLG_SPEECH_ENHANCE_NEW,
                                                        superresolution=FLG_SUPER_RES,
                                                        sd_proc=sdModel)  # ?: [pyannoteaudio, malaya, resemblyzer]
                        
                for row in sd_result[1:]:
                    if 'not' not in row[4].lower():
                        tem_sd_index += 1
                        tem_sd_result.append( [tem_sd_index,
                                                row[1]+globals()['sd_starttime_{}'.format(audio.audio_id)], 
                                                row[2]+globals()['sd_starttime_{}'.format(audio.audio_id)],
                                                row[3],
                                                row[4],
                                                row[5],
                                                row[1],
                                                row[2]])
                        if row[0] == len(sd_result[1:]):
                            globals()['sd_starttime_{}'.format(audio.audio_id)] += row[2]
                            print(globals()['sd_starttime_{}'.format(audio.audio_id)])
                        
                
    else:
        ### * Load single file
        y, sr = librosa.load(audio_file, sr=None, mono=True)  #-->mono: y = y/2

        ### * Process
        ## return list ['index','starttime','endtime','duration','speaker_label','chunk_filename']
        sd_result, audioname, audiopath = uob_mainprocess.sd_process(y, sr, 
                                            audioname=audio.audio_name,
                                            audiopath=audio.path_orig,
                                            audiofile=audio_file,
                                            nr_model=nr_model,   # ?: [nr_model, nr_quantized_model]
                                            se_model=se_model,
                                            se_model_new=se_model_new,
                                            sr_model=sr_model,
                                            vad_model=vad_model_vggvox2,
                                            sv_model=sv_model_speakernet,    # ?: sv_model_speakernet, sv_model_vggvox2
                                            pipeline=pa_pipeline,
                                            chunks=False, #fixed
                                            reducenoise=FLG_REDUCE_NOISE, 
                                            speechenhance=FLG_SPEECH_ENHANCE,
                                            speechenhance_new=FLG_SPEECH_ENHANCE_NEW,
                                            superresolution=FLG_SUPER_RES,
                                            sd_proc=sdModel)  # ?: [pyannoteaudio, malaya, resemblyzer]

        for row in sd_result[1:]:
            if 'not' not in row[4].lower():
                tem_sd_index += 1
                tem_sd_result.append([tem_sd_index,
                                        row[1],
                                        row[2],
                                        row[3],
                                        row[4],
                                        row[5],
                                        0,
                                        0])
    
    print('Speaker Diarization Done!!!')
    print('*' * 30)
    
    #### * Process STT
    final_sd_result = pd.DataFrame(tem_sd_result, columns=['index','starttime','endtime','duration','speaker_label','chunk_filename','chunk_starttime','chunk_endtime'])
    ###  Cut audio into slices by SD result
    namef, namec = os.path.splitext(audio.audio_name)
    slices_path = os.path.join(audiopath, namef + '_slices_' + savetime).replace("\\","/")
    try:
        if not os.path.exists(slices_path):
            os.mkdir(slices_path)
        elif os.path.exists(slices_path):
            shutil.rmtree(slices_path) # delete whole folder
            os.mkdir(slices_path)
    except Exception as e:
        print('Failed to delete or create slices folder %s. Reason: %s' % (slices_path, e))
    
    # Start to cut
    if chunksfolder != '':
        if flg_slice_orig == False:
            for filename in os.listdir(chunksfolder+"_processed/"):
                tem_sd_result_forSlices = final_sd_result[final_sd_result['chunk_filename']==filename]
                tem_sd_result_forSlices['index'] = tem_sd_result_forSlices['index'].astype(str)
                tem_sd_result_forSlices = tem_sd_result_forSlices[['index','chunk_starttime','chunk_endtime']].values.tolist()
                uob_mainprocess.cut_audio_by_timestamps(start_end_list=tem_sd_result_forSlices, audioname=filename, audiofile=os.path.join(chunksfolder+"_processed/",filename).replace("\\","/"), part_path=slices_path)
        else:
            for filename in os.listdir(chunksfolder+"/"):
                if re.match('^(%s_)(\d{4})'%(namef), filename):
                    tem_sd_result_forSlices = final_sd_result[final_sd_result['chunk_filename'].str[-8:]==filename[-8:]]
                    tem_sd_result_forSlices['index'] = tem_sd_result_forSlices['index'].astype(str)
                    tem_sd_result_forSlices = tem_sd_result_forSlices[['index','chunk_starttime','chunk_endtime']].values.tolist()
                    print(tem_sd_result_forSlices)
                    uob_mainprocess.cut_audio_by_timestamps(start_end_list=tem_sd_result_forSlices, audioname=filename, audiofile=os.path.join(chunksfolder,filename).replace("\\","/"), part_path=slices_path)

            
    else:

        if FLG_SUPER_RES == True:
            if FLG_REDUCE_NOISE == False:
                filename_forSlices = namef + "_sr.wav"
            elif FLG_REDUCE_NOISE == True and (FLG_SPEECH_ENHANCE == False and FLG_SPEECH_ENHANCE_NEW == False):
                filename_forSlices = namef + "_nr_sr.wav"
            elif FLG_REDUCE_NOISE == True and (FLG_SPEECH_ENHANCE == True or FLG_SPEECH_ENHANCE_NEW == True):
                filename_forSlices = namef + "_nr_se_sr.wav"
            else:
                raise Exception(
                            f'There is an exception when searching for .wav file to cut into slices.'
                        )
        else:
            if FLG_REDUCE_NOISE == False:
                filename_forSlices = namef + ".wav"
            elif FLG_REDUCE_NOISE == True and (FLG_SPEECH_ENHANCE == False and FLG_SPEECH_ENHANCE_NEW == False):
                filename_forSlices = namef + "_nr.wav"
            elif FLG_REDUCE_NOISE == True and (FLG_SPEECH_ENHANCE == True or FLG_SPEECH_ENHANCE_NEW == True):
                filename_forSlices = namef + "_nr_se.wav"
            else:
                raise Exception(
                            f'There is an exception when searching for .wav file to cut into slices.'
                        )

        if flg_slice_orig == True:
            file_forSlices = audio_file
        else:
            file_forSlices = os.path.join(audiopath, filename_forSlices).replace("\\","/")
        uob_mainprocess.cut_audio_by_timestamps(start_end_list=tem_sd_result, audioname=filename_forSlices, audiofile=file_forSlices, part_path=slices_path)

    print('Cut Slices Done')
    print('*' * 30)
    
    
    ###  Speech to Text Conversion
    ## Load stt model
    stt_model = uob_stt.load_stt_model(stt_model=sttModel,pretrained_model_path=pretrained_model_path, sr=STT_SAMPLERATE)
    ## STT start
    print('*'*30)
    print('STT Conversion Start')
    
    stt = uob_mainprocess.stt_process(sttModel = sttModel, slices_path=slices_path, rec=stt_model, sr = STT_SAMPLERATE)
    
    print('STT Conversion Done')
    print('*'*30)

    ### merge SD and STT
    transactionDf = pd.merge(left = final_sd_result, right = stt, on="index",how='left')
    
    
    #### *  Speaker Labelling
    print('*'*30)
    print("Speaker Labelling Start")
    
    final = uob_mainprocess.speaker_label_func(transactionDf,pretrained_model_path=os.path.join(pretrained_model_path,'label/label_wordvector_model').replace("\\","/"),checklist_path=os.path.join(pretrained_model_path,'label/').replace("\\","/"))
    
    print("Speaker Labelling Done")
    print('*'*30)
    
    
    print(final)
    
    
    #### * Store output to database
    print('*'*30)
    print("Insert Output to Database Start")
    
    uob_storage.dbInsertSTT(finalDf=final, audio_id=audio.audio_id, slices_path=slices_path, username=username)
    
    if FLG_REDUCE_NOISE or FLG_SPEECH_ENHANCE or FLG_SPEECH_ENHANCE_NEW or FLG_SUPER_RES:
        if chunksfolder != '':
            audio_name_processed = "*"
            path_processed = chunksfolder+"_processed/" if flg_slice_orig == False else chunksfolder
        else:
            audio_name_processed = audioname
            path_processed = audiopath
    else:
        audio_name_processed = null
        path_processed = null
    analysis = json.loads(audio.analysis)
    print('json object analysis:', analysis)
    if analysis_name not in analysis.values():
        analysis_key = int(max(analysis.keys()))+1 if analysis!={} else 0
        analysis[analysis_key] = analysis_name
    else:
        print(analysis_name, 'has been done before')
    analysis = json.dumps(analysis)
    print('json string analysis:', analysis)
    uob_storage.dbUpdateAudio_processedInfo(audio_id = audio.audio_id, 
                                            username = username,
                                            audio_name_processed = audio_name_processed, 
                                            path_processed = path_processed,
                                            analysis = analysis
                                            )
    
    print("Insert Output to Database Done")
    print('*'*30)
    
    
    # Clear variables
    globals()['sd_starttime_{}'.format(audio.audio_id)] = 0.0
    
    #### * End of SD+STT
    # print("SD+STT Done!!!")
    # print('*'*30)
    
    return final

    