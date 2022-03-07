# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
''' Logging
Version     Date    Change_by   Description
#00     2022-Feb-28     Zhao Tongtong   Initial version  


'''
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

from pydub import AudioSegment
from pydub.silence import split_on_silence
import sys
import os
import time
from datetime import datetime


def chunk_split_length_limit(chunk,min_silence_len=700,length_limit=60*1000,silence_thresh=-70):
    '''
    split the audio file based on sentences, and limit the longest time for each sentence, return to a list
    Args:
        chunk: audio file
        min_silence_len: if silence exceeds 700ms then split
        length_limit: each chunk lenth limit
        silence_thresh: regard as silence if less than -70dBFS 
    Return:
        done_chunks: chunks list
    '''
    todo_arr = []
    done_chunks = []
    todo_arr.append([chunk,min_silence_len,silence_thresh])
    
    while len(todo_arr)>0:
        # load an audio
        temp_chunk, temp_msl, temp_st = todo_arr.pop(0)
        # split
        # if not too long, then split ok
        if len(temp_chunk)<length_limit:
            done_chunks.append(temp_chunk)
        else:
            # if too long, start to process
            if temp_msl>100: #narrow down the silence length limit
                temp_msl-=100
            elif temp_st<-10: #increase the decible that regarded as silence
                temp_st+=10
            else:
                # output exceptions if the adjustment still cannot meet requriements
                tempname = 'tem_%d.wav'%int(time.time())
                chunk.export(tempname,format='wav')
                print('We have tried. Audio length %d decible %d is still too long. The chunk has been saved to%s'%(len(temp_chunk),temp_msl,temp_st,tempname))
                raise Exception
            # output the chunks in this split, and args
            msg = 'in splitting   length, remain[silence len, dbs]: %d,%d[%d,%d]'%(len(temp_chunk),len(todo_arr),temp_msl,temp_st)
            print(msg)
            # splits
            temp_chunks = split_on_silence(temp_chunk,min_silence_len=temp_msl,silence_thresh=temp_st)
            # split result process
            doing_arr = [[c,temp_msl,temp_st] for c in temp_chunks]
            todo_arr = doing_arr+todo_arr
    return done_chunks



def chunk_join_length_limit(chunks, joint_silence_len=1000,length_limit=60*1000,joint_lenth_limit=5*1000):
    '''
    joint chunks, and limit the longest time for each sentence, return result in a list
    Args:
        chunk: audio file
        joint_silence_len: file interval during joint, default 1.3s=1300ms
        length_limit: after joint, each file length will not exceed this value, default 60s
        joint_lenth_limit:  limit the file can be used for joint
    '''
    
    silence = AudioSegment.silent(duration=joint_silence_len)
    adjust_chunks=[]
    temp = AudioSegment.empty()
    for chunk in chunks:
        length = len(temp)+len(silence)+len(chunk)
        # if length < length_limit: # can join if less than 1 min
        #     temp += silence+chunk
        if length < joint_lenth_limit: # can join if less than 1 min
            if len(temp+silence+chunk) <length_limit:
                temp += silence+chunk
            else:
                adjust_chunks.append(temp)
                temp = chunk
        else: # if more than 1 min, 1. save previous one, 2. restart accumulate
            adjust_chunks.append(temp)
            temp = chunk
    else:
        adjust_chunks.append(temp)
    return adjust_chunks



def prepare_for_analysis(name, sound, 
                         silence_thresh=-70, 
                         min_silence_len = 700, 
                         length_limit = 60*1000, 
                         abandon_chunk_len = 500, 
                         joint_silence_len = 1300,
                         joint_length_limit = 5*1000,
                         withjoint = True):
    '''
    Args:
        name: recording file name
        sound: recording file data
        silence_thresh
        min_silence_len
        length_limit
        abandon_chunk_len
        joint_silence_len
        joint_lenth_limit
        withjoin
    Return:
        total: return the number of chunks
    '''
    
    # split based on sentences, each less than 1 min
    print('start to split\n',' *'*30)
    chunks = chunk_split_length_limit(sound, silence_thresh=silence_thresh, min_silence_len=min_silence_len, length_limit=length_limit)
    print('split ends. return #chunks:',len(chunks),'\n',' *'*30)
    
    # abandon chunks less than 0.5s
    for i in list(range(len(chunks)))[::-1]:
        if len(chunks[i])<=abandon_chunk_len:
            chunks.pop(i)
            
    print('get valid chunks ',len(chunks))
    
    # join conjoint chunks if too short, each chunk no more than 1 min
    if withjoint == True:
        chunks = chunk_join_length_limit(chunks, joint_silence_len=joint_silence_len, length_limit=length_limit,joint_lenth_limit=joint_length_limit)
        print('#chunks after joint: ',len(chunks))
    
    # process save path and name
    # if not os.path.exists('./chunks'): os.mkdir('./chunks')
    nowtime = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists('./chunks_'+name[:5]+'_'+nowtime): 
        os.mkdir('./chunks_'+name[:5]+'_'+nowtime)
    namef, namec = os.path.splitext(name)
    namec = namec[1:]
    print('namef: '+ namef + '; namec: '+namec)
    
    # save parameters into a .txt file
    hyperparams = f"""silence_thresh: {silence_thresh}
min_silence_len: {min_silence_len}
length_limit: {length_limit}
abandon_chunk_len: {abandon_chunk_len}
joint_silence_len: {joint_silence_len} 
joint_length_limit: {joint_length_limit}
withjoint: {withjoint} \n 
    """
    print('hyperparams: '+hyperparams)
    with open('./chunks_'+name[:5]+'_'+nowtime+'/'+"hyperparams.txt","a") as f:  # TODO: setup the path to save paramters
        f.write(hyperparams)
    
    
    # save all chunks
    total = len(chunks)
    for i in range(total):
        new = chunks[i]
        save_name = '%s_%04d.%s'%(namef,i,namec)
        print(save_name)
        new.export('./chunks_'+name[:5]+'_'+nowtime+'/'+save_name, format=namec)
    print('save done')
    
    return total, nowtime
        


def audio_segmentation(name,file):
    ## loading
    # name = 'Bdb001_interaction.wav'
    sound = AudioSegment.from_wav(file)
    print(sound)
    # sound = sound[:3*60*1000]  #use first 3min to test in case the file is too large. adjust parameters according to the result.
    
    ## set hyperparameters
    silence_thresh = -40  # lower than -70dBFS --> silence
    min_silence_len = 900  #  if silence exceeds 700ms then split
    length_limit = 60*1000  # if split, each chunk cannot be more than 30 s
    abandon_chunk_len = 500  # abandon the chunk less than 500ms
    joint_silence_len = 0  # add 1300ms interval when joining the chunks
    joint_lenth_limit = 60*1000
    withjoint = True
    
    ## output
    total_chunks, nowtime = prepare_for_analysis(name, sound, silence_thresh, min_silence_len, length_limit, abandon_chunk_len, joint_silence_len, joint_lenth_limit, withjoint)
    # print(total_chunks)
    
    return total_chunks, nowtime

