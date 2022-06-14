import json
from logging import exception
import os
import shutil
from typing import Any, Optional
from pydub import AudioSegment
import wave
import filetype # to check file type


def check_file_type(audioname, audiopath):
    audiofile = os.path.join(audiopath, audioname)
    kind = filetype.guess(audiofile)
    if kind is None:
        print('Cannot guess file type!')
        return None, None

    print('File extension: %s' % kind.extension)
    print('File MIME type: %s' % kind.mime)
    
    return kind.extension, kind.mime
    



def audio2wav(from_audioname, from_audiopath, to_audioname, to_audiopath):
    from_audiofile = os.path.join(from_audiopath,from_audioname)
    to_audiofile = os.path.join(to_audiopath,to_audioname)
    AudioSegment.from_file(from_audiofile).export(to_audiofile,format='wav')



def get_audio_params(audioname, audiopath):
    '''
    Note: only works on .wav file
    '''
    audiofile = os.path.join(audiopath, audioname)
    f=wave.open(audiofile)
    params = f.getparams()
    channels = f.getnchannels()
    samplerate = f.getframerate()
    sampwidth = f.getsampwidth()
    nframes = f.getnframes()
    f.close()

    print('params: ',params)
    # print('channels: ',channels)
    # print('sample rate: ', samplerate)
    # print('sample width: ', sampwidth)
    # print('nframes: ', nframes)
    
    return params, channels, samplerate, sampwidth, nframes



def standardize_audio(from_audioname, from_audiopath, to_audioname, to_audiopath, sample_rate, no_of_channel):
    '''
    requirements:
        ffmpeg
        $ ffmpeg {global param} {input file param} -i {input file} {output file param} {output file}

    '''
    from_audiofile = os.path.join(from_audiopath,from_audioname)
    to_audiofile = os.path.join(to_audiopath,to_audioname)
    
    # cmd = "ffmpeg -i " + str(from_audiofile) + " -hide_banner " +" -ac " + str(no_of_channel) + " -ar " + str(sample_rate) +' ' + to_audiofile
    # print(cmd)
    # os.system(cmd)
    
    AudioSegment.from_wav(from_audiofile).set_frame_rate(sample_rate).set_channels(no_of_channel).export(to_audiofile,format='wav')
    

def get_audio_meta(audioname,audiopath):

    audiofile = os.path.join(audiopath, audioname)
    f=wave.open(audiofile)
    nframes = f.getnframes()
    rate = f.getframerate()
    duration = nframes / float(rate)
    bytes_size=os.path.getsize(audiofile)
    params = f.getparams()
    nchannels = f.getnchannels()
    samplerate = f.getframerate()
    sampwidth = f.getsampwidth()
    bit_type = f.getsampwidth() * 8
    bit_rate=samplerate * bit_type * nchannels
    f.close()

    # print('params: ',params)
    # print('channels: ',channels)
    # print('sample rate: ', samplerate)
    # print('sample width: ', sampwidth)
    # print('nframes: ', nframes)
    output_obj = {"nchannels":nchannels,"samplerate":samplerate,"sampwidth":sampwidth,"nframes":nframes, "duration":duration, "bytes_size":bytes_size, "bit_type":bit_type,"bit_rate":round(bit_rate)}
    output = json.dumps(output_obj)
    
    return output
        
        
    