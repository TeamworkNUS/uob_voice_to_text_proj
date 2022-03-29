import tensorflow as tf
import os
import logging

# import malaya_speech
from malaya_speech.utils import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)

# from malaya_speech.supervised import unet, classification
from malaya_speech.model.tf import UNET, UNETSTFT, UNET1D

from malaya_speech.utils import featurization
from malaya_speech.config import (
    speakernet_featurizer_config as speakernet_config,
)
from malaya_speech.model.tf import (
    Speakernet,
    Speaker2Vec,
    SpeakernetClassification,
    MarbleNetClassification,
    Classification,
)

import torch
from torch import nn
import numpy as np
import librosa
from time import perf_counter as timer
from typing import Union, List
from pathlib import Path
from resemblyzer.hparams import *

from vosk import Model, KaldiRecognizer, SetLogLevel

from init import pretrained_model_path
import uob_noisereduce, uob_speakerdiarization  #for test




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                    get_model_xxx                                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def get_model_noisereduce(pretrained_model_path = pretrained_model_path, model='resnet-unet',  quantized:bool=False, **kwargs):
    '''
    available models:
        * ``'unet'`` - pretrained UNET.
        * ``'resnet-unet'`` - pretrained resnet-UNET.
        * ``'resnext'`` - pretrained resnext-UNET.
    '''
    _availability = {
        'unet': {
            'Size (MB)': 78.9,
            'Quantized Size (MB)': 20,
            'SUM MAE': 0.862316,
            'MAE_SPEAKER': 0.460676,
            'MAE_NOISE': 0.401640,
            'SDR': 9.17312,
            'ISR': 13.92435,
            'SAR': 13.20592,
        },
        'resnet-unet': {
            'Size (MB)': 96.4,
            'Quantized Size (MB)': 24.6,
            'SUM MAE': 0.82535,
            'MAE_SPEAKER': 0.43885,
            'MAE_NOISE': 0.38649,
            'SDR': 9.45413,
            'ISR': 13.9639,
            'SAR': 13.60276,
        },
        'resnext-unet': {
            'Size (MB)': 75.4,
            'Quantized Size (MB)': 19,
            'SUM MAE': 0.81102,
            'MAE_SPEAKER': 0.44719,
            'MAE_NOISE': 0.363830,
            'SDR': 8.992832,
            'ISR': 13.49194,
            'SAR': 13.13210,
        },
    }
    ## Check model availability
    model = model.lower()
    if model not in _availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.noise_reduction.available_model()`.'
        )
    
    ## Declare model file
    module='noise-reduction'
    path = pretrained_model_path
    if quantized:
        path = os.path.join(module, f'{model}-quantized')
        quantized_path = os.path.join(path, 'model.pb')
        modelfile = os.path.join(pretrained_model_path, quantized_path).replace('\\', '/')
        # print('true')
    else:
        path = os.path.join(module, model, 'model.pb')
        modelfile = os.path.join(pretrained_model_path, path).replace('\\', '/')
        # print('false')
    
    # print(modelfile)
    ## Check if model.pb exists
    path_dict = check_file(
                            model=model,
                            base_path=pretrained_model_path,
                            module=module,
                            keys={'model': 'model.pb'},
                            validate=True,
                            quantized=quantized,
                            **kwargs,
                        )
    print(path_dict)
    
    
    model_object = load_stft(modelfile=modelfile, model=model, module=module, instruments=['voice', 'noise'], quantized=quantized) #

    # print(model_object)

    # ? Test
    # y, sr = malaya_speech.load('The-Singaporean-White-Boy_first60s.wav',sr=44100)
    # y = uob_noisereduce.output_voice_pipeline(y, sr, model_object, frame_duration_ms=15000)
    # print(y)

    return model_object


def get_model_speechenhancement(pretrained_model_path = pretrained_model_path, model='unet', quantized:bool=False, **kwargs):
    '''
    available models:
        * ``'unet'`` - pretrained UNET Speech Enhancement.
        * ``'resnet-unet'`` - pretrained resnet-UNET Speech Enhancement.
        * ``'resnext-unet'`` - pretrained resnext-UNET Speech Enhancement.
    '''
    _sampling_availability = {
        'unet': {
            'Size (MB)': 40.7,
            'Quantized Size (MB)': 10.3,
            'SDR': 9.877178,
            'ISR': 15.916217,
            'SAR': 13.709130,
        },
        'resnet-unet': {
            'Size (MB)': 36.4,
            'Quantized Size (MB)': 9.29,
            'SDR': 9.43617,
            'ISR': 16.86103,
            'SAR': 12.32157,
        },
        'resnext-unet': {
            'Size (MB)': 36.1,
            'Quantized Size (MB)': 9.26,
            'SDR': 9.685578,
            'ISR': 16.42137,
            'SAR': 12.45115,
        },
    }
    
    ## Check model availability
    model = model.lower()
    if model not in _sampling_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.speech_enhancement.available_deep_enhance()`.'
        )

    ## Declare model file
    module = 'speech-enhancement'
    path = pretrained_model_path
    if quantized:
        path = os.path.join(module, f'{model}-quantized')
        quantized_path = os.path.join(path, 'model.pb')
        modelfile = os.path.join(pretrained_model_path, quantized_path).replace('\\', '/')
        # print('true')
    else:
        path = os.path.join(module, model, 'model.pb')
        modelfile = os.path.join(pretrained_model_path, path).replace('\\', '/')
        # print('false')
    
    # print(modelfile)
    ## Check if model.pb exists
    path_dict = check_file(
                            model=model,
                            base_path=pretrained_model_path,
                            module=module,
                            keys={'model': 'model.pb'},
                            validate=True,
                            quantized=quantized,
                            **kwargs,
                            )
    print(path_dict)
    
    model_object = load_1d(
        modelfile=modelfile,
        model=model,
        module=module,
        quantized=quantized,
        **kwargs
    )
    
    return model_object
    


def get_model_vad(pretrained_model_path = pretrained_model_path, model = 'vggvox-v2', quantized:bool=False, **kwargs):
    '''
    available models:
        * ``'vggvox-v1'`` - finetuned VGGVox V1.
        * ``'vggvox-v2'`` - finetuned VGGVox V2.
        * ``'speakernet'`` - finetuned SpeakerNet.
        * ``'marblenet-factor1'`` - Pretrained MarbleNet * factor 1.
        * ``'marblenet-factor3'`` - Pretrained MarbleNet * factor 3.
        * ``'marblenet-factor5'`` - Pretrained MarbleNet * factor 5.
    '''
    
    _availability = {
        'vggvox-v1': {
            'Size (MB)': 70.8,
            'Quantized Size (MB)': 17.7,
            'Accuracy': 0.80984375,
        },
        'vggvox-v2': {
            'Size (MB)': 31.1,
            'Quantized Size (MB)': 7.92,
            'Accuracy': 0.8196875,
        },
        'speakernet': {
            'Size (MB)': 20.3,
            'Quantized Size (MB)': 5.18,
            'Accuracy': 0.7340625,
        },
        'marblenet-factor1': {
            'Size (MB)': 0.526,
            'Quantized Size (MB)': 0.232,
            'Accuracy': 0.8491875,
        },
        'marblenet-factor3': {
            'Size (MB)': 3.21,
            'Quantized Size (MB)': 0.934,
            'Accuracy': 0.83855625,
        },
        'marblenet-factor5': {
            'Size (MB)': 8.38,
            'Quantized Size (MB)': 2.21,
            'Accuracy': 0.843540625,
        }
    }

    ## Check model availability
    model = model.lower()
    if model not in _availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.vad.available_model()`.'
        )
    
    ## Declare model file
    module='vad'
    path = pretrained_model_path
    if quantized:
        path = os.path.join(module, f'{model}-quantized')
        quantized_path = os.path.join(path, 'model.pb')
        modelfile = os.path.join(pretrained_model_path, quantized_path).replace('\\', '/')
        # print('true')
    else:
        path = os.path.join(module, model, 'model.pb')
        modelfile = os.path.join(pretrained_model_path, path).replace('\\', '/')
        # print('false')
    
    # print(modelfile)
    ## Check if model.pb exists
    path_dict = check_file(
                                model=model,
                                base_path=pretrained_model_path,
                                module=module,
                                keys={'model': 'model.pb'},
                                validate=True,
                                quantized=quantized,
                                **kwargs,
                            )
    print(path_dict)
    
    settings = {
        'vggvox-v1': {'frame_len': 0.005, 'frame_step': 0.0005},
        'vggvox-v2': {'hop_length': 24, 'concat': False, 'mode': 'eval'},
        'speakernet': {'frame_ms': 20, 'stride_ms': 1.0},
        'marblenet-factor1': {'feature_type': 'mfcc'},
        'marblenet-factor3': {'feature_type': 'mfcc'},
        'marblenet-factor5': {'feature_type': 'mfcc'},
    }
    

    model_object = load(
        modelfile = modelfile,
        model=model,
        module=module,
        extra=settings[model],
        label=[False, True],
        quantized=quantized,
        **kwargs
    )
    
    # ? Test
    # y, sr = malaya_speech.load('The-Singaporean-White-Boy_first60s.wav',sr=44100)
    # grouped_vad = uob_speakerdiarization.load_vad(y, sr, vad_model = model_object, frame_duration_ms=30, threshold_to_stop=0.3)
    # print(grouped_vad)
    
    return model_object




def get_model_speaker_vector(pretrained_model_path = pretrained_model_path, model: str = 'speakernet', quantized: bool = False, **kwargs):
    '''
    available models:
        * ``'vggvox-v1'`` - VGGVox V1, embedding size 1024, exported from https://github.com/linhdvu14/vggvox-speaker-identification
        * ``'vggvox-v2'`` - VGGVox V2, embedding size 512, exported from https://github.com/WeidiXie/VGG-Speaker-Recognition
        * ``'deep-speaker'`` - Deep Speaker, embedding size 512, exported from https://github.com/philipperemy/deep-speaker
        * ``'speakernet'`` - SpeakerNet, embedding size 7205, exported from https://github.com/NVIDIA/NeMo/tree/main/examples/speaker_recognition
    '''
    
    _availability = {
        'deep-speaker': {
            'Size (MB)': 96.7,
            'Quantized Size (MB)': 24.4,
            'Embedding Size': 512,
            'EER': 0.2187,
        },
        'vggvox-v1': {
            'Size (MB)': 70.8,
            'Quantized Size (MB)': 17.7,
            'Embedding Size': 1024,
            'EER': 0.1407,
        },
        'vggvox-v2': {
            'Size (MB)': 43.2,
            'Quantized Size (MB)': 7.92,
            'Embedding Size': 512,
            'EER': 0.0445,
        },
        'speakernet': {
            'Size (MB)': 35,
            'Quantized Size (MB)': 8.88,
            'Embedding Size': 7205,
            'EER': 0.02122,
        },
    }
    
    ## Check model availability
    model = model.lower()
    if model not in _availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.speaker_vector.available_model()`.'
        )
    
    ## Declare model file
    module='speaker-vector'
    path = pretrained_model_path
    if quantized:
        path = os.path.join(module, f'{model}-quantized')
        quantized_path = os.path.join(path, 'model.pb')
        modelfile = os.path.join(pretrained_model_path, quantized_path).replace('\\', '/')
        # print('true')
    else:
        path = os.path.join(module, model, 'model.pb')
        modelfile = os.path.join(pretrained_model_path, path).replace('\\', '/')
        # print('false')
    
    # print(modelfile)
    ## Check if model.pb exists
    path_dict = check_file(
                                    model=model,
                                    base_path=pretrained_model_path,
                                    module=module,
                                    keys={'model': 'model.pb'},
                                    validate=True,
                                    quantized=quantized,
                                    **kwargs,
                                )
    print(path_dict)
    
    
    
    model_object = load(
        modelfile=modelfile,
        model=model,
        module=module,
        extra={},
        label={},
        quantized=quantized,
        **kwargs
    )
    
    return model_object


def get_model_speaker_change(pretrained_model_path = pretrained_model_path, model: str = 'speakernet', quantized: bool = False, **kwargs):
    '''
    available models:
        * ``'vggvox-v2'`` - finetuned VGGVox V2.
        * ``'speakernet'`` - finetuned SpeakerNet.
    '''
    
    _availability = {
        'vggvox-v2': {
            'Size (MB)': 31.1,
            'Quantized Size (MB)': 7.92,
            'Accuracy': 0.63979,
        },
        'speakernet': {
            'Size (MB)': 20.3,
            'Quantized Size (MB)': 5.18,
            'Accuracy': 0.64524,
        },
    }
    ## Check model availability
    model = model.lower()
    if model not in _availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.speaker_change.available_model()`.'
        )

    ## Declare model file
    module='speaker-change'
    path = pretrained_model_path
    if quantized:
        path = os.path.join(module, f'{model}-quantized')
        quantized_path = os.path.join(path, 'model.pb')
        modelfile = os.path.join(pretrained_model_path, quantized_path).replace('\\', '/')
        # print('true')
    else:
        path = os.path.join(module, model, 'model.pb')
        modelfile = os.path.join(pretrained_model_path, path).replace('\\', '/')
        # print('false')
    
    # print(modelfile)
    ## Check if model.pb exists
    path_dict = check_file(
                                    model=model,
                                    base_path=pretrained_model_path,
                                    module=module,
                                    keys={'model': 'model.pb'},
                                    validate=True,
                                    quantized=quantized,
                                    **kwargs,
                                )
    print(path_dict)
    
    
    settings = {
        'vggvox-v2': {'hop_length': 50, 'concat': False, 'mode': 'eval'},
        'speakernet': {'frame_ms': 20, 'stride_ms': 2},
    }

    model_object = load(
        modelfile=modelfile,
        model=model,
        module=module,
        extra=settings[model],
        label=[False, True],
        quantized=quantized,
        **kwargs
    )
    
    return model_object


def get_model_stt_vosk(pretrained_model_path=pretrained_model_path, sr=16000):
    # sample_rate=sr
    model = Model(pretrained_model_path) #Model("model")
    rec = KaldiRecognizer(model, sr)
    
    return model, rec

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                        Utils                                        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def load_stft(modelfile, model, module, instruments, quantized=False, **kwargs):
    # path = check_file(
    #     file=model,
    #     module=module,
    #     keys={'model': 'model.pb'},
    #     quantized=quantized,
    #     **kwargs,
    # )
    g = load_graph(modelfile, **kwargs)
    inputs = ['Placeholder']
    outputs = [f'logits_{i}' for i in range(len(instruments))]
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    return UNETSTFT(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        instruments=instruments,
        sess=generate_session(graph=g, **kwargs),
        model=model,
        name=module,
    )



def load(modelfile, model, module, extra, label, quantized=False, **kwargs):

    # path = check_file(
    #     file=model,
    #     module=module,
    #     keys={'model': 'model.pb'},
    #     quantized=quantized,
    #     **kwargs,
    # )
    g = load_graph(modelfile, **kwargs)

    vectorizer_mapping = {
        'vggvox-v1': featurization.vggvox_v1,
        'vggvox-v2': featurization.vggvox_v2,
        'deep-speaker': featurization.deep_speaker,
        'speakernet': featurization.SpeakerNetFeaturizer(
            **{**speakernet_config, **extra}
        ),
    }

    if module == 'speaker-vector':
        if model == 'speakernet':
            model_class = Speakernet
        else:
            model_class = Speaker2Vec
    else:
        if model == 'speakernet':
            model_class = SpeakernetClassification
        elif 'marblenet' in model:
            model_class = MarbleNetClassification
        else:
            model_class = Classification

    if model == 'speakernet':
        inputs = ['Placeholder', 'Placeholder_1']
    elif 'marblenet' in model:
        inputs = ['X_placeholder', 'X_len_placeholder']
    else:
        inputs = ['Placeholder']
    outputs = ['logits']

    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    return model_class(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        vectorizer=vectorizer_mapping.get(model),
        sess=generate_session(graph=g, **kwargs),
        model=model,
        extra=extra,
        label=label,
        name=module,
    )


def check_file(
    model,
    # package,
    base_path=pretrained_model_path,
    # s3_file=None,
    module=None,
    keys=None,
    validate=True,
    quantized=False,
    **kwargs,
):
    # home, _ = _get_home(package=package)
    # model = path
    keys = keys.copy()
    keys['version'] = 'version'
    
    path=''
    if quantized:
        path = os.path.join(module, f'{model}-quantized')
        logging.warning('Load quantized model will cause accuracy drop.')
    else:
        path = os.path.join(module, model)

    path_local = os.path.join(base_path, path)
    files_local = {'version': os.path.join(path_local, 'version')}
    
    for key, value in keys.items():
        if '/' in value:
            f_local = os.path.join(path_local, value.split('/')[-1])
        else:
            f_local = os.path.join(path_local, value)
        files_local[key] = f_local
    
    if validate:
        download = False
        version = files_local['version']

        if os.path.isfile(version):
            for key, item in files_local.items():
                if not os.path.exists(item):
                    download = True
                    break
        else:
            download = True

        if download:
            raise Exception(
                f'{model} model is not available. Please verify or download the pretrained model.'
            )
    else:
        if not check_files_local(files_local):
            path = files_local['model']
            path = os.path.sep.join(
                os.path.normpath(path).split(os.path.sep)[1:-1]
            )
            raise Exception(
                f'{path} is not available, please `validate = True`'
            )
    return files_local


def check_files_local(file):
    for key, item in file.items():
        if 'version' in key:
            continue
        if not os.path.isfile(item):
            return False
    return True  
    

def load_1d(modelfile, model, module, quantized=False, **kwargs):
    # path = check_file(
    #     file=model,
    #     module=module,
    #     keys={'model': 'model.pb'},
    #     quantized=quantized,
    #     **kwargs,
    # )
    g = load_graph(modelfile, **kwargs)
    inputs = ['Placeholder']
    outputs = ['logits']
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    return UNET1D(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        sess=generate_session(graph=g, **kwargs),
        model=model,
        name=module,
    )


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                        Class                                        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
class resemblyzer_VoiceEncoder(nn.Module):
    def __init__(self, device: Union[str, torch.device] = None, verbose=True,
                 weights_fpath: Union[Path, str] = None):

        super().__init__()

        # Define the network
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

        # Get the target device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Load the pretrained model'speaker weights
        if weights_fpath is None:
            weights_fpath = Path('__file__').resolve().parent.joinpath(pretrained_model_path,'resemblyzer/pretrained.pt')
        else:
            weights_fpath = Path(weights_fpath)

        if not weights_fpath.exists():
            raise Exception("Couldn't find the voice encoder pretrained model at %s." %
                            weights_fpath)
        start = timer()
        checkpoint = torch.load(weights_fpath, map_location="cpu")
        self.load_state_dict(checkpoint["model_state"], strict=False)
        self.to(device)

        if verbose:
            print("Loaded the voice encoder model on %s in %.2f seconds." %
                  (device.type, timer() - start))
    
    
    def forward(self, mels: torch.FloatTensor):

        # Pass the input through the LSTM layers and retrieve the final hidden state of the last
        # layer. Apply a cutoff to 0 for negative values and L2 normalize the embeddings.
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

    @staticmethod
    def compute_partial_slices(n_samples: int, rate, min_coverage):

        assert 0 < min_coverage <= 1

        # Compute how many frames separate two partial utterances
        samples_per_frame = int((sampling_rate * mel_window_step / 1000))
        n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
        frame_step = int(np.round((sampling_rate / rate) / samples_per_frame))
        assert 0 < frame_step, "The rate is too high"
        assert frame_step <= partials_n_frames, "The rate is too low, it should be %f at least" % \
                                                (sampling_rate / (samples_per_frame * partials_n_frames))

        # Compute the slices
        wav_slices, mel_slices = [], []
        steps = max(1, n_frames - partials_n_frames + frame_step + 1)
        for i in range(0, steps, frame_step):
            mel_range = np.array([i, i + partials_n_frames])
            wav_range = mel_range * samples_per_frame
            mel_slices.append(slice(*mel_range))
            wav_slices.append(slice(*wav_range))

        # Evaluate whether extra padding is warranted or not
        last_wav_range = wav_slices[-1]
        coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
        if coverage < min_coverage and len(mel_slices) > 1:
            mel_slices = mel_slices[:-1]
            wav_slices = wav_slices[:-1]

        return wav_slices, mel_slices

    def embed_utterance(self, wav: np.ndarray, return_partials=False, rate=1.3, min_coverage=0.75):
        def wav_to_mel_spectrogram(wav):
            frames = librosa.feature.melspectrogram(
                wav,
                sampling_rate,
                n_fft=int(sampling_rate * mel_window_length / 1000),
                hop_length=int(sampling_rate * mel_window_step / 1000),
                n_mels=mel_n_channels
            )
            return frames.astype(np.float32).T
        
        # Compute where to split the utterance into partials and pad the waveform with zeros if
        # the partial utterances cover a larger range.
        wav_slices, mel_slices = self.compute_partial_slices(len(wav), rate, min_coverage)
        max_wave_length = wav_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

        # Split the utterance into partials and forward them through the model
        mel = wav_to_mel_spectrogram(wav)
        mels = np.array([mel[s] for s in mel_slices])
        with torch.no_grad():
            mels = torch.from_numpy(mels).to(self.device)
            partial_embeds = self(mels).cpu().numpy()

        # Compute the utterance embedding from the partial embeddings
        raw_embed = np.mean(partial_embeds, axis=0)
        embed = raw_embed / np.linalg.norm(raw_embed, 2)

        if return_partials:
            return embed, partial_embeds, wav_slices
        return embed

    def embed_speaker(self, wavs: List[np.ndarray], **kwargs):

        raw_embed = np.mean([self.embed_utterance(wav, return_partials=False, **kwargs) \
                             for wav in wavs], axis=0)
        return raw_embed / np.linalg.norm(raw_embed, 2)

