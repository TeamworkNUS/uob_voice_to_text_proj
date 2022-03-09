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
                            quantized=False,
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
                                quantized=False,
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
                                    quantized=False,
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
    module='speaker_change'
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
                                    quantized=False,
                                    **kwargs,
                                )
    print(path_dict)
    
    
    settings = {
        'vggvox-v2': {'hop_length': 50, 'concat': False, 'mode': 'eval'},
        'speakernet': {'frame_ms': 20, 'stride_ms': 2},
    }

    model_object = load(
        model=model,
        module=module,
        extra=settings[model],
        label=[False, True],
        quantized=quantized,
        **kwargs
    )
    
    return model_object


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
    



# ? Test
# get_model_noisereduce(model='xxxx')
# get_model_vad(quantized=True)

