import os
from tensorflow.keras.models import model_from_json
from analysis.se_1.data_tools import scaled_in, inv_scaled_ou
from analysis.se_1.data_tools import audio_files_to_numpy, numpy_audio_to_matrix_spectrogram, matrix_spectrogram_to_numpy_audio
from analysis.uob_init import(
    pretrained_model_path
)

class init_params():
    def __init__(self):
        self.nb_samples = 50
        self.weights_folder = os.path.join(pretrained_model_path,'speech-enhancement_new/weights')
        self.name_model = 'model_unet'
        self.min_duration = 0.5
        self.frame_length = 8064
        self.hop_length_frame = 8064
        self.n_fft = 255
        self.hop_length_fft = 63


def load_speechenhancement_model_local():
    # args = parser.parse_args()
    args = init_params()

    # mode = args.mode

    #path to find pre-trained weights / save models
    weights_path = args.weights_folder
    #pre trained model
    name_model = args.name_model

    # load json and create model
    json_file = open(weights_path+'/'+name_model+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_path+'/'+name_model+'.h5')
    print("Loaded SE model from disk")
    return loaded_model


def get_se_output(y, sr, se_model_new):
    """ This function takes as input pretrained weights, noisy voice sound to denoise, predict
    the denoise sound and save it to disk.
    """
    # args = parser.parse_args()
    args = init_params()
    # Minimum duration of audio files to consider
    min_duration = args.min_duration
    #Frame length for training data
    frame_length = args.frame_length
    # hop length for sound files
    hop_length_frame = args.hop_length_frame
    #nb of points for fft(for spectrogram computation)
    n_fft = args.n_fft
    #hop length for fft
    hop_length_fft = args.hop_length_fft

    # Extracting noise and voice from folder and convert to numpy
    audio = audio_files_to_numpy(y, sr,
                                 frame_length, hop_length_frame, min_duration)

    #Dimensions of squared spectrogram
    dim_square_spec = int(n_fft / 2) + 1
    print(dim_square_spec)

    # Create Amplitude and phase of the sounds
    m_amp_db_audio,  m_pha_audio = numpy_audio_to_matrix_spectrogram(
        audio, dim_square_spec, n_fft, hop_length_fft)

    #global scaling to have distribution -1/1
    X_in = scaled_in(m_amp_db_audio)
    #Reshape for prediction
    X_in = X_in.reshape(X_in.shape[0],X_in.shape[1],X_in.shape[2],1)
    #Prediction using loaded network
    X_pred = se_model_new.predict(X_in)
    #Rescale back the noise model
    inv_sca_X_pred = inv_scaled_ou(X_pred)
    #Remove noise model from noisy speech
    X_denoise = m_amp_db_audio - inv_sca_X_pred[:,:,:,0]
    #Reconstruct audio from denoised spectrogram and phase
    print(X_denoise.shape)
    print(m_pha_audio.shape)
    print(frame_length)
    print(hop_length_fft)
    audio_denoise_recons = matrix_spectrogram_to_numpy_audio(X_denoise, m_pha_audio, frame_length, hop_length_fft)
    #Number of frames
    nb_samples = audio_denoise_recons.shape[0]
    #Save all frames in one file
    denoise_long = audio_denoise_recons.reshape(1, nb_samples * frame_length)*10
    result = denoise_long[0, :]
    return result
    # sf.write(dir_save_prediction + audio_output_prediction, denoise_long[0, :], sample_rate)
    # librosa.output.write_wav(dir_save_prediction + audio_output_prediction, denoise_long[0, :], sample_rate)



# import argparse

# parser = argparse.ArgumentParser(description='Speech enhancement prediction')

# #How much frame to create in data_creation mode
# parser.add_argument('--nb_samples', default=50, type=int)
# #folder of saved weights
# parser.add_argument('--weights_folder', default=os.path.join(pretrained_model_path,'speech-enhancement_new/weights'), type=str)
# #Name of saved model to read
# parser.add_argument('--name_model', default='model_unet', type=str)
# # Minimum duration of audio files to consider
# parser.add_argument('--min_duration', default=0.5, type=float)
# # Training data will be frame of slightly above 1 second
# parser.add_argument('--frame_length', default=8064, type=int)
# # hop length for clean voice files separation (no overlap)
# parser.add_argument('--hop_length_frame', default=8064, type=int)
# # Choosing n_fft and hop_length_fft to have squared spectrograms
# parser.add_argument('--n_fft', default=255, type=int)

# parser.add_argument('--hop_length_fft', default=63, type=int)


