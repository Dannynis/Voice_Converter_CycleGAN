import os
import numpy as np
import argparse
import time
import librosa
import tensorflow as tf

from preprocess import *
from model import CycleGAN

processed_data_dir = './processed_data'


def load_speaker_features(file_path):

    mcep_params = np.load(file_path, allow_pickle=True)
    f0s = mcep_params['f0s']
    timeaxes = mcep_params['timeaxes']
    sps = mcep_params['sps']
    aps = mcep_params['aps']
    coded_sps = mcep_params['coded_sps']
    return f0s,timeaxes,sps,aps,coded_sps

def train(train_A_dir, train_B_dir, model_dir, model_name, random_seed, validation_A_dir, validation_B_dir, output_dir, tensorboard_log_dir):

    np.random.seed(random_seed)

    num_epochs = 5000
    mini_batch_size = 1 # mini_batch_size = 1 is better
    generator_learning_rate = 0.0002
    generator_learning_rate_decay = generator_learning_rate / 200000
    discriminator_learning_rate = 0.000001
    discriminator_learning_rate_decay = discriminator_learning_rate / 200000
    sampling_rate = 16000
    num_mcep = 24
    frame_period = 5.0
    n_frames = 64
    lambda_cycle = 10
    lambda_identity = 5

    Speaker_A_features = os.path.join(processed_data_dir, 'wav_A.npz')
    Speaker_B_features = os.path.join(processed_data_dir, 'wav_B.npz')
    start_time = time.time()
    if os.path.exists(Speaker_A_features) and os.path.exists(Speaker_B_features):
        print ('#### loading processed data #######')
        f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = load_speaker_features(Speaker_A_features)
        f0s_B, timeaxes_B, sps_B, aps_B, coded_sps_B = load_speaker_features(Speaker_B_features)
    else:
        print('Preprocessing Data...')



        wavs_A = load_wavs(wav_dir = train_A_dir, sr = sampling_rate)
        wavs_B = load_wavs(wav_dir = train_B_dir, sr = sampling_rate)

        f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = world_encode_data(wavs = wavs_A, fs = sampling_rate, frame_period = frame_period, coded_dim = num_mcep)
        f0s_B, timeaxes_B, sps_B, aps_B, coded_sps_B = world_encode_data(wavs = wavs_B, fs = sampling_rate, frame_period = frame_period, coded_dim = num_mcep)
        if not os.path.exists(processed_data_dir):
            os.makedirs(processed_data_dir)

        np.savez(Speaker_A_features, f0s = f0s_A, timeaxes = timeaxes_A, sps = sps_A, aps = aps_A,coded_sps=coded_sps_A)
        np.savez(Speaker_B_features, f0s = f0s_B, timeaxes = timeaxes_B, sps = sps_B, aps = aps_B,coded_sps=coded_sps_B)


    log_f0s_mean_A, log_f0s_std_A = logf0_statistics(f0s_A)
    log_f0s_mean_B, log_f0s_std_B = logf0_statistics(f0s_B)

    print('Log Pitch A')
    print('Mean: %f, Std: %f' %(log_f0s_mean_A, log_f0s_std_A))
    print('Log Pitch B')
    print('Mean: %f, Std: %f' %(log_f0s_mean_B, log_f0s_std_B))


    coded_sps_A_transposed = transpose_in_list(lst = coded_sps_A)
    coded_sps_B_transposed = transpose_in_list(lst = coded_sps_B)

    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = coded_sps_normalization_fit_transoform(coded_sps = coded_sps_A_transposed)
    print("Input data fixed.")
    coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std = coded_sps_normalization_fit_transoform(coded_sps = coded_sps_B_transposed)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    np.savez(os.path.join(model_dir, 'logf0s_normalization.npz'), mean_A = log_f0s_mean_A, std_A = log_f0s_std_A, mean_B = log_f0s_mean_B, std_B = log_f0s_std_B)
    np.savez(os.path.join(model_dir, 'mcep_normalization.npz'), mean_A = coded_sps_A_mean, std_A = coded_sps_A_std, mean_B = coded_sps_B_mean, std_B = coded_sps_B_std)

    if validation_A_dir is not None:
        validation_A_output_dir = os.path.join(output_dir, 'converted_A')
        if not os.path.exists(validation_A_output_dir):
            os.makedirs(validation_A_output_dir)

    if validation_B_dir is not None:
        validation_B_output_dir = os.path.join(output_dir, 'converted_B')
        if not os.path.exists(validation_B_output_dir):
            os.makedirs(validation_B_output_dir)

    end_time = time.time()
    time_elapsed = end_time - start_time

    print('Preprocessing Done.')

    print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))

    model = CycleGAN(num_features = num_mcep,log_dir=tensorboard_log_dir)

    for epoch in range(num_epochs):
        print('Epoch: %d' % epoch)
        '''
        if epoch > 60:
            lambda_identity = 0
        if epoch > 1250:
            generator_learning_rate = max(0, generator_learning_rate - 0.0000002)
            discriminator_learning_rate = max(0, discriminator_learning_rate - 0.0000001)
        '''

        start_time_epoch = time.time()

        dataset_A, dataset_B = sample_train_data(dataset_A = coded_sps_A_norm, dataset_B = coded_sps_B_norm, n_frames = n_frames)

        n_samples = dataset_A.shape[0]

        for i in range(n_samples // mini_batch_size):

            num_iterations = n_samples // mini_batch_size * epoch + i

            if num_iterations > 10000:
                lambda_identity = 0
            if num_iterations > 200000:
                generator_learning_rate = max(0, generator_learning_rate - generator_learning_rate_decay)
                discriminator_learning_rate = max(0, discriminator_learning_rate - discriminator_learning_rate_decay)

            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size

            optimizer_generator = tf.optimizers.Adam(generator_learning_rate)
            optimizer_discriminator = tf.optimizers.Adam(discriminator_learning_rate)

            # Wrap computation inside a GradientTape for automatic differentiation.
            with tf.GradientTape() as g:

                generator_loss, discriminator_loss = model.train(input_A = dataset_A[start:end], input_B = dataset_B[start:end], lambda_cycle = lambda_cycle,lambda_identity = lambda_identity)


                generator_trainable_variables = tf.concat (model.generator_A2B.trainable_variables,model.generator_B2A.trainable_variables)
                discriminator_trainable_variables = tf.concat (model.discriminator_A.trainable_variables,model.discriminator_B.trainable_variables)

                # Compute gradients.
                generator_gradients = g.gradient(generator_loss, generator_trainable_variables)
                discriminator_gradients = g.gradient(discriminator_loss, discriminator_trainable_variables)


                # Update W and b following gradients.
                optimizer_generator.apply_gradients(zip(generator_gradients, generator_trainable_variables))
                optimizer_generator.apply_gradients(zip(discriminator_gradients, discriminator_trainable_variables))



            if i % 50 == 0:
                #print('Iteration: %d, Generator Loss : %f, Discriminator Loss : %f' % (num_iterations, generator_loss, discriminator_loss))
                print('Iteration: {:07d}, Generator Learning Rate: {:.7f}, Discriminator Learning Rate: {:.7f}, Generator Loss : {:.3f}, Discriminator Loss : {:.3f}'.format(num_iterations, generator_learning_rate, discriminator_learning_rate, generator_loss, discriminator_loss))

        model.save(directory = model_dir, filename = model_name)

        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch

        print('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1)))

        if validation_A_dir is not None:
            if epoch % 50 == 0:
                print('Generating Validation Data B from A...')
                for file in os.listdir(validation_A_dir):
                    filepath = os.path.join(validation_A_dir, file)
                    wav, _ = librosa.load(filepath, sr = sampling_rate, mono = True)
                    wav = wav_padding(wav = wav, sr = sampling_rate, frame_period = frame_period, multiple = 4)
                    f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)
                    f0_converted = pitch_conversion(f0 = f0, mean_log_src = log_f0s_mean_A, std_log_src = log_f0s_std_A, mean_log_target = log_f0s_mean_B, std_log_target = log_f0s_std_B)
                    coded_sp = world_encode_spectral_envelop(sp = sp, fs = sampling_rate, dim = num_mcep)
                    coded_sp_transposed = coded_sp.T
                    coded_sp_norm = (coded_sp_transposed - coded_sps_A_mean) / coded_sps_A_std
                    coded_sp_converted_norm = model.test(inputs = np.array([coded_sp_norm]), direction = 'A2B')[0]
                    coded_sp_converted = coded_sp_converted_norm * coded_sps_B_std + coded_sps_B_mean
                    coded_sp_converted = coded_sp_converted.T
                    coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                    decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)
                    wav_transformed = world_speech_synthesis(f0 = f0_converted, decoded_sp = decoded_sp_converted, ap = ap, fs = sampling_rate, frame_period = frame_period)
                    librosa.output.write_wav(os.path.join(validation_A_output_dir, os.path.basename(file)), wav_transformed, sampling_rate)

        if validation_B_dir is not None:
            if epoch % 50 == 0:
                print('Generating Validation Data A from B...')
                for file in os.listdir(validation_B_dir):
                    filepath = os.path.join(validation_B_dir, file)
                    wav, _ = librosa.load(filepath, sr = sampling_rate, mono = True)
                    wav = wav_padding(wav = wav, sr = sampling_rate, frame_period = frame_period, multiple = 4)
                    f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)
                    f0_converted = pitch_conversion(f0 = f0, mean_log_src = log_f0s_mean_B, std_log_src = log_f0s_std_B, mean_log_target = log_f0s_mean_A, std_log_target = log_f0s_std_A)
                    coded_sp = world_encode_spectral_envelop(sp = sp, fs = sampling_rate, dim = num_mcep)
                    coded_sp_transposed = coded_sp.T
                    coded_sp_norm = (coded_sp_transposed - coded_sps_B_mean) / coded_sps_B_std
                    coded_sp_converted_norm = model.test(inputs = np.array([coded_sp_norm]), direction = 'B2A')[0]
                    coded_sp_converted = coded_sp_converted_norm * coded_sps_A_std + coded_sps_A_mean
                    coded_sp_converted = coded_sp_converted.T
                    coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                    decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)
                    wav_transformed = world_speech_synthesis(f0 = f0_converted, decoded_sp = decoded_sp_converted, ap = ap, fs = sampling_rate, frame_period = frame_period)
                    librosa.output.write_wav(os.path.join(validation_B_output_dir, os.path.basename(file)), wav_transformed, sampling_rate)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Train CycleGAN model for datasets.')

    train_A_dir_default = r"D:\ml+dl+dsp\Voice_Converter_CycleGAN\train\trump"
    train_B_dir_default = r"D:\ml+dl+dsp\Voice_Converter_CycleGAN\train\gaga"
    model_dir_default = './model/sf1_tf2'
    model_name_default = 'sf1_tf2.ckpt'
    random_seed_default = 0
    validation_A_dir_default = './data/evaluation_all/SF1'
    validation_B_dir_default = './data/evaluation_all/TF2'
    output_dir_default = './validation_output'
    tensorboard_log_dir_default = './log'

    parser.add_argument('--train_A_dir', type = str, help = 'Directory for A.', default = train_A_dir_default)
    parser.add_argument('--train_B_dir', type = str, help = 'Directory for B.', default = train_B_dir_default)
    parser.add_argument('--model_dir', type = str, help = 'Directory for saving models.', default = model_dir_default)
    parser.add_argument('--model_name', type = str, help = 'File name for saving model.', default = model_name_default)
    parser.add_argument('--random_seed', type = int, help = 'Random seed for model training.', default = random_seed_default)
    parser.add_argument('--validation_A_dir', type = str, help = 'Convert validation A after each training epoch. If set none, no conversion would be done during the training.', default = validation_A_dir_default)
    parser.add_argument('--validation_B_dir', type = str, help = 'Convert validation B after each training epoch. If set none, no conversion would be done during the training.', default = validation_B_dir_default)
    parser.add_argument('--output_dir', type = str, help = 'Output directory for converted validation voices.', default = output_dir_default)
    parser.add_argument('--tensorboard_log_dir', type = str, help = 'TensorBoard log directory.', default = tensorboard_log_dir_default)

    argv = parser.parse_args()

    train_A_dir = argv.train_A_dir
    train_B_dir = argv.train_B_dir
    model_dir = argv.model_dir
    model_name = argv.model_name
    random_seed = argv.random_seed
    validation_A_dir = None if argv.validation_A_dir == 'None' or argv.validation_A_dir == 'none' else argv.validation_A_dir
    validation_B_dir = None if argv.validation_B_dir == 'None' or argv.validation_B_dir == 'none' else argv.validation_B_dir
    output_dir = argv.output_dir
    tensorboard_log_dir = argv.tensorboard_log_dir

    train(train_A_dir = train_A_dir, train_B_dir = train_B_dir, model_dir = model_dir, model_name = model_name, random_seed = random_seed, validation_A_dir = validation_A_dir, validation_B_dir = validation_B_dir, output_dir = output_dir, tensorboard_log_dir = tensorboard_log_dir)
