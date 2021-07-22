import math
import numpy as np

# input parameters
input_len = 2500
dataset = 'LSST'


def get_conv1d_out_len(in_len, padding, kernel, stride):
    return math.trunc(((in_len + 2 * padding - 1 * (kernel - 1) - 1) / stride) + 1)


def get_maxpool_out_len(in_len, padding, kernel, stride):
    return math.trunc(((in_len + 2 * padding - 1 * (kernel - 1) - 1) / stride) + 1)


def get_features_len(configs, in_len):

    # first out_len
    in_len_1 = in_len
    kernel_1 = configs.kernel_size
    stride_1 = configs.stride
    pading_1 = configs.kernel_size//2

    out_len_1 = get_conv1d_out_len(in_len_1, pading_1, kernel_1, stride_1)
    out_len_1 = get_maxpool_out_len(out_len_1, 1, 2, 2)

    in_len_2 = out_len_1
    kernel_2 = 8
    stride_2 = 1
    pading_2 = 4

    out_len_2 = get_conv1d_out_len(in_len_2, pading_2, kernel_2, stride_2)
    out_len_2 = get_maxpool_out_len(out_len_2, 1, 2, 2)

    in_len_3 = out_len_2
    kernel_3 = 8
    stride_3 = 1
    pading_3 = 4

    out_len_3 = get_conv1d_out_len(in_len_3, pading_3, kernel_3, stride_3)
    out_len_3 = get_maxpool_out_len(out_len_3, 1, 2, 2)

    feature_len = out_len_3
    return feature_len


def generate_config(dataset, input_len, n_channels, n_classes):

    from config_files.template import Config
    configs = Config()

    with open('template.py.txt') as f:
        contents = f.read()
        # print(contents)
        features_len = get_features_len(configs, input_len)

        contents = contents.replace(r'$input_channels', str(n_channels))
        contents = contents.replace(r'$final_out_channels', str(128))
        contents = contents.replace(r'$num_classes', str(n_classes))
        contents = contents.replace(r'$features_len', str(features_len))
        contents = contents.replace(r'$timesteps', str(int(features_len / 2)))

        contents = contents.replace(r'$num_epoch', str(100))
        contents = contents.replace(r'$num_ssl_epoch', str(300))
        contents = contents.replace(r'$timesteps', str(int(features_len / 2)))


        text_file = open(f"{dataset}_Configs.py", "w")
        text_file.write(contents)
        text_file.close()


if __name__ == '__main__':

    datasets = ['ArticularyWordRecognition', 'AtrialFibrilation', 'BasicMotions',
    'CharacterTrajectories', 'Cricket', 'DuckDuckGeese', 'EigenWorms',
    'Epilepsy', 'ERing', 'EthanolConcentration', 'FaceDetection',
    'FingerMovements', 'HandMovementDirection', 'Handwriting',
    'Heartbeat', 'InsectWingbeat', 'JapaneseVowels', 'Libras',
    'LSST', 'MotorImagery', 'NATOPS', 'PEMS-SF', 'PenDigits',
    'Phoneme', 'RacketSports', 'SelfRegulationSCP1', 'SelfRegulationSCP2', \
    'SpokenArabicDigits', 'StandWalkJump', 'UWaveGestureLibrary']

    for dataset in datasets:
        path = f'../data/{dataset}/'

        x_train = np.load(path + 'X_train.npy')
        y_train = np.load(path + 'y_train.npy')

        input_len = x_train.shape[1]
        n_channels = x_train.shape[2]
        n_classes = int(np.amax(y_train)) + 1

        generate_config(dataset, input_len, n_channels, n_classes)