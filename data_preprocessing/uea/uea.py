import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


def get_channel_range(d):
    list_channel_range = []
    for c in range(d.shape[2]):
        cmin = d[:, :, c].min()
        cmax = d[:, :, c].max()
        list_channel_range.append((cmin, cmax))
    return list_channel_range


def proprocess_dataset(dataset = 'LSST', scale = 'minmax_channel'):

    path = f'../../data/{dataset}/'

    x_train = np.load(path + 'X_train.npy')
    y_train = np.load(path + 'y_train.npy')
    x_test = np.load(path + 'X_test.npy')
    y_test = np.load(path + 'y_test.npy')

    x_train = x_train.swapaxes(1, 2)
    x_test = x_test.swapaxes(1, 2)

    l_rng_train = get_channel_range(x_train)
    l_rng_test = get_channel_range(x_test)

    nclass = int(np.amax(y_train)) + 1
    ntimestep = x_train.shape[2]
    nfeature = x_train.shape[1]

    print(f"{dataset} class : {nclass}, ntimestep : {ntimestep}, nfeature : {nfeature}")
    print(f"train channel range : {l_rng_train}")
    print(f"test channel range : {l_rng_test}")

    if scale == 'standard_channel':
        for c in range(nfeature):
            scaler = StandardScaler()
            x_train[:, c, :] = scaler.fit_transform(x_train[:, c, :])
            x_test[:, c, :] = scaler.transform(x_test[:, c, :])
    elif scale == 'minmax_channel':
        for c in range(nfeature):
            scaler = MinMaxScaler()
            x_train[:, c, :] = scaler.fit_transform(x_train[:, c, :])
            x_test[:, c, :] = scaler.transform(x_test[:, c, :])
    elif scale == 'standard_all':
        scaler = StandardScaler()
        origin_shape = x_train.shape
        x_train = x_train.reshape(-1, 1)
        x_train = scaler.fit_transform(x_train)
        x_train = x_train.reshape(origin_shape)
        origin_shape = x_test.shape
        x_test = x_test.reshape(-1, 1)
        x_test = scaler.transform(x_test)
        x_test = x_test.reshape(origin_shape)
    elif scale == 'minmax_all':
        scaler = MinMaxScaler()
        origin_shape = x_train.shape
        x_train = x_train.reshape(-1, 1)
        x_train = scaler.fit_transform(x_train)
        x_train = x_train.reshape(origin_shape)
        origin_shape = x_test.shape
        x_test = x_test.reshape(-1, 1)
        x_test = scaler.transform(x_test)
        x_test = x_test.reshape(origin_shape)

    l_rng_train = get_channel_range(x_train)
    l_rng_test = get_channel_range(x_test)

    print(f"after train channel range : {l_rng_train}")
    print(f"after test channel range : {l_rng_test}")

    dat_dict = dict()
    dat_dict["samples"] = torch.from_numpy(x_train)
    dat_dict["labels"] = torch.from_numpy(y_train)
    torch.save(dat_dict, f"../../data/{dataset}/train.pt")

    dat_dict = dict()
    dat_dict["samples"] = torch.from_numpy(x_test)
    dat_dict["labels"] = torch.from_numpy(y_test)
    torch.save(dat_dict, f"../../data/{dataset}/test.pt")


if __name__ == '__main__':

    datasets = ['ArticularyWordRecognition', 'AtrialFibrilation', 'BasicMotions',
    'CharacterTrajectories', 'Cricket', 'DuckDuckGeese', 'EigenWorms',
    'Epilepsy', 'ERing', 'EthanolConcentration', 'FaceDetection',
    'FingerMovements', 'HandMovementDirection', 'Handwriting',
    'Heartbeat', 'InsectWingbeat', 'JapaneseVowels', 'Libras',
    'LSST', 'MotorImagery', 'NATOPS', 'PEMS-SF', 'PenDigits',
    'Phoneme', 'RacketSports', 'SelfRegulationSCP1', 'SelfRegulationSCP2', \
    'SpokenArabicDigits', 'StandWalkJump', 'UWaveGestureLibrary']

    scale = 'minmax_all'  # 'minmax_channel'

    for dataset in datasets:
        proprocess_dataset(dataset, scale)
