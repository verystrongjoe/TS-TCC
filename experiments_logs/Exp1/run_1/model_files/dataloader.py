import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from .augmentations import DataTransform


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]
        if training_mode == "self_supervised":  # no need to apply Augmentations in other modes
            self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        if self.training_mode == "self_supervised":
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len


def data_generator(args, data_path, configs, training_mode):

    if 'StandWalkJump' in data_path or 'LSST' in data_path:
        device = torch.device(args.device)
        path = 'data/LSST/'
        x_train = np.load(path + 'X_train.npy')
        y_train = np.load(path + 'y_train.npy')
        x_test = np.load(path + 'X_test.npy')
        y_test = np.load(path + 'y_test.npy')

        x_train = x_train.swapaxes(1,2)
        x_test = x_test.swapaxes(1,2)

        train_dataset, valid_dataset, test_dataset = {}, {}, {}
        train_dataset['samples'] = torch.as_tensor(x_train, dtype=torch.float32)
        train_dataset['labels'] = torch.as_tensor(y_train, dtype=torch.float32)

        valid_dataset['samples'] = torch.as_tensor(x_test, dtype=torch.float32)
        valid_dataset['labels'] = torch.as_tensor(y_test, dtype=torch.float32)

        test_dataset['samples'] = torch.as_tensor(x_test, dtype=torch.float32)
        test_dataset['labels'] = torch.as_tensor(y_test, dtype=torch.float32)

        train_dataset = Load_Dataset(train_dataset, configs, training_mode)
        valid_dataset = Load_Dataset(valid_dataset, configs, training_mode)
        test_dataset = Load_Dataset(test_dataset, configs, training_mode)

    else:
        train_dataset = torch.load(os.path.join(data_path, "train.pt"))  # (9200, 1, 178)
        valid_dataset = torch.load(os.path.join(data_path, "train.pt"))  # val.pt
        test_dataset = torch.load(os.path.join(data_path, "test.pt"))

        train_dataset = Load_Dataset(train_dataset, configs, training_mode)
        valid_dataset = Load_Dataset(valid_dataset, configs, training_mode)
        test_dataset = Load_Dataset(test_dataset, configs, training_mode)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=configs.drop_last,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader