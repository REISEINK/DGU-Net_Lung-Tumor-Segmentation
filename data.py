import os
import numpy as np
import torch


def load_image(data_folder, status=None):
    x = np.squeeze(np.load(os.path.join(data_folder, 'x_' + status + '_64x64x32.npy')))
    y = np.squeeze(np.load(os.path.join(data_folder, 'm_' + status + '_64x64x32.npy')))
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)

    return torch.from_numpy(x), torch.from_numpy(y)


def load_image_tensor(data_folder):
    x_train, y_train = load_image(data_folder, 'train')
    x_valid, y_valid = load_image(data_folder, 'valid')

    # return np.concatenate((x_train, x_valid), 0), np.concatenate((y_train, y_valid), 0)
    x_total, y_total = np.concatenate((x_train, x_valid), 0), np.concatenate((y_train, y_valid), 0)
    return torch.from_numpy(x_total), torch.from_numpy(y_total)

def load_image_numpy(data_folder):
    x_train, y_train = load_image(data_folder, 'train')
    x_valid, y_valid = load_image(data_folder, 'valid')

    # return np.concatenate((x_train, x_valid), 0), np.concatenate((y_train, y_valid), 0)
    return np.concatenate((x_train, x_valid), 0), np.concatenate((y_train, y_valid), 0)
