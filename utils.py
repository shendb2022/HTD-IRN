import numpy as np
import os


def getPSNR(x, y):
    max_val = 1.0
    mse = np.mean(np.square(x - y))
    psnr = 10 * np.log10(max_val / mse + 1e-8)
    return psnr


def getSAM(x, y):
    x_y = np.sum(np.multiply(x, y), axis=1)
    x_norm = np.sqrt(np.sum(x ** 2, axis=1))
    y_norm = np.sqrt(np.sum(y ** 2, axis=1))

    cosin_value = x_y / (x_norm * y_norm + 1e-8)
    cosin_value = np.arccos(np.clip(cosin_value, -1, 1))
    angle = cosin_value / np.pi * 180

    return np.mean(angle)


def checkFile(path):
    '''
    if filepath not exist make it
    :param path:
    :return:
    '''
    if not os.path.exists(path):
        os.makedirs(path)


def standard(X):
    max_value = np.max(X)
    min_value = np.min(X)
    if max_value == min_value:
        return X
    return (X - min_value) / (max_value - min_value)
