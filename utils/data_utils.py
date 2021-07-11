import numpy as np


def cut_384(img):
    """
    cut a 512*512 ct img to 385*384
    :param img:
    :return:
    """
    if len(img.shape) > 2:
        ret = img[:, 50:434, 60:444]
    else:
        ret = img[50:434, 60:444]
    return ret


def pad_512(img):
    if len(img.shape) > 2:
        ret = np.pad(img, ((0, 0), (50, 78), (60, 68)), 'constant')
    else:
        ret = np.pad(img, ((50, 78), (60, 68)), 'constant')
    return ret


def window_standardize(img, lower_bound, upper_bound):
    """
    clip the pixel values into [lower_bound, upper_bound], and standardize them
    """
    img = np.clip(img, lower_bound, upper_bound)
    # x=x*2-1: map x to [-1,1]
    img = 2 * (img - lower_bound) / (upper_bound - lower_bound) - 1
    return img
