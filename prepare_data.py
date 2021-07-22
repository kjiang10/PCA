from numpy.lib.type_check import common_type
from sklearn.utils.validation import column_or_1d
import torch
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import os
import copy

from scipy.io import arff
import numpy as np
import pandas as pd
import functools
from sklearn.base import TransformerMixin
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def format_file_name(file_number):
    length = len(str(file_number))
    return '0' * (6 - length) + str(file_number)

def vectorize_image(num_samples):
    output = []
    orig_images = parse_image(num_samples, True)
    for img in orig_images:
        new_image = copy.copy(img)
        img_shape = new_image.shape
        new_image = new_image.reshape(img_shape[0] * img_shape[1] * img_shape[2])
        output.append(new_image)
    output = np.vstack(output)
    return output

def parse_image(num_samples, convert):
    file_path = 'datasets/img_align_celeba/'
    # print(num_files)
    output = []
    for i in range(1, num_samples + 1):
        img = Image.open(file_path + format_file_name(i) + ".jpg")
        if convert:
            img = np.array(img)
        output.append(img)
    return output

def get_celebA_x(idx, num_samples):
    orig_images = parse_image(num_samples, False)
    output = []
    for i in idx:
        output.append(orig_images[i])
    return output

def get_celebA_ya(y_col, a_col, idx, num_samples):
    data = []
    with open('datasets/list_attr_celeba.txt', 'r') as f:
        for i, l  in enumerate(f):
            # should exclude the first two lines
            if i >= num_samples + 2:
                break
            if i >= 2:
                lb = [int(i) for i in " ".join(l.split()).strip().split(" ")[1:]]
                lb = [0 if i == -1 else 1 for i in lb]
                data.append(lb)
            
    # pick the colth attribute as sensitive attribute

    # print(idx)
    col_idx = np.array([y_col, a_col])
    # print(idx, col_idx)

    data_ya = np.asarray(data)[idx][:, col_idx]
    # print(data_ya)
    return data_ya
