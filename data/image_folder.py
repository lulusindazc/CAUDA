###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path
import collections
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset_with_labels(dir, classnames):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    images = []
    labels = []

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            dirname = os.path.split(root)[-1]
            if dirname not in classnames:
                continue

            label = classnames.index(dirname)

            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                labels.append(label)

    return images, labels

# def make_dataset_with_list_labels(dir, txt_list):
#     assert os.path.isdir(dir), '%s is not a valid directory' % dir
#
#     images_file_path = os.path.join(dir,txt_list)
#     image_list=open(images_file_path).readlines()
#     if len(image_list[0].split()) > 2:
#         images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
#     else:
#         images = [(val.split()[0], int(val.split()[1])) for val in image_list]
#
#     return images[:,0], images[:,1]
#     # for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
#     #     for fname in fnames:
#     #         dirname = os.path.split(root)[-1]
#     #         if dirname not in classnames:
#     #             continue
#     #
#     #         label = classnames.index(dirname)
#     #
#     #         if is_image_file(fname):
#     #             path = os.path.join(root, fname)
#     #             images.append(path)
#     #             labels.append(label)
#     #
#     # return images, labels
 
def make_dataset_classwise(dir, category):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    images = []
    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            dirname = os.path.split(root)[-1]
            if dirname != category:
                continue
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images
