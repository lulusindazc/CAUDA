import torchvision.transforms as transforms
from PIL import Image
import torch
from config.config import cfg
from torch import  distributions
import os
import pickle

def get_transform(train=True):
    transform_list = []
    if cfg.DATA_TRANSFORM.RESIZE_OR_CROP == 'resize_and_crop':
        osize = [cfg.DATA_TRANSFORM.LOADSIZE, cfg.DATA_TRANSFORM.LOADSIZE]
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        if train:
            transform_list.append(transforms.RandomCrop(cfg.DATA_TRANSFORM.FINESIZE))
        else:
            if cfg.DATA_TRANSFORM.WITH_FIVE_CROP:
                transform_list.append(transforms.FiveCrop(cfg.DATA_TRANSFORM.FINESIZE)) 
            else:
                transform_list.append(transforms.CenterCrop(cfg.DATA_TRANSFORM.FINESIZE))

    elif cfg.DATA_TRANSFORM.RESIZE_OR_CROP == 'crop':
        if train:
            transform_list.append(transforms.RandomCrop(cfg.DATA_TRANSFORM.FINESIZE))
        else:
            if cfg.DATA_TRANSFORM.WITH_FIVE_CROP:
                transform_list.append(transforms.FiveCrop(cfg.DATA_TRANSFORM.FINESIZE)) 
            else:
                transform_list.append(transforms.CenterCrop(cfg.DATA_TRANSFORM.FINESIZE))
    elif cfg.DATA_TRANSFORM.RESIZE_OR_CROP == 'resize':
        osize = [cfg.DATA_TRANSFORM.LOADSIZE, cfg.DATA_TRANSFORM.LOADSIZE]
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
    if train and cfg.DATA_TRANSFORM.FLIP:
        transform_list.append(transforms.RandomHorizontalFlip())

    to_normalized_tensor = [transforms.ToTensor(),
                            transforms.Normalize(mean=cfg.DATA_TRANSFORM.NORMALIZE_MEAN,
                                       std=cfg.DATA_TRANSFORM.NORMALIZE_STD)]

    if not train and cfg.DATA_TRANSFORM.WITH_FIVE_CROP:
        transform_list += [transforms.Lambda(lambda crops: torch.stack([
                transforms.Compose(to_normalized_tensor)(crop) for crop in crops]))]
    else:
        transform_list += to_normalized_tensor

    return transforms.Compose(transform_list)

def save_samples_probs_pslabel(cfg_n,probs,plabels,epoch,name):
    result_folder = os.path.join(cfg_n.SAVE_DIR, 'result')
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    pickle.dump({'probs': probs, 'labels': plabels }, \
        open(os.path.join(result_folder, str(epoch) + '_' + name + '.pkl'), 'wb'))

def load_trg_plabels(cfg_n,epoch,name):
    result_folder = os.path.join(cfg_n.SAVE_DIR, 'result')
    file_path=os.path.join(result_folder, str(epoch) + '_' + name + '.pkl')
    pdata = pickle.load(open(file_path, 'rb'), encoding='bytes')

    probs = pdata['probs']
    plabels = pdata['labels']
    return probs, plabels

# def metropilis_hastings_sampling(path):
#     u=distributions.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))#