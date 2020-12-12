import torch
import argparse
import os
import numpy as np
from torch.backends import cudnn
from model import model
import data.utils as data_utils
from utils.utils import to_cuda, mean_accuracy, accuracy
from data.custom_dataset_dataloader import CustomDatasetDataLoader
import sys
import pprint
from config.config import cfg, cfg_from_file, cfg_from_list
from math import ceil as ceil
from utils.utils import to_cuda, to_onehot,euclidean_dist

import h5py

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test script.')
    parser.add_argument('--adapted', dest='adapted_model',
                        type=bool,default=True,
                        help='if the model is adapted on target')

    parser.add_argument('--weights', dest='weights',
                        help='initialize with specified model parameters',
                        default='ckpt_best.weights', type=str)
    parser.add_argument('--resume', dest='resume',
                        help='initialize with saved solver status, e.g., ckpt_best.resume',
                        default='ckpt_best.resume', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default="experiments/config/VisDA-2017/visda17_test_val_cfg.yaml", type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--method', dest='method',
                        help='set the method to use',
                        default='CENTER_Source_only', type=str)#CENTER_KMEANS_SP,Source_only
    parser.add_argument('--data_root',
                        help='the dataset root dir path',
                        default='/data/DA_dataset', type=str)
    parser.add_argument('--exp_name', dest='exp_name',
                        help='the experiment name',
                        default='visda17_train2val_2048debug', type=str)


    args = parser.parse_args()
    return args

def save_preds(paths, preds, save_path, filename='preds.txt'):
    assert(len(paths) == preds.size(0))
    with open(os.path.join(save_path, filename), 'w') as f:
        for i in range(len(paths)):
            line = paths[i] + ' ' + str(preds[i].item()) + '\n'
            f.write(line)

def save_preds_gts(gts, preds, save_path, filename='preds_gts.txt'):
    assert(len(gts) == preds.size(0))
    with open(os.path.join(save_path, filename), 'w') as f:
        for i in range(len(gts)):
            line = gts[i] + ' ' + str(preds[i].item()) + '\n'
            f.write(line)

def prepare_data(args):
    test_transform = data_utils.get_transform(False)

    # source = cfg.DATASET.SOURCE_NAME
    target = cfg.TEST.DOMAIN

    # dataroot_S = os.path.join(args.data_root, cfg.DATASET.DATAROOT, source)
    dataroot_T = os.path.join(args.data_root, cfg.DATASET.DATAROOT, target)
    # dataroot_T = os.path.join(cfg.DATASET.DATAROOT, target)

    with open(os.path.join(args.data_root,cfg.DATASET.DATAROOT,'visual', 'category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    assert(len(classes) == cfg.DATASET.NUM_CLASSES)

    dataloader = None

    dataset_type = cfg.TEST.DATASET_TYPE
    batch_size = cfg.TEST.BATCH_SIZE
    dataloader = CustomDatasetDataLoader(dataset_root=dataroot_T, 
                dataset_type=dataset_type, batch_size=batch_size, 
                transform=test_transform, train=False, 
                num_workers=cfg.NUM_WORKERS, classnames=classes)

    return dataloader



def test(args):
    # prepare data
    dataloader = prepare_data(args)

    # initialize model
    model_state_dict = None
    fx_pretrained = True

    bn_domain_map = {}
    if cfg.WEIGHTS != '':
        save_dir = os.path.join(root_path, cfg.SAVE_DIR)
        path = os.path.join(save_dir, args.weights)
        # path=os.path.join(cfg.SAVE_DIR, cfg.WEIGHTS)
        weights_dict = torch.load(path)
        model_state_dict = weights_dict['weights']
        bn_domain_map = weights_dict['bn_domain_map']
        fx_pretrained = False

    if args.adapted_model:
        num_domains_bn = 2 
    else:
        num_domains_bn = 1

    net = model.danet(num_classes=cfg.DATASET.NUM_CLASSES, 
                 state_dict=model_state_dict,
                 feature_extractor=cfg.MODEL.FEATURE_EXTRACTOR, 
                 fx_pretrained=fx_pretrained, 
                 dropout_ratio=cfg.TRAIN.DROPOUT_RATIO,
                 fc_hidden_dims=cfg.MODEL.FC_HIDDEN_DIMS,
                 num_domains_bn=num_domains_bn) 

    net = torch.nn.DataParallel(net)

    if torch.cuda.is_available():
        net.cuda()

    # test 
    res = {}
    res['path'], res['preds'], res['gt'], res['probs'],res['feats'] = [], [], [], [],[]
    net.eval()

    if cfg.TEST.DOMAIN in bn_domain_map:
        domain_id = bn_domain_map[cfg.TEST.DOMAIN]
    else:
        domain_id = 0

    with torch.no_grad():
        net.module.set_bn_domain(domain_id)
        for sample in iter(dataloader): 
            res['path'] += sample['Path']

            if cfg.DATA_TRANSFORM.WITH_FIVE_CROP:
                n, ncrop, c, h, w = sample['Img'].size()
                sample['Img'] = sample['Img'].view(-1, c, h, w)
                img = to_cuda(sample['Img'])
                output = net(img)
                probs =output['probs']
                probs = probs.view(n, ncrop, -1).mean(dim=1)
                features=output['feat']
                features = features.view(n, ncrop, -1).mean(dim=1)
            else:
                img = to_cuda(sample['Img'])
                output = net(img)
                probs = output['probs']
                features = output['feat'].data

            preds = torch.max(probs, dim=1)[1]
            res['preds'] += [preds]
            res['probs'] += [probs]
            res['feats'] += [features]

            if 'Label' in sample:
                label = to_cuda(sample['Label'])
                res['gt'] += [label] 
            print('Processed %d samples.' % len(res['path']))

        preds = torch.cat(res['preds'], dim=0)
        feats = torch.cat(res['feats'], dim=0)
        save_dir = os.path.join(root_path, cfg.SAVE_DIR)
        save_preds(res['path'], preds, save_dir)


        if 'gt' in res and len(res['gt']) > 0:
            gts = torch.cat(res['gt'], dim=0)
            probs = torch.cat(res['probs'], dim=0)

            f=h5py.File(os.path.join(save_dir,'src_data.h5'),'w')
            f['feat']=feats.cpu().data
            f['gts']=gts.cpu().data
            f['preds']=preds.cpu().data
            f.close()
            assert(cfg.EVAL_METRIC == 'mean_accu' or cfg.EVAL_METRIC == 'accuracy')
            if cfg.EVAL_METRIC == "mean_accu": 
                eval_res,eval_list = mean_accuracy(probs, gts)
                print('Test mean_accu: %.4f' % (eval_res))
                print('Each category acc:{}'.format(eval_list))
                out_str = "Test mean:accuracy:{}. each category accuracy:{}\n".format(eval_res,eval_list)
                with open(os.path.join(save_dir, 'result_acc.txt'), 'a') as f:
                    f.write(out_str)

            elif cfg.EVAL_METRIC == "accuracy":
                eval_res = accuracy(probs, gts)
                print('Test accuracy: %.4f' % (eval_res))

    print('Finished!')

if __name__ == '__main__':
    cudnn.benchmark = True 
    args = parse_args()
    root_path = os.path.abspath(os.path.dirname(__file__))
    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(os.path.join(root_path, args.cfg_file))
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    if args.weights is not None:
        save_dir = os.path.join(root_path, cfg.SAVE_DIR, cfg.EXP_NAME)
        cfg.WEIGHTS = os.path.join(save_dir, args.weights)
    if args.exp_name is not None:
        cfg.EXP_NAME = args.exp_name + "_" + args.method
    if args.resume is not None:
        save_dir = os.path.join(root_path, cfg.SAVE_DIR, cfg.EXP_NAME)
        cfg.RESUME = os.path.join(save_dir, args.resume)

    print('Using config:')
    pprint.pprint(cfg)

    cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, cfg.EXP_NAME)
    print('Output will be saved to %s.' % cfg.SAVE_DIR)

    # visiualization()
    test(args)
