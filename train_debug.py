import torch
import argparse
import os
import numpy as np
from torch.backends import cudnn
from model import model
from config.config import cfg, cfg_from_file, cfg_from_list
from data.prepare_data import *
import sys
import pprint
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import random

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train script.')
    parser.add_argument('--weights', dest='weights',
                        help='initialize with specified model parameters,e.g., ckpt_best.weights',
                        default=None, type=str)
    parser.add_argument('--resume', dest='resume',
                        help='initialize with saved solver status, e.g., ckpt_best.resume',
                        default=None,type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default="experiments/config/Office-31/C2C/office31_train_amazon2webcam_cfg.yaml", type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--method', dest='method',
                        help='set the method to use', 
                        default='CENTER_KMEANS_SP', type=str)
    parser.add_argument('--backbone', dest='backbone',
                        help='set the network backbone to use',
                        default='resnet', type=str)
    parser.add_argument('--exp_name', dest='exp_name',
                        help='the experiment name', 
                        default='office31_a2w_2048debug_noCF', type=str)
    parser.add_argument('--data_root',
                        help='the dataset root dir path',

    args = parser.parse_args()


    return args

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) #cpu
    torch.cuda.manual_seed_all(seed)  #并行gpu
    torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True   #训练集变化不大时使训练加速

def train(args):
    bn_domain_map = {}
    sp_data=None

    from solver.Kmeans_SP_solver import KMEANSSPSolver as Solver
    dataloaders = prepare_data(args)
    num_domains_bn = 2
    sp_data=prepare_data_SP(args)

    # initialize model
    model_state_dict = None
    fx_pretrained = True
    resume_dict = None

    if cfg.RESUME != '':
        resume_dict = torch.load(cfg.RESUME)
        model_state_dict = resume_dict['model_state_dict']
        fx_pretrained = False
    elif cfg.WEIGHTS != '':
        param_dict = torch.load(cfg.WEIGHTS)
        model_state_dict = param_dict['weights']
        bn_domain_map = param_dict['bn_domain_map']
        fx_pretrained = False

    if args.backbone=='lenet':
        net = model.lenet(num_classes=cfg.DATASET.NUM_CLASSES,
                          state_dict=model_state_dict,
                          dropout_ratio=cfg.TRAIN.DROPOUT_RATIO,
                          fc_hidden_dims=cfg.MODEL.FC_HIDDEN_DIMS,
                          num_domains_bn=num_domains_bn)
    else:
        net = model.danet(num_classes=cfg.DATASET.NUM_CLASSES,
                     state_dict=model_state_dict,
                     feature_extractor=cfg.MODEL.FEATURE_EXTRACTOR,
                     frozen=[cfg.TRAIN.STOP_GRAD],
                     fx_pretrained=fx_pretrained,
                     dropout_ratio=cfg.TRAIN.DROPOUT_RATIO,
                     fc_hidden_dims=cfg.MODEL.FC_HIDDEN_DIMS,
                     num_domains_bn=num_domains_bn)

    net = torch.nn.DataParallel(net)
    if torch.cuda.is_available():
       net.cuda()

    # initialize solver
    train_solver = Solver(net, dataloaders, bn_domain_map=bn_domain_map, resume=resume_dict,sp_data=sp_data)

    # train 
    train_solver.solve()
    print('Finished!')

if __name__ == '__main__':
    cudnn.benchmark = True
    cudnn.enabled = True
    setup_seed(1)
    args = parse_args()

    root_path=os.path.abspath(os.path.dirname(__file__))
    print(root_path)
    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(os.path.join(root_path,args.cfg_file))
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)


    if args.exp_name is not None:
        cfg.EXP_NAME = args.exp_name+"_"+args.method
    if args.resume is not None:
        save_dir=os.path.join(root_path, cfg.SAVE_DIR, cfg.EXP_NAME)
        cfg.RESUME = os.path.join(save_dir,args.resume)
    if args.weights is not None:
        save_dir = os.path.join(root_path, cfg.SAVE_DIR, cfg.EXP_NAME)
        cfg.WEIGHTS = os.path.join(save_dir, args.weights)

    print('Using config:')
    pprint.pprint(cfg)

    cfg.SAVE_DIR = os.path.join(root_path,cfg.SAVE_DIR, cfg.EXP_NAME)
    print('Output will be saved to %s.' % cfg.SAVE_DIR)

    train(args)
