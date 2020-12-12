import torch
from torch import Tensor
import torch.nn as nn
from torch import optim
from solver.utils import set_param_groups
import torch.nn.functional as F

def build_optimizer(net,opt):
    # opt = self.opt
    param_groups = set_param_groups(net,dict({'FC': opt.TRAIN.LR_MULT,
                                                       'FC1': opt.TRAIN.LR_MULT}))  # opt.TRAIN.LR_MULT
    assert opt.TRAIN.OPTIMIZER in ["Adam", "SGD"],\
         \
        "Currently do not support your specified optimizer."

    if opt.TRAIN.OPTIMIZER == "Adam":
        optimizer = optim.Adam(param_groups,
                                    lr=opt.TRAIN.BASE_LR, betas=[opt.ADAM.BETA1, opt.ADAM.BETA2],#opt.TRAIN.BASE_LR  5*1e-5
                                    weight_decay=opt.TRAIN.WEIGHT_DECAY)

    elif opt.TRAIN.OPTIMIZER == "SGD":
        optimizer = optim.SGD(param_groups,
                                   lr=opt.TRAIN.BASE_LR, momentum=opt.TRAIN.MOMENTUM,
                                   weight_decay=opt.TRAIN.WEIGHT_DECAY)

    return optimizer

class SPLLoss(nn.Module):
    def __init__(self, n_samples):
        super(SPLLoss, self).__init__()
        self.threshold = 0.8
        self.growing_factor = 1.3
        self.v = torch.zeros(n_samples).int()
        self.loss=nn.CrossEntropyLoss(ignore_index=-1, reduction='none')#.cuda()#nn.NLLLoss(reduction="none")

    def forward(self, input, target,index):
        super_loss =self.loss(input, target)
        v = self.spl_loss(super_loss)
        self.v[index] = v.cpu().data
        loss=super_loss * v
        loss=loss.mean()
        # loss=loss-(self.threshold*self.v).sum()
        return loss

    def increase_threshold(self):
        self.threshold *= self.growing_factor
    def reset_threshold(self,value=0.1):
        self.threshold=value
    def spl_loss(self, super_loss):
        v = super_loss < self.threshold
        return v.int()