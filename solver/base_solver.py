import torch
import torch.nn as nn
import os
from . import utils as solver_utils 
from utils.utils import to_cuda, mean_accuracy, accuracy
from torch import optim
from math import ceil as ceil
from config.config import cfg
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from discrepancy.sploss import SPLLoss

class BaseSolver:
    def __init__(self, net, dataloader, bn_domain_map={}, resume=None,sp_data=None, **kwargs):
        self.opt = cfg
        self.source_name = self.opt.DATASET.SOURCE_NAME.replace('_train',"") if '_train' in self.opt.DATASET.SOURCE_NAME else self.opt.DATASET.SOURCE_NAME
        self.target_name = self.opt.DATASET.TARGET_NAME.replace('_train',"") if '_train' in self.opt.DATASET.TARGET_NAME else self.opt.DATASET.TARGET_NAME

        self.net = net
        # self.sp_net=net
        self.init_data(dataloader)

        self.log_dir = os.path.join(self.opt.SAVE_DIR, 'tensorboard_sam_c2c')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        
        self.CELoss = nn.CrossEntropyLoss()
        self.CELoss_P=nn.CrossEntropyLoss( ignore_index=-1, reduction='none')#.cuda()
        if torch.cuda.is_available():
            self.CELoss.cuda()
            self.CELoss_P.cuda()
        
        self.sp_data=sp_data

        self.loop = 0
        self.iters = 0
        self.iters_per_loop = None
        self.history = {}
        self.best_test_acc =0
     
        self.base_lr = self.opt.TRAIN.BASE_LR
        self.momentum = self.opt.TRAIN.MOMENTUM

        self.bn_domain_map = bn_domain_map

        self.optim_state_dict = None
        self.resume = False
        if resume is not None:
            self.resume = True
            # self.loop = resume['loop']
            # self.iters = resume['iters']
            self.best_test_acc =resume['best_acc']
            self.history = resume['history']
            # self.optim_state_dict = resume['optimizer_state_dict']
            self.bn_domain_map = resume['bn_domain_map']
            print('Resume Training from loop %d, iters %d.' % \
			(self.loop, self.iters))

        self.build_optimizer()

    def init_data(self, dataloader):
        self.train_data = {key: dict() for key in dataloader if key != 'test'}
        for key in self.train_data.keys():
            if key not in dataloader:
                continue
            cur_dataloader = dataloader[key]
            self.train_data[key]['loader'] = cur_dataloader 
            self.train_data[key]['iterator'] = None

        if 'test' in dataloader:
            self.test_data = dict()
            self.test_data['loader'] = dataloader['test']
        
    def build_optimizer(self):
        opt = self.opt
        param_groups = solver_utils.set_param_groups(self.net, 
		dict({'FC': opt.TRAIN.LR_MULT,'FC1': opt.TRAIN.LR_MULT}))#opt.TRAIN.LR_MULT

        param_G_groups = solver_utils.set_param_seperate(self.net, dict({'feature_extractor': 1.0}))
        param_FC_groups = solver_utils.set_param_seperate(self.net,dict({'FC': opt.TRAIN.LR_MULT,'FC1': opt.TRAIN.LR_MULT}))#'FC': opt.TRAIN.LR_MULT,'FC1': opt.TRAIN.LR_MULT
        assert opt.TRAIN.OPTIMIZER in ["Adam", "SGD"], \
            "Currently do not support your specified optimizer."

        if opt.TRAIN.OPTIMIZER == "Adam":
            self.optimizer = optim.Adam(param_groups, 
			lr=self.base_lr, betas=[opt.ADAM.BETA1, opt.ADAM.BETA2], 
			weight_decay=opt.TRAIN.WEIGHT_DECAY)

            self.G_optimizer=optim.Adam(param_G_groups,lr=self.base_lr, betas=[opt.ADAM.BETA1, opt.ADAM.BETA2],
			weight_decay=opt.TRAIN.WEIGHT_DECAY)

            self.C_optimizer = optim.Adam(param_FC_groups, lr=self.base_lr, betas=[opt.ADAM.BETA1, opt.ADAM.BETA2],
                                          weight_decay=opt.TRAIN.WEIGHT_DECAY)
            
        elif opt.TRAIN.OPTIMIZER == "SGD":
            self.optimizer = optim.SGD(param_groups, 
			lr=self.base_lr, momentum=self.momentum, 
			weight_decay=opt.TRAIN.WEIGHT_DECAY)

           
            
            self.G_optimizer = optim.SGD(param_G_groups,
			lr=self.base_lr, momentum=self.momentum,
			weight_decay=opt.TRAIN.WEIGHT_DECAY)

            self.C_optimizer =optim.SGD(param_FC_groups,
			lr=self.base_lr, momentum=self.momentum,
			weight_decay=opt.TRAIN.WEIGHT_DECAY)

        if self.optim_state_dict is not None:
            self.optimizer.load_state_dict(self.optim_state_dict)

    def update_lr(self):
        iters = self.iters
        if self.opt.TRAIN.LR_SCHEDULE == 'exp':
            solver_utils.adjust_learning_rate_exp(self.base_lr, 
			self.optimizer, iters, 
                        decay_rate=self.opt.EXP.LR_DECAY_RATE,
			decay_step=self.opt.EXP.LR_DECAY_STEP)

        elif self.opt.TRAIN.LR_SCHEDULE == 'inv':
            solver_utils.adjust_learning_rate_inv(self.base_lr, self.optimizer, 
		    iters, self.opt.INV.ALPHA, self.opt.INV.BETA)

        else:
            raise NotImplementedError("Currently don't support the specified \
                    learning rate schedule: %s." % self.opt.TRAIN.LR_SCHEDULE)

    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1,dim=1) - F.softmax(out2,dim=1)))

    def logging(self, loss, accu):
        print('[loop: %d, iters: %d]: ' % (self.loop, self.iters))
        loss_names = ""
        loss_values = ""
        lr_value=''
        for key in loss:
            loss_names += key + ","
            loss_values += '%.4f,' % (loss[key])
        loss_names = loss_names[:-1] + ': '
        loss_values = loss_values[:-1] + ';'
        lr_value="lr:"+str(self.optimizer.param_groups[0]['lr'])
        loss_str = self.source_name+"->"+self.target_name+":"+lr_value+" "+loss_names + loss_values + (' source %s: %.4f.' %
                    (self.opt.EVAL_METRIC, accu))
        print(loss_str)

    def model_eval(self, preds, gts):
        assert(self.opt.EVAL_METRIC in ['mean_accu', 'accuracy']), \
             "Currently don't support the evaluation metric you specified."

        if self.opt.EVAL_METRIC == "mean_accu": 
            res,res_list = mean_accuracy(preds, gts)
            return res, res_list
        elif self.opt.EVAL_METRIC == "accuracy":
            res= accuracy(preds, gts)
            return res,res

    def save_ckpt(self,best=False):
        save_path = self.opt.SAVE_DIR
        os.makedirs(save_path,exist_ok=True)
        # ckpt_resume = os.path.join(save_path, 'ckpt_%d_%d.resume' % (self.loop, self.iters))
        # ckpt_weights = os.path.join(save_path, 'ckpt_%d_%d.weights' % (self.loop, self.iters))
        # torch.save({'loop': self.loop,
        #             'iters': self.iters,
        #             'best_acc':self.best_test_acc,
        #             'model_state_dict': self.net.module.state_dict(),
        #             'optimizer_state_dict': self.optimizer.state_dict(),
        #             'history': self.history,
        #             'bn_domain_map': self.bn_domain_map
        #             }, ckpt_resume)
        # 
        # torch.save({'weights': self.net.module.state_dict(),
        #             'bn_domain_map': self.bn_domain_map
        #             }, ckpt_weights)
        if best:
            ckpt_resume = os.path.join(save_path, 'ckpt_best.resume')
            ckpt_weights = os.path.join(save_path, 'ckpt_best.weights')
            torch.save({'loop': self.loop,
                        'iters': self.iters,
                        'best_acc': self.best_test_acc,
                        'model_state_dict': self.net.module.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'history': self.history,
                        'bn_domain_map': self.bn_domain_map
                        }, ckpt_resume)

            torch.save({'weights': self.net.module.state_dict(),
                        'bn_domain_map': self.bn_domain_map
                        }, ckpt_weights)
            
            
        
        
    def complete_training(self):
        if self.loop > self.opt.TRAIN.MAX_LOOP:
            return True

    def register_history(self, key, value, history_len):
        if key not in self.history:
            self.history[key] = [value]
        else:
            self.history[key] += [value]
        
        if len(self.history[key]) > history_len:
            self.history[key] = \
                 self.history[key][len(self.history[key]) - history_len:]
       
    def solve(self):
        print('Training Done!')

    def get_samples(self, data_name):
        assert(data_name in self.train_data)
        assert('loader' in self.train_data[data_name] and \
               'iterator' in self.train_data[data_name])

        data_loader = self.train_data[data_name]['loader']
        data_iterator = self.train_data[data_name]['iterator']
        assert data_loader is not None and data_iterator is not None, \
            'Check your dataloader of %s.' % data_name 

        try:
            sample = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            sample = next(data_iterator)
            self.train_data[data_name]['iterator'] = data_iterator
        return sample

    def get_samples_categorical(self, data_name, category):
        assert(data_name in self.train_data)
        assert('loader' in self.train_data[data_name] and \
               'iterator' in self.train_data[data_name])

        data_loader = self.train_data[data_name]['loader'][category]
        data_iterator = self.train_data[data_name]['iterator'][category]
        assert data_loader is not None and data_iterator is not None, \
            'Check your dataloader of %s.' % data_name

        try:
            sample = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            sample = next(data_iterator)
            self.train_data[data_name]['iterator'][category] = data_iterator

        return sample

    def test(self):
        self.net.eval()
        preds = []
        gts = []
        for sample in iter(self.test_data['loader']):
            data, gt = to_cuda(sample['Img']), to_cuda(sample['Label'])
            logits = self.net(data)['logits']
            preds += [logits]
            gts += [gt]

        preds = torch.cat(preds, dim=0)
        gts = torch.cat(gts, dim=0)

        res,res_list = self.model_eval(preds, gts)
        return res,res_list

    def clear_history(self, key):
        if key in self.history:
            self.history[key].clear()
            
    def pseudo_ce_loss(self,preds_tensor):
        tgt_pred = preds_tensor.detach()
        p_label_tgt, weight_tgt_p, c_weight_tgt_p = solver_utils.entropy_weight(tgt_pred.cpu())
        ce_loss = self.CELoss_P(preds_tensor, p_label_tgt) * weight_tgt_p.float()
        ce_loss = ce_loss * c_weight_tgt_p
        ce_loss = ce_loss.sum() / preds_tensor.size(0)
        return ce_loss
    
    def proto_center_pseudo(self,preds_tensor,weights,log_space=True):
        '''
        preds_tensor: batch_size * num_classes
        weights: batch_size * num_classes

        '''
        ## method 1: Loss=sum(i=1,batch_size)sum(k=1,num_classes)(-w_ik*log(preds_ik))
        # probs=F.log_softmax(preds_tensor,dim=1)
        # ce_loss=torch.mul(probs,-weights).sum(1) # -wlogp, (batch_size,1)
        # ce_loss = ce_loss.sum() / preds_tensor.size(0)
        ## method 2 instance wise w_i=weights[i,pse_label], Loss=w_i*CE(preds,pse_label)
        #weights: batch_size * 1

        ####  hard assignement
        # ce_loss = -torch.log(weights)+(self.CELoss_P(preds_tensor, pseudo_label) *weights)#/weights.sum()#[pseudo_label]
        # ce_loss=ce_loss.sum()/ preds_tensor.size(0)
        #
        ## soft assignment # weights is of dimension (num_data,num_class)
        
        if log_space:
            log_soft=nn.LogSoftmax(dim=1)
            preds = -log_soft(preds_tensor)
        else:
            preds=-torch.log(preds_tensor)
            
        # log_soft=nn.LogSoftmax(dim=1)
        # Nll=nn.NLLLoss()
        # weights = to_cuda(weights[range(weights.size(0)), pseudo_label])
        # preds=-log_soft(preds_tensor)
        ce_loss=(preds*weights).mean(1)
        ce_loss=ce_loss.sum()/ preds_tensor.size(0)
        return ce_loss

    def solve(self):
        pass

    def update_network(self, **kwargs):
        pass

