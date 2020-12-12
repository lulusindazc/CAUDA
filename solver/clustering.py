import torch
from torch.nn import functional as F
import torch.nn as nn
from utils.utils import to_cuda, to_onehot,euclidean_dist
# from tools.prepare_data import prepare_data_SP
# from scipy.optimize import linear_sum_assignment
from math import ceil
# # from .distributions import GaussianMixture
# from sklearn.mixture import GaussianMixture
# from solver.utils import compute_mean_variance_labelled
# from data.utils import save_samples_probs_pslabel,load_trg_plabels
import numpy as np
# from pycave.bayes import GMM
from discrepancy.hungarian import Hungarian
from model import model
from discrepancy.sploss import SPLLoss,build_optimizer
import os
import model.utils as model_utils
# import umap
# import matplotlib.pyplot as plt

class DIST(object):
    def __init__(self, dist_type='cos'):
        self.dist_type = dist_type 

    def get_dist(self, pointA, pointB, cross=False):
        return getattr(self, self.dist_type)(
		pointA, pointB, cross)

    def cos(self, pointA, pointB, cross):
      
        pointA = F.normalize(pointA, dim=1)
        pointB = F.normalize(pointB, dim=1)
        if not cross:
            return 0.5 * (1.0 - torch.sum(pointA * pointB, dim=1))
        else:
            NA = pointA.size(0)
            NB = pointB.size(0)
            assert(pointA.size(1) == pointB.size(1))
            return 0.5 * (1.0 - torch.matmul(pointA, pointB.transpose(0, 1)))


class Clustering(object):
    def __init__(self,eps, feat_key, max_len=1000, dist_type='cos',opt=None,data=None):
        self.eps = eps
        self.Dist = DIST(dist_type)
        self.samples = {}
        self.path2label = {}
        self.path2prob = {}
        self.center_change = None
        self.stop = False
        self.feat_key = feat_key
        self.max_len = max_len
        self.ckpt_resume,self.ckpt_weights = None,None
        save_path = opt.SAVE_DIR
        os.makedirs(save_path, exist_ok=True)
        ckpt_resume = os.path.join(save_path, 'sp_ckpt_best.resume')
        ckpt_weights = os.path.join(save_path, 'sp_ckpt_best.weights')
        if os.path.exists(ckpt_resume):
            self.ckpt_resume = ckpt_resume
        if os.path.exists(ckpt_weights):
            self.ckpt_weights = ckpt_weights

        self.spnet = model.spnet(num_classes=opt.DATASET.NUM_CLASSES,
                     feature_extractor='resnet50',
                     frozen=[opt.TRAIN.STOP_GRAD],
                     fx_pretrained=False,
                     dropout_ratio=opt.TRAIN.DROPOUT_RATIO,
                     fc_hidden_dims=opt.MODEL.FC_HIDDEN_DIMS,
                     num_domains_bn=1)
        self.spnet = torch.nn.DataParallel(self.spnet)
        if torch.cuda.is_available():
            self.spnet.cuda()
        self.data=data
        self.best_acc=0
        self.data_iter=None#iter(self.data)
        self.iters_per_loop = len(self.data)
        self.opt=opt
        self.sp_loss = SPLLoss(n_samples=len(self.data.dataset.data_paths)).cuda()
        self.optimizer = build_optimizer(self.spnet, self.opt)
        self.CELoss_P = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')  # .cuda()

    def set_init_centers(self, init_centers):
        self.centers = init_centers
        self.init_centers = init_centers
        self.num_classes = self.centers.size(0)

    def clustering_stop(self, centers):
        if centers is None:
            self.stop = False
        else:
            dist = self.Dist.get_dist(centers, self.centers) 
            dist = torch.mean(dist, dim=0)
            print('dist %.4f' % dist.item())
            self.stop = dist.item() < self.eps

    def assign_labels(self, feats):
        dists = self.Dist.get_dist(feats, self.centers, cross=True)
        _, labels = torch.min(dists, dim=1)
        return dists, labels

    def align_centers(self):
        # Optimal assignments

        cost1 = euclidean_dist(self.centers, self.init_centers)
        cost1 = cost1.cpu().numpy()
        hungarian = Hungarian(cost1)
        hungarian.calculate()
        indx_results = hungarian.get_results()
        indx_target = np.zeros(len(indx_results), dtype=np.int64)  # torch.zeros(len(indx_results), dtype=torch.int64)
        for out_tup in indx_results:  # out_tup(source_proto_index,target_pesduo_proto_index)
            indx_target[out_tup[0]] = out_tup[1]
            
        return indx_target

    def get_sample(self):
        data_iter=self.data_iter
        data_loader=self.data
        try:
            sample = next(data_iter)
        except StopIteration:
            self.data_iter = iter(data_loader)
            sample = next(data_iter)
            self.data_iter = data_iter
        return sample

    def refine_label(self,loop):
        # self.data.dataset.initialize_path(path=self.samples['data'],label=self.samples['label'])
        self.data_iter = iter(self.data)

        stop=False
        update_iters,loss_iter=0,0
        # loop=0
        data_gt, data_paths, preds = [], [], []
        while not stop:
        # for epoch in range(refine_epochs):
            self.spnet.train()
            self.spnet.zero_grad()
            sample=self.get_sample()
            # for sample in iter(self.data):
            data = to_cuda(sample['Img'])
            label = to_cuda(sample['Label'])
            index = sample['index'].cpu().data
            # data_paths += sample['Path']
            output = self.spnet(data)
            # ce_loss = CELoss(output['probs'],label)
        
            loss=self.sp_loss(output['logits'],label,index)

            loss.backward()
            loss_iter += loss
            self.optimizer.step()

            # if update_iters>5:
            #     print("The loop:{},The iterations:{},The threshold: {},sp loss:{}".format(loop,update_iters,self.sp_loss.threshold,loss_iter))
            update_iters+=1
            # if update_iters % self.iters_per_loop==0:
            #     # loop+=1
            #     # loss_iter = 0

            if update_iters >= self.iters_per_loop:
                stop = True
            else:
                stop = False
        self.sp_loss.increase_threshold()


    def collect_samples(self, net, loader):
        with torch.no_grad():
            net.eval()
            data_feat, data_gt, data_paths,preds = [], [], [],[]
            # index=[]
            confident=[]
            ind=0
            for sample in iter(loader): 
                data = sample['Img'].cuda()
                data_paths += sample['Path']
                if 'Label' in sample.keys():
                    data_gt += [to_cuda(sample['Label'])]
                    
                
                output = net.forward(data)
                feature = output[self.feat_key].data 
                data_feat += [feature]
                logits = output['logits'].data
                preds += [logits]
                if 'Label' in sample.keys():
                    confident += [self.CELoss_P(logits,to_cuda(sample['Label']))]
                
                
            preds = torch.cat(preds, dim=0)
            preds = torch.max(preds, dim=1).indices
            self.samples['data'] = data_paths
            self.samples['gt'] = torch.cat(data_gt, dim=0) \
                        if len(data_gt)>0 else None
            self.samples['confident']=torch.cat(confident,dim=0) \
                        if len(confident)>0 else None
            self.samples['feature'] = torch.cat(data_feat, dim=0)
            self.samples['preds']=preds
    
    def save_spnet(self,best):
        save_path = self.opt.SAVE_DIR
        os.makedirs(save_path, exist_ok=True)
        self.ckpt_resume = os.path.join(save_path, 'sp_ckpt_best.resume')
        self.ckpt_weights = os.path.join(save_path, 'sp_ckpt_best.weights')
        torch.save({
                    'best_acc': best,
                    'model_state_dict': self.spnet.module.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, self.ckpt_resume)

        torch.save({'weights': self.spnet.module.state_dict(),
                    }, self.ckpt_weights)
    def load_spnet(self):
        model_state_dict=None
        if self.ckpt_resume is not None:
            resume_dict = torch.load(self.ckpt_resume)
            model_state_dict = resume_dict['model_state_dict']
            self.best_acc=resume_dict['best_acc']
            fx_pretrained = False
        elif self.ckpt_weights is not None:
            param_dict = torch.load(self.ckpt_weights)
            model_state_dict = param_dict['weights']
        if model_state_dict is not None:
            model_utils.init_weights(self.spnet.module, model_state_dict, 1, False)
            print('loading the best static for spnet!')
        return model_state_dict
            
    def feature_clustering(self, net, loader):
        centers = None 
        self.stop = False 

        self.collect_samples(net, loader)
        feature = self.samples['feature']

        refs = to_cuda(torch.LongTensor(range(self.num_classes)).unsqueeze(1))
        num_samples = feature.size(0)
        num_split = ceil(1.0 * num_samples / self.max_len)

        assign_pre_labels = []
        while True:
            self.clustering_stop(centers)
            if centers is not None:
                self.centers = centers
            if self.stop: break

            centers = 0
            count = 0

            start = 0
            assign_pre_labels = []
            for N in range(num_split):
                cur_len = min(self.max_len, num_samples - start)
                cur_feature = feature.narrow(0, start, cur_len)
                dist2center, labels = self.assign_labels(cur_feature)
                assign_pre_labels += [labels.cpu().data.numpy()]

                labels_onehot = to_onehot(labels, self.num_classes)
                count += torch.sum(labels_onehot, dim=0)
                labels = labels.unsqueeze(0)
                mask = (labels == refs).unsqueeze(2).type(torch.cuda.FloatTensor)
                reshaped_feature = cur_feature.unsqueeze(0)    
                # update centers
                centers += torch.sum(reshaped_feature * mask, dim=1)
                start += cur_len

    
            mask = (count.unsqueeze(1) > 0).type(torch.cuda.FloatTensor) 
            centers = mask * centers + (1 - mask) * self.init_centers

        cluster2label = self.align_centers()
        assign_labels = [cluster2label[ps_lb] for ps_lb in assign_pre_labels]
        # reorder the centers
        self.centers = self.centers[cluster2label, :]

        dist2center, labels = [], []
        start = 0
        count = 0
        for N in range(num_split):
            cur_len = min(self.max_len, num_samples - start)
            cur_feature = feature.narrow(0, start, cur_len)
            cur_dist2center, cur_labels = self.assign_labels(cur_feature)

            labels_onehot = to_onehot(cur_labels, self.num_classes)
            count += torch.sum(labels_onehot, dim=0)

            dist2center += [cur_dist2center]
            labels += [cur_labels]
            start += cur_len

        pse_labels=torch.cat(labels, dim=0)# initial assigned by the Kmeans
        self.samples['label'] =pse_labels
        preds = self.samples['label']  # .cpu().data
        target = self.samples['gt']
        init_acc = 100.0 * torch.sum(preds == target).item() / preds.size(0)
        print("Kmeans initial acc:{}".format(init_acc))
        ### self-paced training for label refinement
        loop=0

        self.data.dataset.initialize_path(path=self.samples['data'], label=self.samples['label'].cpu().data.numpy().tolist())
        # best_acc=0
        best_acc=self.best_acc
        count=0
        if self.ckpt_resume is not None:
            value=self.opt.SPNET.LOSS_THRESHOLD#0.1
        else:
            value=self.opt.SPNET.LOSS_INIT_THRESHOLD#1.0
        self.sp_loss.reset_threshold(value)
        self.load_spnet()
        cur_images =0

        while True:
            # self.data_iter = iter(self.data)
            self.refine_label(loop)
            self.collect_samples(self.spnet, loader)

            preds=self.samples['preds']#.cpu().data
            target=self.samples['gt']
            # preds = torch.max(preds, dim=1).indices
            acc=100.0 * torch.sum(preds == target).item() / preds.size(0)
            if acc>best_acc:
                best_acc=acc
                self.save_spnet(best_acc)
                count=0
            else:
                count+=1
            loop +=1
            cur_images = 100.0 * self.sp_loss.v.sum() / len(self.samples['data'])
            if cur_images>99.0:
                # count += 1
                if loop>self.opt.SPNET.MAX_EPOCHS or (loop>10 and count>5):#best_acc>init_acc:# or count>5: #or (loop>10 and count>5):
                    break

            print("The loop is {}, number of coverd images: {},with acc:{},initial acc:{},best acc:{}".format(loop,cur_images,acc,init_acc,best_acc))
        self.load_spnet()
        self.collect_samples(self.spnet, loader)
        if best_acc>init_acc:
            print("using the refined label with selp-paced learning")
            self.samples['label'] =self.samples['preds']#.cpu().data
        
        ###
        # self.samples['label'] = torch.cat(labels, dim=0)#to_cuda(torch.tensor(assign_labels)).squeeze(0)#torch.cat(labels, dim=0)
        self.samples['dist2center'] = torch.cat(dist2center, dim=0)

        consis_mask=(self.samples['preds']==self.samples['label'])#(self.samples['dist2center']<0.1)#
        self.samples['mask']=consis_mask

        consis_count=consis_mask.type(torch.cuda.FloatTensor).sum()/num_samples
        # cluster2label = self.align_centers()
        # # reorder the centers
        # self.centers = self.centers[cluster2label, :]
        print("Consistency rate:{}".format(consis_count))
        distmat=self.samples['dist2center']
        dis_max = torch.max(distmat, dim=1)[0].unsqueeze(1).repeat(1, distmat.size(1))
        weights0 = F.normalize(dis_max - distmat, p=1, dim=1)
        self.samples['weights']=weights0

        self.samples['consis_acc']=consis_count
        
        # re-label the data according to the index
        # num_samples = len(self.samples['feature'])
        for k in range(num_samples):
            self.samples['label'][k] = cluster2label[self.samples['label'][k]].item()

        self.center_change = torch.mean(self.Dist.get_dist(self.centers, \
                    self.init_centers))

        for i in range(num_samples):
            self.path2label[self.samples['data'][i]] = self.samples['label'][i]
            self.path2prob[self.samples['data'][i]] = self.samples['weights'][i, :].cpu().data

        del self.samples['feature']

