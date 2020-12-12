import torch
import torch.nn as nn
import os
from . import utils as solver_utils
from utils.utils import to_cuda, to_onehot
from torch import optim
from . import clustering
from .base_solver import BaseSolver
from discrepancy.c2c_intra import C2C_INTRA
from model import model

class KMEANSSPSolver(BaseSolver):
    def __init__(self, net, dataloader, bn_domain_map={}, resume=None, **kwargs):
        super(KMEANSSPSolver, self).__init__(net, dataloader, \
                      bn_domain_map=bn_domain_map, resume=resume, **kwargs)

        if len(self.bn_domain_map) == 0:
            self.bn_domain_map = {self.source_name: 0, self.target_name: 1}

        self.clustering_source_name = 'clustering_' + self.source_name
        self.clustering_target_name = 'clustering_' + self.target_name

        assert('categorical' in self.train_data)

        num_layers = len(self.opt.CDD.ALIGNMENT_FEAT_KEYS)#len(self.net.module.FC) + 1

        self.c2c = C2C_INTRA(kernel_num=self.opt.CDD.KERNEL_NUM, kernel_mul=self.opt.CDD.KERNEL_MUL,
                       num_layers=num_layers, num_classes=self.opt.DATASET.NUM_CLASSES,
                       intra_only=self.opt.CDD.INTRA_ONLY)

        self.discrepancy_key = 'intra' if self.opt.CDD.INTRA_ONLY else 'cdd'
        self.clustering = clustering.Clustering(self.opt.CLUSTERING.EPS, 
                                        self.opt.CLUSTERING.FEAT_KEY, 
                                        self.opt.CLUSTERING.BUDGET,opt=self.opt,data=self.sp_data)

        self.spnet = model.spnet(num_classes=self.opt.DATASET.NUM_CLASSES,
                                 feature_extractor=self.opt.MODEL.FEATURE_EXTRACTOR,
                                 frozen=[self.opt.TRAIN.STOP_GRAD],
                                 fx_pretrained=False,
                                 dropout_ratio=self.opt.TRAIN.DROPOUT_RATIO,
                                 fc_hidden_dims=self.opt.MODEL.FC_HIDDEN_DIMS,
                                 num_domains_bn=1)
        self.spnet = torch.nn.DataParallel(self.spnet)
        if torch.cuda.is_available():
            self.spnet.cuda()


        self.clustered_target_samples = {}
        self.consistency=0.0

    def complete_training(self):
        if self.loop >= self.opt.TRAIN.MAX_LOOP:
            return True

        if 'target_centers' not in self.history or \
                'ts_center_dist' not in self.history or \
                'target_labels' not in self.history:
            return False

        if len(self.history['target_centers']) < 2 or \
		len(self.history['ts_center_dist']) < 1 or \
		len(self.history['target_labels']) < 2:
           return False

        # target centers along training
        target_centers = self.history['target_centers']
        eval1 = torch.mean(self.clustering.Dist.get_dist(target_centers[-1],
			target_centers[-2])).item()

        # target-source center distances along training
        eval2 = self.history['ts_center_dist'][-1].item()

        # target labels along training
        path2label_hist = self.history['target_labels']
        paths = self.clustered_target_samples['data']
        num = 0
        for path in paths:
            pre_label = path2label_hist[-2][path]
            cur_label = path2label_hist[-1][path]
            if pre_label != cur_label:
                num += 1
        eval3 = 1.0 * num / len(paths)

        return (eval1 < self.opt.TRAIN.STOP_THRESHOLDS[0] and \
                eval2 < self.opt.TRAIN.STOP_THRESHOLDS[1] and \
                eval3 < self.opt.TRAIN.STOP_THRESHOLDS[2])

    def solve(self):
        stop = False
        if self.resume:
            self.iters += 1
            self.loop += 1
        filtered_classes=[]
        while True: 
            stop,filtered_classes=self.clustering_stage()
            self.writer.add_scalar('consistency accuracy', self.consistency, self.loop)
            if stop:
                break
            # update train data setting
            self.compute_iters_per_loop(filtered_classes)
            # k-step update of network parameters through forward-backward process
            self.update_network(filtered_classes)
            self.loop += 1

        print('Training Done!')
    
    def clustering_stage(self):
        # updating the target label hypothesis through clustering
        target_hypt = {}
        filtered_classes = []
        # with torch.no_grad():
            # self.update_ss_alignment_loss_weight()
        print('Clustering based on %s...' % self.source_name)
        self.update_labels()
        self.clustered_target_samples = self.clustering.samples
        target_centers = self.clustering.centers
        center_change = self.clustering.center_change
        path2label = self.clustering.path2label

        # updating the history
        self.register_history('target_centers', target_centers,
                              self.opt.CLUSTERING.HISTORY_LEN)
        self.register_history('ts_center_dist', center_change,
                              self.opt.CLUSTERING.HISTORY_LEN)
        self.register_history('target_labels', path2label,
                              self.opt.CLUSTERING.HISTORY_LEN)

        if self.clustered_target_samples is not None and \
                self.clustered_target_samples['gt'] is not None:
            preds = to_onehot(self.clustered_target_samples['label'],
                              self.opt.DATASET.NUM_CLASSES)
            gts = self.clustered_target_samples['gt']
            res,_ = self.model_eval(preds, gts)
            print('Clustering %s: %.4f' % (self.opt.EVAL_METRIC, res))

        # check if meet the stop condition
        stop = self.complete_training()
        if stop:
            return stop,filtered_classes

        # filtering the clustering results
        target_hypt, filtered_classes = self.filtering()

        # update dataloaders
        self.construct_categorical_dataloader(target_hypt, filtered_classes)
        # # update train data setting
        # self.compute_iters_per_loop(filtered_classes)
        return stop,filtered_classes
    
    def update_labels(self):
        net = self.net
        net.eval()
        opt = self.opt

        source_dataloader = self.train_data[self.clustering_source_name]['loader']
        net.module.set_bn_domain(self.bn_domain_map[self.source_name])

        source_centers = solver_utils.get_centers(net, 
		source_dataloader, self.opt.DATASET.NUM_CLASSES, 
                self.opt.CLUSTERING.FEAT_KEY)
        init_target_centers = source_centers

        self.weight_dict = self.clustering.path2prob
        target_dataloader = self.train_data[self.clustering_target_name]['loader']
        net.module.set_bn_domain(self.bn_domain_map[self.target_name])

        self.clustering.set_init_centers(init_target_centers)
        self.clustering.feature_clustering(net, target_dataloader)

    def filtering(self):
        threshold = self.opt.CLUSTERING.FILTERING_THRESHOLD
        min_sn_cls = self.opt.TRAIN.MIN_SN_PER_CLASS
        target_samples = self.clustered_target_samples
        self.consistency =target_samples['consis_acc']
        # filtering the samples
        chosen_samples=target_samples

        # filtering the classes
        filtered_classes = solver_utils.filter_class(
		chosen_samples['label'], min_sn_cls, self.opt.DATASET.NUM_CLASSES)

        print('The number of filtered classes: %d.' % len(filtered_classes))
        return chosen_samples, filtered_classes

    def construct_categorical_dataloader(self, samples, filtered_classes):
        # update self.dataloader
        target_classwise = solver_utils.split_samples_classwise(
			samples, self.opt.DATASET.NUM_CLASSES)

        dataloader = self.train_data['categorical']['loader']
        classnames = dataloader.classnames
        dataloader.class_set = [classnames[c] for c in filtered_classes]
        dataloader.target_paths = {classnames[c]: target_classwise[c]['data'] \
                      for c in filtered_classes}
        dataloader.num_selected_classes = min(self.opt.TRAIN.NUM_SELECTED_CLASSES, len(filtered_classes))
        dataloader.construct()

    def CAS(self):
        samples = self.get_samples('categorical')

        source_samples = samples['Img_source']
        source_sample_paths = samples['Path_source']
        source_nums = [len(paths) for paths in source_sample_paths]

        target_samples = samples['Img_target']
        target_sample_paths = samples['Path_target']
        target_nums = [len(paths) for paths in target_sample_paths]

        target_weights = []
        for paths in target_sample_paths:
            cur_target_weights = []
            for path in paths:
                weight_tensor = self.weight_dict[path]
                cur_target_weights += [weight_tensor]
            target_weights.append(torch.stack(cur_target_weights))
        target_weights = torch.cat(target_weights)

        source_sample_labels = samples['Label_source']
        self.selected_classes = [labels[0].item() for labels in source_sample_labels]
        assert(self.selected_classes == 
               [labels[0].item() for labels in  samples['Label_target']])
        return source_samples, source_nums, target_samples, target_nums,to_cuda(target_weights)
            
    def prepare_feats(self, feats):
        return [feats[key] for key in feats if key in self.opt.CDD.ALIGNMENT_FEAT_KEYS]
    
    def prepare_center_probs(self, feats,num_t):
        mean_feat=feature_mean(num_t,feats['feat'])
        mean_probs=self.net.module.FC['logits'](mean_feat)

        return mean_probs
    
    def compute_iters_per_loop(self, filtered_classes):
        if len(filtered_classes)>0:
            self.iters_per_loop = int(len(self.train_data['categorical']['loader'])) * self.opt.TRAIN.UPDATE_EPOCH_PERCENTAGE
        else:
           
            self.iters_per_loop = len(self.train_data[self.source_name]['loader'])
                
        print('Iterations in one loop: %d' % (self.iters_per_loop))

    def update_network(self, filtered_classes):
        # initial configuration
        stop = False
        update_iters = 0

        self.train_data[self.source_name]['iterator'] = \
                     iter(self.train_data[self.source_name]['loader'])
        if len(filtered_classes)>0:
            self.train_data['categorical']['iterator'] = \
                         iter(self.train_data['categorical']['loader'])

        while not stop:
            # update learning rate
            self.update_lr()

            # set the status of network
            self.net.train()
            self.net.zero_grad()

            loss = 0
            ce_loss_iter = 0
            cdd_loss_iter = 0
            fcd_loss_iter=0
            tgt_pse_iter=0
            # coventional sampling for training on labeled source data
            source_sample = self.get_samples(self.source_name) 
            source_data, source_gt = source_sample['Img'],\
                          source_sample['Label']

            source_data = to_cuda(source_data)
            source_gt = to_cuda(source_gt)
            self.net.module.set_bn_domain(self.bn_domain_map[self.source_name])
            source_preds = self.net(source_data)['logits']

            # compute the cross-entropy loss
            ce_loss = self.CELoss(source_preds, source_gt)
            ce_loss.backward()

            ce_loss_iter += ce_loss
            loss += ce_loss
         
            if len(filtered_classes) > 0:
                # update the network parameters
                # 1) class-aware sampling
                source_samples_cls, source_nums_cls, \
                       target_samples_cls, target_nums_cls,target_weights = self.CAS()

                # 2) forward and compute the loss
                source_cls_concat = torch.cat([to_cuda(samples) 
                            for samples in source_samples_cls], dim=0)
                target_cls_concat = torch.cat([to_cuda(samples) 
                            for samples in target_samples_cls], dim=0)

                self.net.module.set_bn_domain(self.bn_domain_map[self.source_name])
                feats_source = self.net(source_cls_concat)
                self.net.module.set_bn_domain(self.bn_domain_map[self.target_name])
                feats_target = self.net(target_cls_concat)

                # prepare the features
                feats_toalign_S = self.prepare_feats(feats_source)
                feats_toalign_T = self.prepare_feats(feats_target)

                cdd_loss = self.c2c.forward(feats_toalign_S, feats_toalign_T,
                               source_nums_cls, target_nums_cls)[self.discrepancy_key]

                cdd_loss *= self.opt.CDD.LOSS_WEIGHT

                loss2 = cdd_loss #+ fcd_loss#+tgt_pse_ce_loss
                loss2.backward()
                # cdd_loss.backward()
                self.writer.add_scalar('c2c_loss', cdd_loss, self.iters)
                cdd_loss_iter += cdd_loss

                loss += loss2#cdd_loss

            # update the network
            self.optimizer.step()
            self.writer.add_scalar('cls_loss_source', ce_loss, self.iters)
            if self.opt.TRAIN.LOGGING and (update_iters+1) % \
                      (max(1, self.iters_per_loop // self.opt.TRAIN.NUM_LOGGING_PER_LOOP)) == 0:
                accu,_ = self.model_eval(source_preds, source_gt)
                cur_loss = {'ce_loss': ce_loss_iter, 'cdd_loss': cdd_loss_iter
                            'total_loss': loss}
                self.logging(cur_loss, accu)

            self.opt.TRAIN.TEST_INTERVAL = min(1.0, self.opt.TRAIN.TEST_INTERVAL)
            self.opt.TRAIN.SAVE_CKPT_INTERVAL = min(1.0, self.opt.TRAIN.SAVE_CKPT_INTERVAL)

            if self.opt.TRAIN.TEST_INTERVAL > 0 and \
		(update_iters+1) % int(self.opt.TRAIN.TEST_INTERVAL * self.iters_per_loop) == 0:
                with torch.no_grad():
                    self.net.module.set_bn_domain(self.bn_domain_map[self.target_name])
                    # accu,_ = self.test()
                    accu, acc_list = self.test()
                    out_str = "Loop:{},Test mean:accuracy:{}. each category accuracy:{}\n".format(self.loop, accu,
                                                                                                  acc_list)
                    print(out_str)
                    self.writer.add_scalar('test_accuarcy', accu,
                                           (self.iters + 1) / int(self.opt.TRAIN.TEST_INTERVAL * self.iters_per_loop))
                    if accu > self.best_test_acc:
                        self.best_test_acc = accu
                        self.save_ckpt(best=True)
                        with open(os.path.join(self.opt.SAVE_DIR, 'result_acc.txt'), 'a') as f:
                            f.write(out_str)
                    print('Test at (loop %d, iters: %d) with %s: %.4f.,best_acc:%.4f' % (self.loop,
                                                                                         self.iters,
                                                                                         self.opt.EVAL_METRIC, accu,
                                                                                         self.best_test_acc))

            if self.opt.TRAIN.SAVE_CKPT_INTERVAL > 0 and \
		(update_iters+1) % int(self.opt.TRAIN.SAVE_CKPT_INTERVAL * self.iters_per_loop) == 0:
                self.save_ckpt()

            update_iters += 1
            self.iters += 1

            # update stop condition
            if update_iters >= self.iters_per_loop:
                stop = True
            else:
                stop = False

