import torch
from utils.utils import to_cuda,euclidean_dist
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as scp


def filter_samples(samples, threshold=0.05):
    batch_size_full = len(samples['data'])
    # min_dist = torch.min(samples['dist2center'], dim=1)[0]
    min_dist = samples['confident']
    mask = min_dist < threshold
    # max_prob=torch.squeeze(torch.max(samples['prob_gmm'], 1)[0].type(torch.LongTensor))
    # mask =max_prob >threshold
    filtered_data = [samples['data'][m] 
		for m in range(mask.size(0)) if mask[m].item() == 1]
    filtered_label = torch.masked_select(samples['label'], mask)
    filtered_gt = torch.masked_select(samples['gt'], mask) \
                     if samples['gt'] is not None else None

    filtered_samples = {}
    filtered_samples['data'] = filtered_data
    filtered_samples['label'] = filtered_label
    filtered_samples['gt'] = filtered_gt

    left_samples={}
    left_samples['data']=[samples['data'][m]
		for m in range(mask.size(0)) if mask[m].item() == 0]
    left_samples['label']=torch.masked_select(samples['label'], ~mask)
    left_samples['gt']=torch.masked_select(samples['gt'], ~mask) \
                     if samples['gt'] is not None else None
    assert len(filtered_samples['data']) == filtered_samples['label'].size(0)
    print('select %f' % (1.0 * len(filtered_data) / batch_size_full))

    return filtered_samples,left_samples

def filter_samples_mask(samples):
    batch_size_full = len(samples['data'])
    min_dist = torch.min(samples['dist2center'], dim=1)[0]
    mask = samples['mask']
    # max_prob=torch.squeeze(torch.max(samples['prob_gmm'], 1)[0].type(torch.LongTensor))
    # mask =max_prob >threshold
    filtered_data = [samples['data'][m]
		for m in range(mask.size(0)) if mask[m].item() == 1]
    filtered_label = torch.masked_select(samples['label'], mask)
    filtered_gt = torch.masked_select(samples['gt'], mask) \
                     if samples['gt'] is not None else None

    filtered_samples = {}
    filtered_samples['data'] = filtered_data
    filtered_samples['label'] = filtered_label
    filtered_samples['gt'] = filtered_gt

    left_samples={}
    left_samples['data']=[samples['data'][m]
		for m in range(mask.size(0)) if mask[m].item() == 0]
    left_samples['label']=torch.masked_select(samples['label'], ~mask)
    left_samples['gt']=torch.masked_select(samples['gt'], ~mask) \
                     if samples['gt'] is not None else None
    assert len(filtered_samples['data']) == filtered_samples['label'].size(0)
    print('select %f' % (1.0 * len(filtered_data) / batch_size_full))

    return filtered_samples,left_samples

def filter_class(labels, num_min, num_classes):
    filted_classes = []
    for c in range(num_classes):   
        mask = (labels == c)
        count = torch.sum(mask).item()
        if count >= num_min:
            filted_classes.append(c)

    return filted_classes

def split_samples_classwise(samples, num_classes):
    data = samples['data'] 
    label = samples['label']
    gt = samples['gt']
    samples_list = []
    for c in range(num_classes):
        mask = (label == c)
        data_c = [data[k] for k in range(mask.size(0)) if mask[k].item() == 1]
        label_c = torch.masked_select(label, mask)
        gt_c = torch.masked_select(gt, mask) if gt is not None else None
        samples_c = {}
        samples_c['data'] = data_c
        samples_c['label'] = label_c
        samples_c['gt'] = gt_c
        samples_list.append(samples_c)

    return samples_list

def adjust_learning_rate_exp(lr, optimizer, iters, decay_rate=0.1, decay_step=25):
    lr = lr * (decay_rate ** (iters // decay_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']

def adjust_learning_rate_RevGrad(lr, optimizer, max_iter, cur_iter, alpha=10, beta=0.75):
    p = 1.0 * cur_iter / (max_iter - 1)
    lr = lr / pow(1.0 + alpha * p, beta)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']

def adjust_learning_rate_inv(lr, optimizer, iters, alpha=0.001, beta=0.75):
    lr = lr / pow(1.0 + alpha * iters, beta)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']

def set_param_groups(net, lr_mult_dict):
    params = []
    modules = net.module._modules
    for name in modules:
        module = modules[name]
        if name in lr_mult_dict:
            params += [{'params': module.parameters(), 'lr_mult': lr_mult_dict[name]}]
        else:
            params += [{'params': module.parameters(), 'lr_mult': 1.0}]

    return params

def set_param_seperate(net, lr_mult_dict):
    params = []
    modules = net.module._modules
    for name in lr_mult_dict.keys():
        if name in modules:
            module=modules[name]
            params+= [{'params': module.parameters(), 'lr_mult': lr_mult_dict[name]}]
    return params

def get_centers(net, dataloader, num_classes, key='feat'):        
    centers = 0 
    refs = to_cuda(torch.LongTensor(range(num_classes)).unsqueeze(1))

    for sample in iter(dataloader):
        data = to_cuda(sample['Img'])
        gt = to_cuda(sample['Label'])
        batch_size = data.size(0)

        output = net.forward(data)[key]
        feature = output.data 
        feat_len = feature.size(1)
    
        gt = gt.unsqueeze(0).expand(num_classes, -1)
        mask = (gt == refs).unsqueeze(2).type(torch.cuda.FloatTensor)
        feature = feature.unsqueeze(0)
        # update centers
        centers += torch.sum(feature * mask, dim=1)

    return centers


def get_gmm_mu_sigma(net, dataloader, num_classes, key='feat',name="source"):
    centers = 0
    refs = to_cuda(torch.LongTensor(range(num_classes)).unsqueeze(1))
    features,gts = [],[]
    preds,data_paths=[],[]
    samples=dict()
    for sample in iter(dataloader):
        data = to_cuda(sample['Img'])
        gt = to_cuda(sample['Label'])
        batch_size = data.size(0)

        output = net.forward(data)
        feature = output[key].data
        feat_len = feature.size(1)

        gt = gt.unsqueeze(0).expand(num_classes, -1)
        mask = (gt == refs).unsqueeze(2).type(torch.cuda.FloatTensor)
        feature = feature.unsqueeze(0)
        # update centers
        centers += torch.sum(feature * mask, dim=1)
        # update variances
        features += [feature]
        gts+=[gt]

        # for sampling
        data_paths += sample['Path']

        # if name=='target':
        logits = output['logits']
        preds += [logits]

    # features = torch.cat(features, dim=0)
    # gts = torch.cat(gts, dim=0)
    #
    preds = torch.cat(preds, dim=0)
    preds=torch.max(preds, dim=1).indices

    samples['data'] = data_paths
    samples['gt'] = torch.cat(gts, dim=0) \
        if len(gts) > 0 else None
    samples['feature'] = torch.cat(features, dim=0)
    samples['preds'] = preds


    if name=='target':
        means_vec, sigma_vec = compute_mean_variance_labelled(features, preds)
    else:
        means_vec, sigma_vec = compute_mean_variance_labelled(features, gts)
    samples['means'] =means_vec
    samples['var']=sigma_vec
    return samples

def compute_mean_variance_labelled(embeddings,gts):
    """
    embeddings: features
    gts: label
    compute parameters of gaussian model (mu,sigma)
    """

    gts_cpu = gts.to('cpu')
    # embeddings=F.softmax(embeddings,dim=1)
    embeddings_cpu = embeddings.to("cpu")
    def supp_idxs(c):
        # FIXME when torch will support where as np
        return gts_cpu.eq(c).nonzero(as_tuple=False).squeeze(1)

    classes = torch.unique(gts_cpu)
    # n_classes = len(classes)


    idxs_group = list(map(supp_idxs, classes))
    ## cetroids (num_class,feat_dim)
    prototypes = torch.stack([embeddings_cpu[idx_list].mean(0) for idx_list in idxs_group])
    # prototypes=F.normalize(prototypes,p=2,dim=1)
    variances=[]
    cov_matrix=[]
    for idx_list in idxs_group:
        feats=embeddings_cpu[idx_list]
        means=feats.mean(0).unsqueeze(0)#.repeat(feats.size(0),1)
        n=feats.size(0)
        m,d=means.size()

        feats_ex=feats.unsqueeze(1).expand(n, m, d)
        means_ex=means.unsqueeze(0).expand(n, m, d)

        sigma_2=torch.pow(feats_ex-means_ex, 2).squeeze(1).mean(0).unsqueeze(0)# (n,m,d)
        M=feats_ex-means_ex
        M_t=(feats_ex-means_ex).permute(0,2,1)
        # cov=torch.inverse(M_t.matmul(M).mean(0)).unsqueeze(0)
        variances+=[torch.pow(sigma_2,0.5)]
        # cov_matrix +=[cov]
    variances=to_cuda(torch.cat(variances,dim=0))
    # cov_matrix=to_cuda(torch.cat(cov_matrix))
    return to_cuda(prototypes),variances,variances

def entropy_weight(Z):
    num_class=Z.size(1)
    Z=F.softmax(Z, dim=1)
    probs_l1 = F.normalize(Z, dim=1).numpy()
    probs_l1[probs_l1 < 0] = 0
    entropy = scp.entropy(probs_l1.T)
    weights = 1 - entropy / np.log(num_class)
    weights = weights / np.max(weights)
    p_labels = np.argmax(probs_l1, 1)
    class_weights = np.ones((num_class,), dtype=np.float32)
    # Compute the weight for each class
    for i in range(num_class):
        cur_idx = np.where(np.asarray(p_labels) == i)[0]
        class_weights[i] = (float(Z.shape[0]) / num_class)
        if cur_idx.size>0:
            class_weights[i] /= cur_idx.size

    c_weight = [class_weights[p_labels[i]] for i in range(len(p_labels))]
    p_labels=torch.tensor(p_labels).cuda()
    weights = torch.tensor(weights).cuda()
    c_weight=torch.tensor(c_weight).cuda()

    return p_labels,weights,c_weight


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