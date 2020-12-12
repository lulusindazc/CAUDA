import torch
import torch.nn as nn
import numpy as np
import faiss


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)

def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def to_onehot(label, num_classes):
    identity = to_cuda(torch.eye(num_classes))
    onehot = torch.index_select(identity, 0, label)
    return onehot

def mean_accuracy(preds, target):
    num_classes = preds.size(1)
    preds = torch.max(preds, dim=1).indices
    accu_class = []
    for c in range(num_classes):
        mask = (target == c)
        c_count = torch.sum(mask).item()
        if c_count == 0: continue
        preds_c = torch.masked_select(preds, mask)
        accu_class += [1.0 * torch.sum(preds_c == c).item() / c_count]
    return 100.0 * np.mean(accu_class),accu_class

def accuracy(preds, target):
    preds = torch.max(preds, dim=1).indices
    return 100.0 * torch.sum(preds == target).item() / preds.size(0)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def run_kmeans(x,opt,img_index):
    """
    Args:
        x: data to be clustered
    """

    # print('performing kmeans clustering')
    results = {'im2cluster': [], 'centroids': [], 'density': [],'img2cluster_dict':dict()}

    # for seed, num_cluster in enumerate(opt.DATASET.NUM_CLASSES):
        # intialize faiss clustering parameters
    num_cluster=opt.DATASET.NUM_CLASSES
    d = x.shape[1]
    k = int(num_cluster)
    clus = faiss.Clustering(d, k)
    clus.verbose = False#True
    clus.niter = 20
    clus.nredo = 5
    clus.seed = 0
    clus.max_points_per_centroid = 1000
    clus.min_points_per_centroid = 10

    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    # cfg.device = 'gpu'
    index = faiss.GpuIndexFlatL2(res, d, cfg)

    clus.train(x, index)

    D, I = index.search(x, 1)  # for each sample, find cluster distance and assignments
    im2cluster = [int(n[0]) for n in I]
    img2cluster_dict=dict()
    for i,img_ind in enumerate(list(img_index)):
        img2cluster_dict[img_ind]=im2cluster[i]
    # get cluster centroids
    centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

    # sample-to-centroid distances for each cluster 
    Dcluster = [[] for c in range(k)]
    for im, i in enumerate(im2cluster):
        Dcluster[i].append(D[im][0])

    # concentration estimation (phi)        
    density = np.zeros(k)
    for i, dist in enumerate(Dcluster):
        if len(dist) > 1:
            d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
            density[i] = d

            # if cluster only has one point, use the max to estimate its concentration        
    dmax = density.max()
    for i, dist in enumerate(Dcluster):
        if len(dist) <= 1:
            density[i] = dmax

    density = density.clip(np.percentile(density, 10),
                           np.percentile(density, 90))  # clamp extreme values for stability
    density = opt.CLUSTERING.TEMPERATURE * density / density.mean()  # scale the mean to temperature 

    # convert to cuda Tensors for broadcast
    centroids = torch.Tensor(centroids).cuda()
    # centroids = nn.functional.normalize(centroids, p=2, dim=1)

    im2cluster = torch.LongTensor(im2cluster).cuda()
    density = torch.Tensor(density).cuda()

    results['centroids'].append(centroids)
    results['density'].append(density)
    results['im2cluster'].append(im2cluster)
    results['img2cluster_dict']=img2cluster_dict
    return results



def kmeans(x,opt):
    """
    Args:
        x: data to be clustered
    """

    # print('performing kmeans clustering')
    results = {'im2cluster': [], 'centroids': [], 'density': []}

    # for seed, num_cluster in enumerate(opt.DATASET.NUM_CLASSES):
        # intialize faiss clustering parameters
    num_cluster=opt.DATASET.NUM_CLASSES
    d = x.shape[1]
    k = int(num_cluster)
    clus = faiss.Clustering(d, k)
    clus.verbose = False#True
    clus.niter = 20
    clus.nredo = 5
    clus.seed = 0
    clus.max_points_per_centroid = 1000
    clus.min_points_per_centroid = 10

    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    # cfg.device = 'gpu'
    index = faiss.GpuIndexFlatL2(res, d, cfg)

    clus.train(x, index)

    D, I = index.search(x, 1)  # for each sample, find cluster distance and assignments
    im2cluster = [int(n[0]) for n in I]
    # get cluster centroids
    centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)
    # sample-to-centroid distances for each cluster
    Dcluster = [[] for c in range(k)]
    for im, i in enumerate(im2cluster):
        Dcluster[i].append(D[im][0])

    # concentration estimation (phi)
    density = np.zeros(k)
    for i, dist in enumerate(Dcluster):
        if len(dist) > 1:
            d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
            density[i] = d

            # if cluster only has one point, use the max to estimate its concentration
    dmax = density.max()
    for i, dist in enumerate(Dcluster):
        if len(dist) <= 1:
            density[i] = dmax

    density = density.clip(np.percentile(density, 10),
                           np.percentile(density, 90))  # clamp extreme values for stability
    density = opt.CLUSTERING.TEMPERATURE * density / density.mean()  # scale the mean to temperature

    # convert to cuda Tensors for broadcast
    centroids = torch.Tensor(centroids).cuda()
    # centroids = nn.functional.normalize(centroids, p=2, dim=1)

    im2cluster = torch.LongTensor(im2cluster).cuda()
    density = torch.Tensor(density).cuda()

    results['centroids'].append(centroids)
    results['density'].append(density)
    results['im2cluster'].append(im2cluster)

    return results

