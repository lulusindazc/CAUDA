import torch
from torch import distributions, nn
import torch.nn.functional as F
import numpy as np
import math
from solver.utils import compute_mean_variance_labelled,DIST
from utils.utils import to_cuda, to_onehot
# import umap

class SSLGaussMixture(torch.distributions.Distribution):

    def __init__(self,n_components, d,feat_key, device=None):
        self.n_components, self.d =n_components,d# means.shape
        self.means = torch.zeros((n_components,d))#means

        self.Dist = DIST('cos')

        self.inv_cov_stds = math.log(math.exp(1.0) - 1.0) * torch.ones((n_components), device=device)

        self.weights = torch.ones((n_components), device=device)
        self.device = device
        self.samples=dict()
        self.path2label = {}
        self.path2prob = {}
        self.feat_key = feat_key

    @property
    def gaussians(self):
        gaussians = [distributions.MultivariateNormal(mean, F.softplus(inv_std)**2 * torch.eye(self.d).to(self.device))
                          for mean, inv_std in zip(self.means, self.inv_cov_stds)]
        return gaussians

    def set_init_centers(self, init_centers,init_var):
        self.means = init_centers
        self.inv_cov_stds=init_var

    def set_init_centers_net(self, net, loader):
        self.collect_samples(net, loader)
        features = self.samples['feature']

        # import numpy as np
        # import scipy.io as io
        #
        # feat_array = np.array(features.cpu().data)
        #
        # # np.savetxt('feature_results.txt', feat_array)
        # io.savemat('Source_C_Feature_600_64.mat', {'feat': feat_array})

        preds = self.samples['gt']
        means_vec, sigma_vec,cov_vec = compute_mean_variance_labelled(features, preds)

        # mean_array= np.array(means_vec.cpu().data)
        # io.savemat('Source_C_mean_12_64.mat', {'feat': mean_array})
        self.set_init_centers(means_vec, sigma_vec)
        # return means_vec,sigma_vec

    def collect_samples(self, net, loader):
        data_feat, data_gt, data_paths,preds = [], [], [],[]
        with torch.no_grad():
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
        preds = torch.cat(preds, dim=0)
        preds = torch.max(preds, dim=1).indices
        self.samples['data'] = data_paths
        self.samples['gt'] = torch.cat(data_gt, dim=0) \
            if len(data_gt) > 0 else None
        feature = torch.cat(data_feat, dim=0)
        # feature = self.feature_reduction(feature)
        self.samples['feature'] = feature#torch.cat(data_feat, dim=0)#F.softmax(,dim=1)
        self.samples['preds']=preds


    def assign_labels(self, feats):
        dists = self.Dist.get_dist(feats, self.means, cross=True)
        _, labels = torch.min(dists, dim=1)
        return dists, labels

    def feature_reduction(self,feature):
        feature_numpy=feature.cpu().numpy()
        feature_numpy_red=umap.UMAP(n_neighbors=5,n_components=100,metric="correlation").fit_transform(feature_numpy)
        feature_red=to_cuda(torch.from_numpy(feature_numpy_red))
        return feature_red

    def feature_clustering(self,net, loader,collect_sample=False):
        if collect_sample:
            self.collect_samples(net, loader)
        feature = self.samples['feature']
        
        # uniform prior
        k=self.means.size(0)
        pi = torch.empty(k).fill_(1. / k)
        feat_llp=get_likelihoods(feature.cpu(),self.means.cpu(),torch.log(self.inv_cov_stds).cpu())#self.class_probs(feature)
        y_p =to_cuda(get_posteriors(feat_llp,pi.log()))
        gt=self.samples['gt']
        preds = torch.max(y_p, dim=1).indices
        source_acc=torch.sum(preds==gt).item()/y_p.size(0)
        print('Source GMM model prediction acc:{}'.format(source_acc))
        self.samples['prob_gmm'] = y_p
        dist2center, labels_cen = self.assign_labels(feature)  #
        self.samples['dist2center'] = dist2center
        self.samples['label']=self.samples['gt']
        num_samples = len(feature)
        for i in range(num_samples):
            self.path2label[self.samples['data'][i]] = self.samples['label'][i].item()
            self.path2prob[self.samples['data'][i]] = self.samples['prob_gmm'][i,:].cpu()

        del self.samples['feature']

    def parameters(self):
       return [self.means, self.inv_cov_std, self.weights]
        
    def sample(self, sample_shape, gaussian_id=None):
        if gaussian_id is not None:
            g = self.gaussians[gaussian_id]
            samples = g.sample(sample_shape)
        else:
            n_samples = sample_shape[0]
            idx = np.random.choice(self.n_components, size=(n_samples, 1), p=F.softmax(self.weights))
            all_samples = [g.sample(sample_shape) for g in self.gaussians]
            samples = all_samples[0]
            for i in range(self.n_components):
                mask = np.where(idx == i)
                samples[mask] = all_samples[i][mask]
        return samples

    def log_prob(self, x, y=None, label_weight=1.):
        all_log_probs = torch.cat([g.log_prob(x)[:, None] for g in self.gaussians], dim=1)
        mixture_log_probs = torch.logsumexp(all_log_probs + torch.log(F.softmax(self.weights)), dim=1)
        if y is not None:
            log_probs = torch.zeros_like(mixture_log_probs)
            mask = (y == -1)
            log_probs[mask] += mixture_log_probs[mask]
            for i in range(self.n_components):
                #Pavel: add class weights here? 
                mask = (y == i)
                log_probs[mask] += all_log_probs[:, i][mask] * label_weight
            return log_probs
        else:
            return mixture_log_probs

    def class_logits(self, x):
        log_probs = torch.cat([g.log_prob(x)[:, None] for g in self.gaussians], dim=1)
        log_probs_weighted = log_probs + torch.log(F.softmax(self.weights))
        return log_probs_weighted

    def classify(self, x):
        log_probs = self.class_logits(x)
        return torch.argmax(log_probs, dim=1)

    def class_probs(self, x):
        log_probs = self.class_logits(x)
        return F.softmax(log_probs, dim=1)

#PAVEL: remove later
class SSLGaussMixtureClassifier(SSLGaussMixture):
    
    def __init__(self, means, cov_std=1., device=None):
        super(SSLGaussMixtureClassifier).__init__(means, cov_std, device)
        self.classifier = nn.Sequential(nn.Linear(self.d, self.n_components))

    def parameters(self):
       return self.classifier.parameters() 

    def forward(self, x):
        return self.classifier.forward(x)

    def log_prob(self, x, y, label_weight=1.):
        all_probs = [torch.exp(g.log_prob(x)) for g in self.gaussians]
        probs = sum(all_probs) / self.n_components
        x_logprobs = torch.log(probs)

        mask = (y != -1)
        labeled_x, labeled_y = x[mask], y[mask].long()
        preds = self.forward(labeled_x)
        y_logprobs = F.cross_entropy(preds, labeled_y)

        return x_logprobs - y_logprobs

