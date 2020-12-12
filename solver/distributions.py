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


import torch
import numpy as np

from math import pi
from scipy.special import logsumexp
#
#
class GaussianMixture(torch.nn.Module):
    """
    https://github.com/ldeecke/gmm-torch/blob/master/gmm.py
    Fits a mixture of k=1,..,K Gaussians to the input data (K is supplied via n_components). Input tensors are expected to be flat with dimensions (n: number of samples, d: number of features).
    The model then extends them to (n, 1, d).
    The model parametrization (mu, sigma) is stored as (1, k, d), and probabilities are shaped (n, k, 1) if they relate to an individual sample, or (1, k, 1) if they assign membership probabilities to one of the mixture components.
    """
    def __init__(self, n_components, n_features, mu_init=None, var_init=None, eps=1.e-6):
        """
        Initializes the model and brings all tensors into their required shape.
        The class expects data to be fed as a flat tensor in (n, d).
        The class owns:
            x:              torch.Tensor (n, 1, d)
            mu:             torch.Tensor (1, k, d)
            var:            torch.Tensor (1, k, d)
            pi:             torch.Tensor (1, k, 1)
            eps:            float
            n_components:   int
            n_features:     int
            log_likelihood: float
        args:
            n_components:   int
            n_features:     int
        options:
            mu_init:        torch.Tensor (1, k, d)
            var_init:       torch.Tensor (1, k, d)
            eps:            float
        """
        super(GaussianMixture, self).__init__()

        self.n_components = n_components
        self.n_features = n_features

        self.mu_init = mu_init
        self.var_init = var_init
        self.eps = eps

        self.log_likelihood = -np.inf

        self._init_params()


    def _init_params(self):
        if self.mu_init is not None:
            # assert self.mu_init.size() == (1, self.n_components, self.n_features),"Input mu_init does not have required tensor dimensions (1, %i, %i)" % (self.n_components, self.n_features)
            # (1, k, d)
            if len(self.mu_init.size()) == 2:
                # (k, d) --> (1, k, d)
                self.mu_init = self.mu_init.unsqueeze(0)
            self.mu = torch.nn.Parameter(self.mu_init,requires_grad=False)
        else:
            self.mu = torch.nn.Parameter(torch.randn(1, self.n_components, self.n_features), requires_grad=False)

        if self.var_init is not None:
            # assert self.var_init.size() == (1, self.n_components, self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i)" % (self.n_components, self.n_features)
            # (1, k, d)
            if len(self.var_init.size()) == 2:
                # (k, d) --> (1, k, d)
                self.var_init = self.var_init.unsqueeze(0)
            self.var = torch.nn.Parameter(self.var_init, requires_grad=False)
        else:
            self.var = torch.nn.Parameter(torch.ones(1, self.n_components, self.n_features), requires_grad=False)
        # (1, k, 1)
        self.pi = torch.nn.Parameter(torch.Tensor(1, self.n_components, 1), requires_grad=False).fill_(1./self.n_components)
        self.params_fitted = False

    def get_parameters(self):
        return [self.mu.data.squeeze(0), self.var.data.squeeze(0), self.pi.data.squeeze()]

    def check_size(self, x):
        if len(x.size()) == 2:
            # (n, d) --> (n, 1, d)
            x = x.unsqueeze(1)

        return x



    def bic(self, x):
        """
        Bayesian information criterion for a batch of samples.
        args:
            x:      torch.Tensor (n, d) or (n, 1, d)
        returns:
            bic:    float
        """
        x = self.check_size(x)
        n = x.shape[0]

        # Free parameters for covariance, means and mixture components
        free_params = self.n_features * self.n_components + self.n_features + self.n_components - 1

        bic = -2. * self.__score(x, sum_data=False).mean() * n + free_params * np.log(n)

        return bic


    def fit(self, x, delta=1e-3, n_iter=100, warm_start=False):
        """
        Fits model to the data.
        args:
            x:          torch.Tensor (n, d) or (n, k, d)
        options:
            delta:      float
            n_iter:     int
            warm_start: bool
        """
        if not warm_start and self.params_fitted:
            self._init_params()

        x = self.check_size(x)

        i = 0
        j = np.inf

        while (i <= n_iter) and (j >= delta):

            log_likelihood_old = self.log_likelihood
            mu_old = self.mu
            var_old = self.var

            self.__em(x)
            self.log_likelihood = self.__score(x)

            if (self.log_likelihood.abs() == float("Inf")) or (self.log_likelihood == float("nan")):
                # When the log-likelihood assumes inane values, reinitialize model
                self.__init__(self.n_components,
                    self.n_features,
                    mu_init=self.mu_init,
                    var_init=self.var_init,
                    eps=self.eps)

            i += 1
            j = self.log_likelihood - log_likelihood_old

            if j <= delta:
                # When score decreases, revert to old parameters
                self.__update_mu(mu_old)
                self.__update_var(var_old)

        self.params_fitted = True


    def predict(self, x, probs=False):
        """
        Assigns input data to one of the mixture components by evaluating the likelihood under each.
        If probs=True returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            probs:      bool
        returns:
            p_k:        torch.Tensor (n, k)
            (or)
            y:          torch.LongTensor (n)
        """
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        log_resp = weighted_log_prob - log_prob_norm
        p_k = torch.exp(log_resp)
        den=p_k.sum(1, keepdim=True)
        if probs:

            p_k =torch.squeeze(p_k / (den))
            return p_k
        else:
            y_p=torch.squeeze(torch.max(p_k, 1)[1].type(torch.LongTensor))
            return y_p

    # K, num_sam = log_likelihoods.size()
    # posteriors = log_likelihoods  # + log_pi[:, None]
    #
    # pos_logsumexp = torch.logsumexp(posteriors, dim=0,
    #                                 keepdim=True)  # max + (posteriors- max).exp().sum(dim=0, keepdim=True).log()
    # posteriors = posteriors - pos_logsumexp  # logsumexp(v, dim=0, keepdim=True)
    # posteriors = posteriors.exp_().t()
    def predict_proba(self, x):
        """
        Returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            y:          torch.LongTensor (n)
        """
        return self.predict(x, probs=True)


    def score_samples(self, x):
        """
        Computes log-likelihood of samples under the current model.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            score:      torch.LongTensor (n)
        """
        x = self.check_size(x)

        score = self.__score(x, sum_data=False)
        return score


    def _estimate_log_prob(self, x):
        """
        Returns a tensor with dimensions (n, k, 1), which indicates the log-likelihood that samples belong to the k-th Gaussian.
        args:
            x:            torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob:     torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        mu = self.mu
        # prec = torch.rsqrt(self.var)
        prec =1.0/self.var
        # mu_x=mu-x
        # mu_2=mu*mu
        # x_2=x*x
        # x_mu=x*mu
        tmp=mu * mu + x * x - 2 * x * mu
        log_p=torch.sum((mu * mu + x * x - 2 * x * mu) * (prec), dim=2, keepdim=True)

        log_det=0.5*torch.sum(torch.log(prec), dim=2, keepdim=True)
        log_likelihoods=-.5 * (self.n_features * to_cuda(torch.log(torch.tensor([2. * pi]))) + log_p) + log_det
        log_likelihoods=log_likelihoods
        return log_likelihoods

    # a = (x - mean) ** 2
    # log_p = -0.5 * (logvar + a / logvar.exp())
    # log_p = log_p + log_norm_constant
    # -0.5 * np.log(2 * np.pi)
    # log_det = torch.sum(logvar, dim=2, keepdim=True)
    # -.5 * (self.n_features * np.log(2. * pi) + log_p) + log_det
    def _e_step(self, x):
        """
        Computes log-responses that indicate the (logarithmic) posterior belief (sometimes called responsibilities) that a data point was generated by one of the k mixture components.
        Also returns the mean of the mean of the logarithms of the probabilities (as is done in sklearn).
        This is the so-called expectation step of the EM-algorithm.
        args:
            x:              torch.Tensor (n,d) or (n, 1, d)
        returns:
            log_prob_norm:  torch.Tensor (1)
            log_resp:       torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        log_resp = weighted_log_prob - log_prob_norm

        return torch.mean(log_prob_norm), log_resp

    # pos_logsumexp = torch.logsumexp(posteriors, dim=0,
    #                                 keepdim=True)  # max + (posteriors- max).exp().sum(dim=0, keepdim=True).log()
    # posteriors = posteriors - pos_logsumexp  # logsumexp(v, dim=0, keepdim=True)
    # posteriors = posteriors.exp_().t()

    def _m_step(self, x, log_resp):
        """
        From the log-probabilities, computes new parameters pi, mu, var (that maximize the log-likelihood). This is the maximization step of the EM-algorithm.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            log_resp:   torch.Tensor (n, k, 1)
        returns:
            pi:         torch.Tensor (1, k, 1)
            mu:         torch.Tensor (1, k, d)
            var:        torch.Tensor (1, k, d)
        """
        x = self.check_size(x)

        resp = torch.exp(log_resp)

        pi = torch.sum(resp, dim=0, keepdim=True) + self.eps
        mu = torch.sum(resp * x, dim=0, keepdim=True) / pi

        x2 = (resp * x * x).sum(0, keepdim=True) / pi
        mu2 = mu * mu
        xmu = (resp * mu * x).sum(0, keepdim=True) / pi
        var = x2 - 2 * xmu + mu2 + self.eps

        pi = pi / x.shape[0]

        return pi, mu, var


    def __em(self, x):
        """
        Performs one iteration of the expectation-maximization algorithm by calling the respective subroutines.
        args:
            x:          torch.Tensor (n, 1, d)
        """
        _, log_resp = self._e_step(x)
        pi, mu, var = self._m_step(x, log_resp)

        self.__update_pi(pi)
        self.__update_mu(mu)
        self.__update_var(var)


    def __score(self, x, sum_data=True):
        """
        Computes the log-likelihood of the data under the model.
        args:
            x:                  torch.Tensor (n, 1, d)
            sum_data:           bool
        returns:
            score:              torch.Tensor (1)
            (or)
            per_sample_score:   torch.Tensor (n)
        """
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=1)

        if sum_data:
            return per_sample_score.sum()
        else:
            return torch.squeeze(per_sample_score)


    def __update_mu(self, mu):
        """
        Updates mean to the provided value.
        args:
            mu:         torch.FloatTensor
        """

        assert mu.size() in [(self.n_components, self.n_features), (1, self.n_components, self.n_features)], "Input mu does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (self.n_components, self.n_features, self.n_components, self.n_features)

        if mu.size() == (self.n_components, self.n_features):
            self.mu = mu.unsqueeze(0)
        elif mu.size() == (1, self.n_components, self.n_features):
            self.mu.data = mu


    def __update_var(self, var):
        """
        Updates variance to the provided value.
        args:
            var:        torch.FloatTensor
        """

        assert var.size() in [(self.n_components, self.n_features), (1, self.n_components, self.n_features)], "Input var does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (self.n_components, self.n_features, self.n_components, self.n_features)

        if var.size() == (self.n_components, self.n_features):
            self.var = var.unsqueeze(0)
        elif var.size() == (1, self.n_components, self.n_features):
            self.var.data = var


    def __update_pi(self, pi):
        """
        Updates pi to the provided value.
        args:
            pi:         torch.FloatTensor
        """

        assert pi.size() in [(1, self.n_components, 1)], "Input pi does not have required tensor dimensions (%i, %i, %i)" % (1, self.n_components, 1)

        self.pi.data = pi


import numpy as np

log_norm_constant = -0.5 * np.log(2 * np.pi)

def log_gaussian(x, mean=0, logvar=0.):
    """
    Returns the component-wise density of x under the gaussian parameterised
    by `mean` and `logvar`
    :param x: (*) torch.Tensor
    :param mean: float or torch.FloatTensor with dimensions (*)
    :param logvar: float or torch.FloatTensor with dimensions (*)
    :param normalize: include normalisation constant?
    :return: (*) log density
    """
    if type(logvar) == 'float':
        logvar = x.new(1).fill_(logvar)

    a = (x - mean) ** 2
    log_p = -0.5 * (logvar + a / logvar.exp())
    log_p = log_p + log_norm_constant

    return log_p

def reparameterize(mu, logvar):
    """
    Draw a sample z ~ N(mu, std), such that it is differentiable
    with respect to `mu` and `logvar`, by sampling e ~ N(0, I) and
    performing a location-scale transform.
    :param mu: torch.Tensor
    :param logvar: torch.Tensor
    :return:
    """
    e = mu.new_empty(*mu.size()).normal_()
    std = (logvar * 0.5).exp()
    return mu + std * e

def get_likelihoods(X, mu, logvar, log=True):
    """
    Compute the likelihood of each data point under each gaussians.
    :param X: design matrix (examples, features)
    :param mu: the component means (K, features)
    :param logvar: the component log-variances (K, features)
    :param log: return value in log domain?
        Note: exponentiating can be unstable in high dimensions.
    :return likelihoods: (K, examples)
    """

    # get feature-wise log-likelihoods (K, examples, features)
    log_likelihoods = log_gaussian(
        X[None, :, :],  # (1, examples, features)
        mu[:, None, :],  # (K, 1, features)
        logvar[:, None, :]  # (K, 1, features)
    )

    # sum over the feature dimension
    log_likelihoods = log_likelihoods.sum(-1)

    if not log:
        log_likelihoods.exp_()
    
    return log_likelihoods

def get_posteriors(log_likelihoods, log_pi):
    """
    Calculate the the posterior probabities log p(z|x), assuming a uniform prior over
    components (for this step only).
    :param likelihoods: the relative likelihood p(x|z), of each data point under each mode (K, examples)
    :param log_pi: log prior (K)
    :return: the log posterior p(z|x) (K, examples)
    """
    K,num_sam=log_likelihoods.size()
    posteriors = log_likelihoods#  + log_pi[:, None]

    pos_logsumexp =torch.logsumexp(posteriors,dim=0, keepdim=True) #max + (posteriors- max).exp().sum(dim=0, keepdim=True).log()
    posteriors = posteriors - pos_logsumexp#logsumexp(v, dim=0, keepdim=True)
    posteriors=posteriors.exp_().t()
    return posteriors
# import matplotlib.cm as cm
# import numpy as np
# import torch
from matplotlib import pyplot as plt
# from torch.nn import Module, Parameter
# from torch.nn.functional import softmax, log_softmax
# dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
# from pykeops.torch import Kernel, kernel_product
# #
# class GaussianMixture_opt(Module):
#     def __init__(self, M, sparsity=0, D=2):
#         super(GaussianMixture_opt, self).__init__()
#         """
#         https://www.kernel-operations.io/keops/_auto_tutorials/gaussian_mixture/plot_gaussian_mixture.html#sphx-glr-auto-tutorials-gaussian-mixture-plot-gaussian-mixture-py
#         """
#         self.params = {'id': Kernel('gaussian(x,y)')}
#         # We initialize our model with random blobs scattered across
#         # the unit square, with a small-ish radius:
#         self.mu = #Parameter(torch.rand(M, D).type(dtype))
#         self.A = 15 * torch.ones(M, 1, 1) * torch.eye(D, D).view(1, D, D)
#         self.covar=
#         self.A = Parameter((self.A).type(dtype).contiguous())
#         self.w = Parameter(torch.ones(M, 1).type(dtype))
#         self.sparsity = sparsity
#
#
#     def update_covariances(self):
#         """Computes the full covariance matrices from the model's parameters."""
#         (M, D, _) = self.A.shape
#         self.params['gamma'] = (torch.matmul(self.A, self.A.transpose(1, 2))).view(M, D * D) / 2
#
#
#     def covariances_determinants(self):
#         """Computes the determinants of the covariance matrices.
#
#         N.B.: PyTorch still doesn't support batched determinants, so we have to
#               implement this formula by hand.
#         """
#         S = self.params['gamma']
#         if S.shape[1] == 2 * 2:
#             dets = S[:, 0] * S[:, 3] - S[:, 1] * S[:, 2]
#         else:
#             raise NotImplementedError
#         return dets.view(-1, 1)
#
#
#     def weights(self):
#         """Scalar factor in front of the exponential, in the density formula."""
#         return softmax(self.w, 0) * self.covariances_determinants().sqrt()
#
#
#     def weights_log(self):
#         """Logarithm of the scalar factor, in front of the exponential."""
#         return log_softmax(self.w, 0) + .5 * self.covariances_determinants().log()
#
#
#     def likelihoods(self, sample):
#         """Samples the density on a given point cloud."""
#         self.update_covariances()
#         return kernel_product(self.params, sample, self.mu, self.weights(), mode='sum')
#
#
#     def log_likelihoods(self, sample):
#         """Log-density, sampled on a given point cloud."""
#         self.update_covariances()
#         return kernel_product(self.params, sample, self.mu, self.weights_log(), mode='lse')
#
#
#     def neglog_likelihood(self, sample):
#         """Returns -log(likelihood(sample)) up to an additive factor."""
#         ll = self.log_likelihoods(sample)
#         log_likelihood = torch.mean(ll)
#         # N.B.: We add a custom sparsity prior, which promotes empty clusters
#         #       through a soft, concave penalization on the class weights.
#         return -log_likelihood + self.sparsity * softmax(self.w, 0).sqrt().mean()
#
#
#     def get_sample(self, N):
#         """Generates a sample of N points."""
#         raise NotImplementedError()
#
#     def predict(self, x, probs=False):
#         """
#         Assigns input data to one of the mixture components by evaluating the likelihood under each.
#         If probs=True returns normalized probabilities of class membership.
#         args:
#             x:          torch.Tensor (n, d) or (n, 1, d)
#             probs:      bool
#         returns:
#             p_k:        torch.Tensor (n, k)
#             (or)
#             y:          torch.LongTensor (n)
#         """
#         x = self.check_size(x)
#
#         weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
#         log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
#         log_resp = weighted_log_prob - log_prob_norm
#         p_k = torch.exp(log_resp)
#         if probs:
#
#             p_k = torch.squeeze(p_k / (p_k.sum(1, keepdim=True)))
#             return p_k
#         else:
#             y_p = torch.squeeze(torch.max(p_k, 1)[1].type(torch.LongTensor))
#             return y_p
#
#     def predict_proba(self, x):
#         """
#         Returns normalized probabilities of class membership.
#         args:
#             x:          torch.Tensor (n, d) or (n, 1, d)
#         returns:
#             y:          torch.LongTensor (n)
#         """
#         return self.predict(x, probs=True)
#
#     def _e_step(self, x):
#         """
#         Computes log-responses that indicate the (logarithmic) posterior belief (sometimes called responsibilities) that a data point was generated by one of the k mixture components.
#         Also returns the mean of the mean of the logarithms of the probabilities (as is done in sklearn).
#         This is the so-called expectation step of the EM-algorithm.
#         args:
#             x:              torch.Tensor (n,d) or (n, 1, d)
#         returns:
#             log_prob_norm:  torch.Tensor (1)
#             log_resp:       torch.Tensor (n, k, 1)
#         """
#         x = self.check_size(x)
#
#         weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
#
#         log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
#         log_resp = weighted_log_prob - log_prob_norm
#
#         return torch.mean(log_prob_norm), log_resp
#
#         # pos_logsumexp = torch.logsumexp(posteriors, dim=0,
#         #                                 keepdim=True)  # max + (posteriors- max).exp().sum(dim=0, keepdim=True).log()
#         # posteriors = posteriors - pos_logsumexp  # logsumexp(v, dim=0, keepdim=True)
#         # posteriors = posteriors.exp_().t()
#
#     def _m_step(self, x, log_resp):
#         """
#         From the log-probabilities, computes new parameters pi, mu, var (that maximize the log-likelihood). This is the maximization step of the EM-algorithm.
#         args:
#             x:          torch.Tensor (n, d) or (n, 1, d)
#             log_resp:   torch.Tensor (n, k, 1)
#         returns:
#             pi:         torch.Tensor (1, k, 1)
#             mu:         torch.Tensor (1, k, d)
#             var:        torch.Tensor (1, k, d)
#         """
#         x = self.check_size(x)
#
#         resp = torch.exp(log_resp)
#
#         pi = torch.sum(resp, dim=0, keepdim=True) + self.eps
#         mu = torch.sum(resp * x, dim=0, keepdim=True) / pi
#
#         x2 = (resp * x * x).sum(0, keepdim=True) / pi
#         mu2 = mu * mu
#         xmu = (resp * mu * x).sum(0, keepdim=True) / pi
#         var = x2 - 2 * xmu + mu2 + self.eps
#
#         pi = pi / x.shape[0]
#
#         return pi, mu, var
#
#     def __em(self, x):
#         """
#         Performs one iteration of the expectation-maximization algorithm by calling the respective subroutines.
#         args:
#             x:          torch.Tensor (n, 1, d)
#         """
#         _, log_resp = self._e_step(x)
#         pi, mu, var = self._m_step(x, log_resp)
#
#         self.__update_pi(pi)
#         self.__update_mu(mu)
#         self.__update_var(var)
#
#     def __score(self, x, sum_data=True):
#         """
#         Computes the log-likelihood of the data under the model.
#         args:
#             x:                  torch.Tensor (n, 1, d)
#             sum_data:           bool
#         returns:
#             score:              torch.Tensor (1)
#             (or)
#             per_sample_score:   torch.Tensor (n)
#         """
#         weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
#         per_sample_score = torch.logsumexp(weighted_log_prob, dim=1)
#
#         if sum_data:
#             return per_sample_score.sum()
#         else:
#             return torch.squeeze(per_sample_score)
#
#     def __update_mu(self, mu):
#         """
#         Updates mean to the provided value.
#         args:
#             mu:         torch.FloatTensor
#         """
#
#         assert mu.size() in [(self.n_components, self.n_features), (1, self.n_components,
#                                                                     self.n_features)], "Input mu does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (
#         self.n_components, self.n_features, self.n_components, self.n_features)
#
#         if mu.size() == (self.n_components, self.n_features):
#             self.mu = mu.unsqueeze(0)
#         elif mu.size() == (1, self.n_components, self.n_features):
#             self.mu.data = mu
#
#     def __update_var(self, var):
#         """
#         Updates variance to the provided value.
#         args:
#             var:        torch.FloatTensor
#         """
#
#         assert var.size() in [(self.n_components, self.n_features), (1, self.n_components,
#                                                                      self.n_features)], "Input var does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (
#         self.n_components, self.n_features, self.n_components, self.n_features)
#
#         if var.size() == (self.n_components, self.n_features):
#             self.var = var.unsqueeze(0)
#         elif var.size() == (1, self.n_components, self.n_features):
#             self.var.data = var
#
#     def __update_pi(self, pi):
#         """
#         Updates pi to the provided value.
#         args:
#             pi:         torch.FloatTensor
#         """
#
#         assert pi.size() in [
#             (1, self.n_components, 1)], "Input pi does not have required tensor dimensions (%i, %i, %i)" % (
#         1, self.n_components, 1)
#
#         self.pi.data = pi

    # def plot(self, sample):
    #     """Displays the model."""
    #     plt.clf()
    #     # Heatmap:
    #     heatmap = self.likelihoods(grid)
    #     heatmap = heatmap.view(res, res).data.cpu().numpy()  # reshape as a "background" image
    #
    #     scale = np.amax(np.abs(heatmap[:]))
    #     plt.imshow(-heatmap, interpolation='bilinear', origin='lower',
    #                vmin=-scale, vmax=scale, cmap=cm.RdBu,
    #                extent=(0, 1, 0, 1))
    #
    #     # Log-contours:
    #     log_heatmap = self.log_likelihoods(grid)
    #     log_heatmap = log_heatmap.view(res, res).data.cpu().numpy()
    #
    #     scale = np.amax(np.abs(log_heatmap[:]))
    #     levels = np.linspace(-scale, scale, 41)
    #
    #     plt.contour(log_heatmap, origin='lower', linewidths=1., colors="#C8A1A1",
    #                 levels=levels, extent=(0, 1, 0, 1))
    #
    #     # Scatter plot of the dataset:
    #     xy = sample.data.cpu().numpy()
    #     plt.scatter(xy[:, 0], xy[:, 1], 100 / len(xy), color='k')
