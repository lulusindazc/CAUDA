from . import resnet 
from .lenet import LeNetEncoder,SMFeature
from .domain_specific_module import BatchNormDomain
from utils import utils
from . import utils as model_utils
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.utils import to_cuda
from solver.utils import set_param_groups

backbones = [resnet]

class FC_BN_ReLU_Domain(nn.Module):
    def __init__(self, in_dim, out_dim, num_domains_bn):
        super(FC_BN_ReLU_Domain, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = BatchNormDomain(out_dim, num_domains_bn, nn.BatchNorm1d)
        self.relu = nn.ReLU(inplace=True)
        self.bn_domain = 0

    def set_bn_domain(self, domain=0):
        assert(domain < self.num_domains_bn), "The domain id exceeds the range."
        self.bn_domain = domain
        self.bn.set_domain(self.bn_domain)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DANet(nn.Module):
    def __init__(self, num_classes, feature_extractor='resnet101', 
                 fx_pretrained=True, fc_hidden_dims=[], frozen=[], 
                 num_domains_bn=2, dropout_ratio=(0.5,)):
        super(DANet, self).__init__()
        self.feature_extractor = utils.find_class_by_name(
               feature_extractor, backbones)(pretrained=fx_pretrained, 
               frozen=frozen, num_domains=num_domains_bn)

        self.bn_domain = 0
        self.num_domains_bn = num_domains_bn

        feat_dim = self.feature_extractor.out_dim
        self.in_dim = feat_dim

        self.FC = nn.ModuleDict()

        self.dropout = nn.ModuleDict()
        self.num_hidden_layer = len(fc_hidden_dims)

        in_dim = feat_dim
        for k in range(self.num_hidden_layer):
            cur_dropout_ratio = dropout_ratio[k] if k < len(dropout_ratio) \
                      else 0.0
            self.dropout[str(k)] = nn.Dropout(p=cur_dropout_ratio)
            out_dim = fc_hidden_dims[k]
            self.FC[str(k)] = FC_BN_ReLU_Domain(in_dim, out_dim, 
                  num_domains_bn)

            in_dim = out_dim

        cur_dropout_ratio = dropout_ratio[self.num_hidden_layer] \
                  if self.num_hidden_layer < len(dropout_ratio) else 0.0

        self.dropout['logits'] = nn.Dropout(p=cur_dropout_ratio)
        self.FC['logits'] = nn.Linear(in_dim, num_classes)


        for key in self.FC:
            for m in self.FC[key].modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)

        self.proto_fc= nn.Linear(in_dim, num_classes)
        for m in self.proto_fc.modules():
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

        self.register_buffer('src_centers', (
                to_cuda(torch.rand(num_classes, feat_dim)) - 0.5) * 2)#.to(device)
        self.register_buffer('tgt_centers', (
                to_cuda(torch.rand(num_classes, feat_dim)) - 0.5) * 2)  # .to(device)

    def set_bn_domain(self, domain=0):
        assert(domain < self.num_domains_bn), \
               "The domain id exceeds the range."
        self.bn_domain = domain
        for m in self.modules():
            if isinstance(m, BatchNormDomain):
                m.set_domain(domain)
    def update_src_centers(self,centers,center_deltas=None):
        if center_deltas is not None:
            self.src_centers=centers - center_deltas
        else:
            self.src_centers=centers
    def update_tgt_centers(self,centers,center_deltas=None):
        if center_deltas is not None:
            self.tgt_centers=centers - center_deltas
        else:
            self.tgt_centers=centers
    def update_proro_fc(self,centers):
        for m in self.proto_fc.modules():
            m.weight.data.add_(centers)
        
    def forward(self, x):
        feat = self.feature_extractor(x).view(-1, self.in_dim)

        to_select = {}
        to_select['feat'] = feat

        x = feat
        for key in self.FC:
            x = self.dropout[key](x)
            x = self.FC[key](x)
            to_select[key] = x

        to_select['probs'] = F.softmax(x, dim=1)
        to_select['log_probs'] = F.log_softmax(x, dim=1)
        return to_select


class DAMCDNet(nn.Module):
    def __init__(self, num_classes, feature_extractor='resnet101',
                 fx_pretrained=True, fc_hidden_dims=[], frozen=[],
                 num_domains_bn=2, dropout_ratio=(0.5,)):
        super(DAMCDNet, self).__init__()
        self.feature_extractor = utils.find_class_by_name(
            feature_extractor, backbones)(pretrained=fx_pretrained,
                                          frozen=frozen, num_domains=num_domains_bn)

        self.bn_domain = 0
        self.num_domains_bn = num_domains_bn

        feat_dim = self.feature_extractor.out_dim
        self.in_dim = feat_dim

        self.FC = nn.ModuleDict()
        self.FC1 = nn.ModuleDict()
        self.dropout = nn.ModuleDict()
        self.num_hidden_layer = len(fc_hidden_dims)

        in_dim = feat_dim
        for k in range(self.num_hidden_layer):
            cur_dropout_ratio = dropout_ratio[k] if k < len(dropout_ratio) \
                else 0.0
            self.dropout[str(k)] = nn.Dropout(p=cur_dropout_ratio)
            out_dim = fc_hidden_dims[k]
            self.FC[str(k)] = FC_BN_ReLU_Domain(in_dim, out_dim,
                                                num_domains_bn)
            self.FC1[str(k)] = FC_BN_ReLU_Domain(in_dim, out_dim,
                                                 num_domains_bn)
            in_dim = out_dim

        cur_dropout_ratio = dropout_ratio[self.num_hidden_layer] \
            if self.num_hidden_layer < len(dropout_ratio) else 0.0

        self.dropout['logits'] = nn.Dropout(p=cur_dropout_ratio)
        self.FC['logits'] = nn.Linear(in_dim, num_classes)
        self.FC1['logits'] = nn.Linear(in_dim, num_classes)

        for key in self.FC:
            for m in self.FC[key].modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
            for m in self.FC1[key].modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
        self.proto_fc = nn.Linear(in_dim, num_classes)
        for m in self.proto_fc.modules():
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

        self.register_buffer('src_centers', (
                to_cuda(torch.rand(num_classes, feat_dim)) - 0.5) * 2)  # .to(device)
        self.register_buffer('tgt_centers', (
                to_cuda(torch.rand(num_classes, feat_dim)) - 0.5) * 2)  # .to(device)

    def set_bn_domain(self, domain=0):
        assert (domain < self.num_domains_bn), \
            "The domain id exceeds the range."
        self.bn_domain = domain
        for m in self.modules():
            if isinstance(m, BatchNormDomain):
                m.set_domain(domain)

    def update_src_centers(self, centers, center_deltas=None):
        if center_deltas is not None:
            self.src_centers = centers - center_deltas
        else:
            self.src_centers = centers

    def update_tgt_centers(self, centers, center_deltas=None):
        if center_deltas is not None:
            self.tgt_centers = centers - center_deltas
        else:
            self.tgt_centers = centers

    def update_proro_fc(self, centers):
        for m in self.proto_fc.modules():
            m.weight.data.add_(centers)

    def forward(self, x):
        feat = self.feature_extractor(x).view(-1, self.in_dim)

        to_select = {}
        to_select['feat'] = feat

        x0 = feat
        x1=feat
        for key in self.FC:
            x0 = feat
            x0 = self.dropout[key](x0)
            x0 = self.FC[key](x0)
            to_select[key] = x0

        for key in self.FC1:
            x1 = self.dropout[key](x1)
            x1 = self.FC1[key](x1)
            to_select[key+'1'] = x1

        to_select['probs'] = F.softmax(x0, dim=1)
        to_select['log_probs'] = F.log_softmax(x0, dim=1)

        to_select['probs1'] = F.softmax(x1, dim=1)
        to_select['log_probs1'] = F.log_softmax(x1, dim=1)
        return to_select

def danet(num_classes, feature_extractor, fx_pretrained=True,
          frozen=[], dropout_ratio=0.5, state_dict=None,
          fc_hidden_dims=[], num_domains_bn=1, **kwargs):

    model = DANet(feature_extractor=feature_extractor, 
                num_classes=num_classes, frozen=frozen, 
                fx_pretrained=fx_pretrained, 
                dropout_ratio=dropout_ratio, 
                fc_hidden_dims=fc_hidden_dims,
                num_domains_bn=num_domains_bn, **kwargs)

    if state_dict is not None:
        model_utils.init_weights(model, state_dict, num_domains_bn, False)

    return model

def damcdnet(num_classes, feature_extractor, fx_pretrained=True,
          frozen=[], dropout_ratio=0.5, state_dict=None,
          fc_hidden_dims=[], num_domains_bn=1, **kwargs):

    model = DAMCDNet(feature_extractor=feature_extractor,
                num_classes=num_classes, frozen=frozen,
                fx_pretrained=fx_pretrained,
                dropout_ratio=dropout_ratio,
                fc_hidden_dims=fc_hidden_dims,
                num_domains_bn=num_domains_bn, **kwargs)

    if state_dict is not None:
        model_utils.init_weights(model, state_dict, num_domains_bn, False)

    return model
class LeNet(nn.Module):
    def __init__(self, num_classes, 
                 fc_hidden_dims=[],
                 num_domains_bn=2, dropout_ratio=(0.5,)):
        super(LeNet, self).__init__()
        self.feature_extractor = LeNetEncoder()
        self.bn_domain = 0
        self.num_domains_bn = num_domains_bn

        feat_dim = self.feature_extractor.fc1.out_features
        self.in_dim = feat_dim

        self.FC = nn.ModuleDict()
        self.dropout = nn.ModuleDict()
        self.num_hidden_layer = len(fc_hidden_dims)

        in_dim = feat_dim
        for k in range(self.num_hidden_layer):
            cur_dropout_ratio = dropout_ratio[k] if k < len(dropout_ratio) \
                else 0.0
            self.dropout[str(k)] = nn.Dropout(p=cur_dropout_ratio)
            out_dim = fc_hidden_dims[k]
            self.FC[str(k)] = FC_BN_ReLU_Domain(in_dim, out_dim,
                                                num_domains_bn)
            in_dim = out_dim

        cur_dropout_ratio = dropout_ratio[self.num_hidden_layer] \
            if self.num_hidden_layer < len(dropout_ratio) else 0.0

        self.dropout['logits'] = nn.Dropout(p=cur_dropout_ratio)
        self.FC['logits'] = nn.Linear(in_dim, num_classes)

        for key in self.FC:
            for m in self.FC[key].modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
                    
    def set_bn_domain(self, domain=0):
        assert(domain < self.num_domains_bn), \
               "The domain id exceeds the range."
        self.bn_domain = domain
        for m in self.modules():
            if isinstance(m, BatchNormDomain):
                m.set_domain(domain)

    def forward(self, x):
        feat = self.feature_extractor(x).view(-1, self.in_dim)

        to_select = {}
        to_select['feat'] = feat

        x = feat
        for key in self.FC:
            x = self.dropout[key](x)
            x = self.FC[key](x)
            to_select[key] = x

        to_select['probs'] = F.softmax(x, dim=1)
        to_select['log_probs'] = F.log_softmax(x, dim=1)
        return to_select


class LeNet_sm(nn.Module):
    def __init__(self, num_classes,
                 fc_hidden_dims=[],
                 num_domains_bn=2, dropout_ratio=(0.5,)):
        super(LeNet_sm, self).__init__()
        self.feature_extractor = SMFeature()
        self.bn_domain = 0
        self.num_domains_bn = num_domains_bn

        feat_dim = self.feature_extractor.fc1.out_features
        self.in_dim = feat_dim

        self.FC = nn.ModuleDict()
        self.dropout = nn.ModuleDict()
        self.num_hidden_layer = len(fc_hidden_dims)

        in_dim = feat_dim
        for k in range(self.num_hidden_layer):
            cur_dropout_ratio = dropout_ratio[k] if k < len(dropout_ratio) \
                else 0.0
            self.dropout[str(k)] = nn.Dropout(p=cur_dropout_ratio)
            out_dim = fc_hidden_dims[k]
            self.FC[str(k)] = FC_BN_ReLU_Domain(in_dim, out_dim,
                                                num_domains_bn)
            in_dim = out_dim

        cur_dropout_ratio = dropout_ratio[self.num_hidden_layer] \
            if self.num_hidden_layer < len(dropout_ratio) else 0.0

        self.dropout['logits'] = nn.Dropout(p=cur_dropout_ratio)
        self.FC['logits'] = nn.Linear(in_dim, num_classes)

        for key in self.FC:
            for m in self.FC[key].modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def set_bn_domain(self, domain=0):
        assert (domain < self.num_domains_bn), \
            "The domain id exceeds the range."
        self.bn_domain = domain
        for m in self.modules():
            if isinstance(m, BatchNormDomain):
                m.set_domain(domain)

    def forward(self, x):
        feat = self.feature_extractor(x).view(-1, self.in_dim)

        to_select = {}
        to_select['feat'] = feat

        x = feat
        for key in self.FC:
            x = self.dropout[key](x)
            x = self.FC[key](x)
            to_select[key] = x

        to_select['probs'] = F.softmax(x, dim=1)
        to_select['log_probs'] = F.log_softmax(x, dim=1)
        return to_select

def lenet(num_classes,
          dropout_ratio=0.5, state_dict=None,
          fc_hidden_dims=[], num_domains_bn=1):
    model = LeNet(num_classes=num_classes,
                  dropout_ratio=dropout_ratio,
                  fc_hidden_dims=fc_hidden_dims,
                  num_domains_bn=num_domains_bn)

    if state_dict is not None:
        model_utils.init_weights(model, state_dict, num_domains_bn, False)

    return model




## for self-paced training in target domain  
def spnet(num_classes, feature_extractor, fx_pretrained=True,
          frozen=[], dropout_ratio=0.5, state_dict=None,
          fc_hidden_dims=[], num_domains_bn=1, **kwargs):

    model = DANet(feature_extractor=feature_extractor, 
                num_classes=num_classes, frozen=frozen, 
                fx_pretrained=fx_pretrained, 
                dropout_ratio=dropout_ratio, 
                fc_hidden_dims=fc_hidden_dims,
                num_domains_bn=num_domains_bn, **kwargs)

    if state_dict is not None:
        model_utils.init_weights(model, state_dict, num_domains_bn, False)

    return model