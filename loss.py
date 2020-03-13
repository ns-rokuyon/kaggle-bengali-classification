import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_toolbelt.losses import FocalLoss
from lib.cutmix import cutmix_criterion


def get_criterion(loss_type,
                  weights=None, dim=None,
                  n_class=None, s=30, **kwargs):
    if loss_type == 'ce':
        return F.cross_entropy
    elif loss_type == 'weighted_ce':
        return nn.CrossEntropyLoss(weight=weights).cuda()
    elif loss_type == 'ohem':
        return OHEMCrossEntropyLoss(**kwargs).cuda()
    elif loss_type == 'ns':
        return NormSoftmaxLoss(dim, n_class).cuda()
    elif loss_type == 'af':
        return ArcFaceLoss(dim, n_class, s=s, m=0.4).cuda()
    elif loss_type == 'focal':
        return FocalLoss().cuda()
    elif loss_type == 'reduced_focal':
        return FocalLoss(reduced_threshold=0.5).cuda()
    else:
        raise ValueError(loss_type)


class OHEMCrossEntropyLoss(nn.Module):
    def __init__(self, rate=0.7, **kwargs):
        super().__init__()
        self.rate = rate

    def forward(self, logit, t):
        batch_size = logit.size(0)
        loss = F.cross_entropy(logit, t,
                               reduction='none',
                               ignore_index=-1)
        loss, _ = loss.topk(k=int(self.rate * batch_size))
        return torch.mean(loss)

    def adjust_rate(self, epoch):
        current_rate = self.rate
        if epoch == 1:
            self.rate = 1.0
        elif epoch == 21:
            self.rate = 0.8
        elif epoch == 41:
            self.rate = 0.7
        elif epoch == 61:
            self.rate = 0.6
        elif epoch == 81:
            self.rate = 0.5
        return current_rate, self.rate


class NormSoftmaxLoss(nn.Module):
    """http://github.com/azgo14/classification_metric_learning
    """
    def __init__(self, dim, n_class, temperature=0.05):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(n_class, dim))
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, t):
        norm_x = F.normalize(x)
        norm_weight = F.normalize(self.weight, dim=1)
        logit = F.linear(norm_x, norm_weight)

        loss = self.loss_fn(logit / self.temperature, t)
        return loss


class ArcFaceLoss(nn.Module):
    def __init__(self, dim, n_class, s=30.0, m=0.5):
        super().__init__()
        self.dim = dim
        self.n_class = n_class
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.weight = nn.Parameter(
            torch.FloatTensor(self.n_class, self.dim)
        )
        nn.init.xavier_uniform_(self.weight)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, t):
        norm_x = F.normalize(x)
        norm_weight = F.normalize(self.weight, dim=1)
        cos = F.linear(norm_x, norm_weight)
        sin = torch.sqrt(1.0 - torch.pow(cos, 2))
        phi = cos * self.cos_m - sin * self.sin_m
        phi = torch.where(cos > self.th, phi, cos - self.mm)

        one_hot = torch.zeros(cos.size()).to(x.device)
        one_hot.scatter_(1, t.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cos)
        output *= self.s

        loss = self.loss_fn(output, t)
        return loss


def focal_loss(input, target, OHEM_percent=None, n_class=None):
    """https://github.com/SeuTao/Humpback-Whale-Identification-Challenge-2019_2nd_palce_solution/blob/master/loss/loss.py
    """
    gamma = 2
    assert target.size() == input.size()

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    invprobs = F.logsigmoid(-input * (target * 2 - 1))
    loss = (invprobs * gamma).exp() * loss

    if OHEM_percent is None:
        return loss.mean()
    else:
        OHEM, _ = loss.topk(k=int(n_class * OHEM_percent), dim=1, largest=True, sorted=True)
        return OHEM.mean()


class SoftTripleLoss(nn.Module):
    """https://raw.githubusercontent.com/idstcv/SoftTriple/master/loss/SoftTriple.py
    """
    def __init__(self, dim, cN, K=2, la=20, gamma=0.1, tau=0.2, margin=0.01):
        super().__init__()
        self.la = la
        self.gamma = 1./gamma
        self.tau = tau
        self.margin = margin
        self.cN = cN
        self.K = K
        self.fc = nn.Parameter(torch.Tensor(dim, cN*K))
        self.weight = torch.zeros(cN*K, cN*K, dtype=torch.bool).cuda()
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i*K+j, i*K+j+1:(i+1)*K] = 1
        nn.init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

    def forward(self, input, target):
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc*self.gamma, dim=2)
        simClass = torch.sum(prob*simStruc, dim=2)
        marginM = torch.zeros(simClass.shape).cuda()
        marginM[torch.arange(0, marginM.shape[0]), target] = self.margin
        lossClassify = F.cross_entropy(self.la*(simClass-marginM), target)
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.cN*self.K*(self.K-1.))
            return lossClassify+self.tau*reg
        else:
            return lossClassify
