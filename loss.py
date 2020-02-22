import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.cutmix import cutmix_criterion


def get_criterion(loss_type, **kwargs):
    if loss_type == 'ce':
        return F.cross_entropy
    elif loss_type == 'ohem':
        return OHEMCrossEntropyLoss(**kwargs).cuda()
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
