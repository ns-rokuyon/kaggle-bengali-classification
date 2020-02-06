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