import torch
import torch.nn as nn
import numpy as np


def mixup(x, alpha=1.0):
    rand_index = torch.randperm(x.size(0)).cuda()
    lam = np.random.beta(alpha, alpha)
    x = lam * x + (1 - lam) * x[rand_index, :]
    return x, rand_index, lam


def mixup_criterion(logit, ta, tb, lam,
                     criterion=None):
    if criterion is None:
        criterion =  nn.CrossEntropyLoss(reduction='mean')
    return lam * criterion(logit, ta) + \
           (1 - lam) * criterion(logit, tb)