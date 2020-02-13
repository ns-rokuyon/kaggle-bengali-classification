import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
from torchvision.models import resnet34, resnet50


def make_model_1ch_input(model):
    layers = list(model.children())
    weight = layers[0].weight
    layers[0] = nn.Conv2d(1, 64,
                          kernel_size=7,
                          stride=2,
                          padding=2,
                          bias=False)
    layers[0].weight = nn.Parameter(torch.mean(weight, dim=1, keepdim=True))
    return nn.Sequential(*layers)


def make_backend_resnet34(pretrained=True):
    model = resnet34(pretrained=pretrained)
    layers = list(model.children())[:-2]
    backend = nn.Sequential(*layers)
    return make_model_1ch_input(backend)


def make_backend_se_resnext50_32x4d(pretrained=True):
    model = pretrainedmodels.se_resnext50_32x4d(
        pretrained='imagenet' if pretrained else None
    )
    weight = model.layer0[0].weight
    conv1 = nn.Conv2d(1, 64,
                      kernel_size=7, stride=2, padding=3, bias=False)
    conv1.weight = nn.Parameter(torch.mean(weight, dim=1, keepdim=True))
    layer0 = nn.Sequential(
        conv1,
        model.layer0[1],
        model.layer0[2],
        model.layer0[3]
    )
    model.layer0 = layer0
    del model.avg_pool
    del model.dropout
    del model.last_linear
    return model


def make_backend_se_resnext101_32x4d(pretrained=True):
    model = pretrainedmodels.se_resnext101_32x4d(
        pretrained='imagenet' if pretrained else None
    )
    weight = model.layer0[0].weight
    conv1 = nn.Conv2d(1, 64,
                      kernel_size=7, stride=2, padding=3, bias=False)
    conv1.weight = nn.Parameter(torch.mean(weight, dim=1, keepdim=True))
    layer0 = nn.Sequential(
        conv1,
        model.layer0[1],
        model.layer0[2],
        model.layer0[3]
    )
    model.layer0 = layer0
    return model


def gemp(x, p=3, eps=1e-6):
    return F.avg_pool2d(
        x.clamp(min=eps).pow(p),
        (x.size(-2), x.size(-1))
    ).pow(1./p)


class GeMP(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gemp(x, p=self.p, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + \
            '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
            ', ' + 'eps=' + str(self.eps) + ')'


def global_pooling(pooling_type='gap'):
    if pooling_type == 'gap':
        pool = nn.AdaptiveAvgPool2d(1)
    elif pooling_type == 'gemp':
        pool = GeMP(p=3)
    else:
        raise ValueError(f'Invalid pooling: {pooling_type}')
    return pool


class MultiHeadCenterClassifier(nn.Module):
    def __init__(self, in_channel, dim=64,
                 temperature=0.05,
                 n_grapheme=168, n_vowel=11, n_consonant=7,
                 pooling='gap'):
        super().__init__()
        self.temperature = temperature
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.pool = global_pooling(pooling_type=pooling)

        self.W_g = torch.nn.Parameter(torch.Tensor(self.n_grapheme, dim))
        self.W_v = torch.nn.Parameter(torch.Tensor(self.n_vowel, dim))
        self.W_c = torch.nn.Parameter(torch.Tensor(self.n_consonant, dim))

        stdv = 1.0 / math.sqrt(dim)
        self.W_g.data.uniform_(-stdv, stdv)
        self.W_v.data.uniform_(-stdv, stdv)
        self.W_c.data.uniform_(-stdv, stdv)

        self.head_g = nn.Sequential(
            nn.Linear(in_channel, dim),
            nn.BatchNorm1d(dim)
        )
        self.head_v = nn.Sequential(
            nn.Linear(in_channel, dim),
            nn.BatchNorm1d(dim)
        )
        self.head_c = nn.Sequential(
            nn.Linear(in_channel, dim),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        logit_g = F.linear(
            F.normalize(self.head_g(x)),
            F.normalize(self.W_g)
        ) / self.temperature

        logit_v = F.linear(
            F.normalize(self.head_v(x)),
            F.normalize(self.W_v)
        ) / self.temperature

        logit_c = F.linear(
            F.normalize(self.head_c(x)),
            F.normalize(self.W_c)
        ) / self.temperature

        return logit_g, logit_v, logit_c


class MultiHeadCenterClassifier2(nn.Module):
    def __init__(self, in_channel, dim=64,
                 temperature=0.05,
                 n_grapheme=168, n_vowel=11, n_consonant=7,
                 pooling='gap'):
        super().__init__()
        self.temperature = temperature
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.pool = global_pooling(pooling_type=pooling)

        self.W_g = torch.nn.Parameter(torch.Tensor(self.n_grapheme, dim))
        self.W_v = torch.nn.Parameter(torch.Tensor(self.n_vowel, dim))
        self.W_c = torch.nn.Parameter(torch.Tensor(self.n_consonant, dim))

        stdv = 1.0 / math.sqrt(dim)
        self.W_g.data.uniform_(-stdv, stdv)
        self.W_v.data.uniform_(-stdv, stdv)
        self.W_c.data.uniform_(-stdv, stdv)

        self.head = nn.Sequential(
            nn.Linear(in_channel, dim),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        x = self.head(x)

        logit_g = F.linear(
            F.normalize(x),
            F.normalize(self.W_g)
        ) / self.temperature

        logit_v = F.linear(
            F.normalize(x),
            F.normalize(self.W_v)
        ) / self.temperature

        logit_c = F.linear(
            F.normalize(x),
            F.normalize(self.W_c)
        ) / self.temperature

        return logit_g, logit_v, logit_c


class MultiHeadClassifier(nn.Module):
    def __init__(self, in_channel,
                 n_grapheme=168, n_vowel=11, n_consonant=7,
                 pooling='gap'):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.pool = global_pooling(pooling_type=pooling)
        self.head_g = nn.Sequential(
            nn.BatchNorm1d(in_channel),
            nn.Dropout(0.5),
            nn.Linear(in_channel, n_grapheme)
        )
        self.head_v = nn.Sequential(
            nn.BatchNorm1d(in_channel),
            nn.Dropout(0.5),
            nn.Linear(in_channel, n_vowel)
        )
        self.head_c = nn.Sequential(
            nn.BatchNorm1d(in_channel),
            nn.Dropout(0.5),
            nn.Linear(in_channel, n_consonant)
        )

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        logit_g = self.head_g(x)
        logit_v = self.head_v(x)
        logit_c = self.head_c(x)

        return logit_g, logit_v, logit_c
        

class BengaliResNet34(nn.Module):
    def __init__(self,
                 pretrained=True,
                 pooling='gap',
                 **kwargs):
        super().__init__()
        self.backend = make_backend_resnet34(pretrained=pretrained)
        self.multihead = MultiHeadClassifier(512, pooling=pooling)

    def forward(self, x):
        x = self.backend(x)
        logit_g, logit_v, logit_c = self.multihead(x)
        return logit_g, logit_v, logit_c


class BengaliSEResNeXt50(nn.Module):
    def __init__(self,
                 pretrained=True,
                 pooling='gap',
                 **kwargs):
        super().__init__()
        self.backend = make_backend_se_resnext50_32x4d(pretrained=pretrained)
        self.multihead = MultiHeadClassifier(2048, pooling=pooling)

    def forward(self, x):
        x = self.backend.features(x)
        logit_g, logit_v, logit_c = self.multihead(x)
        return logit_g, logit_v, logit_c


class BengaliSEResNeXt50NS(nn.Module):
    def __init__(self,
                 pretrained=True,
                 pooling='gap',
                 dim=64,
                 **kwargs):
        super().__init__()
        self.backend = make_backend_se_resnext50_32x4d(pretrained=pretrained)
        self.multihead = MultiHeadCenterClassifier2(2048, dim=dim, pooling=pooling)

    def forward(self, x):
        x = self.backend.features(x)
        feat_g, feat_v, feat_c = self.multihead(x)
        return feat_g, feat_v, feat_c


def create_init_model(arch, **kwargs):
    if arch == 'BengaliResNet34':
        model = BengaliResNet34(**kwargs)
    elif arch == 'BengaliSEResNeXt50':
        model = BengaliSEResNeXt50(**kwargs)
    elif arch == 'BengaliSEResNeXt50NS':
        model = BengaliSEResNeXt50NS(**kwargs)
    else:
        raise ValueError(arch)

    return model