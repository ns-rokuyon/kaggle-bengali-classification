import torch
import torch.nn as nn
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


class MultiHeadClassifier(nn.Module):
    def __init__(self, in_channel, n_grapheme=168, n_vowel=11, n_consonant=7):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.pool = nn.AdaptiveAvgPool2d(1)
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
    def __init__(self, pretrained=True):
        super().__init__()
        self.backend = make_backend_resnet34(pretrained=pretrained)
        self.multihead = MultiHeadClassifier(512)

    def forward(self, x):
        x = self.backend(x)
        logit_g, logit_v, logit_c = self.multihead(x)
        return logit_g, logit_v, logit_c


class BengaliSEResNeXt50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backend = make_backend_se_resnext50_32x4d(pretrained=pretrained)
        self.multihead = MultiHeadClassifier(2048)

    def forward(self, x):
        x = self.backend.features(x)
        logit_g, logit_v, logit_c = self.multihead(x)
        return logit_g, logit_v, logit_c


def create_init_model(arch, **kwargs):
    if arch == 'BengaliResNet34':
        model = BengaliResNet34(**kwargs)
    elif arch == 'BengaliSEResNeXt50':
        model = BengaliSEResNeXt50(**kwargs)
    else:
        raise ValueError(arch)

    return model