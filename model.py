import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
from torchvision.models import resnet34, resnet50
from typing import Tuple, List


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


def make_backend_resnet34(pretrained=True, use_maxblurpool=False):
    model = resnet34(pretrained=pretrained)
    layers = list(model.children())[:-2]
    backend = nn.Sequential(*layers)
    if use_maxblurpool:
        backend[3] = MaxBlurPool2d(3, True)
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


def set_batchnorm_eval(module):
    """
    >>> model.train()
    >>> model.apply(set_batchnorm_eval)
    """
    classname = module.__class__.__name__
    if classname.find('BatchNorm') != -1:
        module.eval()


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


def normalize_kernel2d(input: torch.Tensor) -> torch.Tensor:
    r"""Normalizes both derivative and smoothing kernel.
    """
    if len(input.size()) < 2:
        raise TypeError("input should be at least 2D tensor. Got {}"
                        .format(input.size()))
    norm: torch.Tensor = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))


def compute_padding(kernel_size: Tuple[int, int]) -> List[int]:
    """Computes padding tuple."""
    # 4 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    assert len(kernel_size) == 2, kernel_size
    computed = [k // 2 for k in kernel_size]

    # for even kernels we need to do asymetric padding :(
    return [computed[1] - 1 if kernel_size[0] % 2 == 0 else computed[1],
            computed[1],
            computed[0] - 1 if kernel_size[1] % 2 == 0 else computed[0],
            computed[0]]


def filter2D(input: torch.Tensor, kernel: torch.Tensor,
             border_type: str = 'reflect',
             normalized: bool = False) -> torch.Tensor:
    r"""Function that convolves a tensor with a kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH, kW)`.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): If True, kernel will be L1 normalized.

    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Input kernel type is not a torch.Tensor. Got {}"
                        .format(type(kernel)))

    if not isinstance(border_type, str):
        raise TypeError("Input border_type is not string. Got {}"
                        .format(type(kernel)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))

    if not len(kernel.shape) == 3:
        raise ValueError("Invalid kernel shape, we expect 1xHxW. Got: {}"
                         .format(kernel.shape))

    borders_list: List[str] = ['constant', 'reflect', 'replicate', 'circular']
    if border_type not in borders_list:
        raise ValueError("Invalid border_type, we expect the following: {0}."
                         "Got: {1}".format(borders_list, border_type))

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(0).to(input.device).to(input.dtype)
    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)
    # pad the input tensor
    height, width = tmp_kernel.shape[-2:]
    padding_shape: List[int] = compute_padding((height, width))
    input_pad: torch.Tensor = F.pad(input, padding_shape, mode=border_type)
    b, c, hp, wp = input_pad.shape
    # convolve the tensor with the kernel. Pick the fastest alg
    kernel_numel: int = height * width
    if kernel_numel > 81:
        return F.conv2d(input_pad.reshape(b * c, 1, hp, wp), tmp_kernel, padding=0, stride=1).view(b, c, h, w)
    return F.conv2d(input_pad, tmp_kernel.expand(c, -1, -1, -1), groups=c, padding=0, stride=1)


def _get_pyramid_gaussian_kernel() -> torch.Tensor:
    """Utility function that return a pre-computed gaussian kernel."""
    return torch.tensor([[
        [1., 4., 6., 4., 1.],
        [4., 16., 24., 16., 4.],
        [6., 24., 36., 24., 6.],
        [4., 16., 24., 16., 4.],
        [1., 4., 6., 4., 1.]
    ]]) / 256.


class PyrDown(nn.Module):
    r"""Blurs a tensor and downsamples it.

    Args:
        borde_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.

    Return:
        torch.Tensor: the downsampled tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H / 2, W / 2)`

    Examples:
        >>> input = torch.rand(1, 2, 4, 4)
        >>> output = kornia.transform.PyrDown()(input)  # 1x2x2x2
    """

    def __init__(self, border_type: str = 'reflect') -> None:
        super(PyrDown, self).__init__()
        self.border_type: str = border_type
        self.kernel: torch.Tensor = _get_pyramid_gaussian_kernel()

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # blur image
        x_blur: torch.Tensor = filter2D(input, self.kernel, self.border_type)

        # downsample.
        out: torch.Tensor = F.interpolate(x_blur, scale_factor=0.5, mode='bilinear',
                                          align_corners=False)
        return out



def pyrdown(
        input: torch.Tensor,
        border_type: str = 'reflect') -> torch.Tensor:
    r"""Blurs a tensor and downsamples it.
    See :class:`~kornia.transform.PyrDown` for details.

    https://github.com/kornia/kornia/blob/master/kornia/geometry/transform/pyramid.py
    """
    return PyrDown(border_type)(input)


def _compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
    """Computes zero padding tuple."""
    padding = [(k - 1) // 2 for k in kernel_size]
    return padding[0], padding[1]


class MaxBlurPool2d(nn.Module):
    r"""Creates a module that computes pools and blurs and downsample a given
    feature map.

    See :cite:`zhang2019shiftinvar` for more details.

    https://raw.githubusercontent.com/kornia/kornia/master/kornia/contrib/max_blur_pool.py

    Args:
        kernel_size (int): the kernel size for max pooling..
        ceil_mode (bool): should be true to match output size of conv2d with same kernel size.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H / 2, W / 2)`

    Returns:
        torch.Tensor: the transformed tensor.

    Examples:
        >>> input = torch.rand(1, 4, 4, 8)
        >>> pool = kornia.contrib.MaxBlurPool2d(kernel_size=3)
        >>> output = pool(input)  # 1x4x2x4
    """

    def __init__(self, kernel_size: int, ceil_mode: bool = False) -> None:
        super(MaxBlurPool2d, self).__init__()
        self.ceil_mode: bool = ceil_mode
        self.kernel_size: Tuple[int, int] = (kernel_size, kernel_size)
        self.padding: Tuple[int, int] = _compute_zero_padding(self.kernel_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # compute local maxima
        x_max: torch.Tensor = F.max_pool2d(
            input, kernel_size=self.kernel_size,
            padding=self.padding, stride=1, ceil_mode=self.ceil_mode)

        # blur and downsample
        x_down: torch.Tensor = pyrdown(x_max)
        return x_down



class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes,
                 kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, inplanes, kernel_size,
            stride, padding=padding, dilation=dilation,
            groups=inplanes, bias=bias
        )
        self.bn = nn.BatchNorm2d(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class JPU(nn.Module):
    """https://github.com/wuhuikai/FastFCN/blob/master/encoding/models/deeplabv3.py
    """
    def __init__(self, in_channels=None, width=512):
        in_channels = in_channels or [512, 1024, 2048]
        assert len(in_channels) == 3
        super().__init__()
        self.out_channel = width * 4
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[0], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.d1 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=3,
                            padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.d2 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=3,
                            padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.d3 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=3,
                            padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.d4 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=3,
                            padding=8, dilation=8, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )

    def forward(self, *inputs):
        x3, x4, x5 = inputs
        f3 = self.conv3(x3)
        f4 = self.conv4(x4)
        f5 = self.conv5(x5)

        _, _, h, w = f3.size()

        f4 = F.interpolate(f4, (h, w), mode='bilinear', align_corners=True)
        f5 = F.interpolate(f5, (h, w), mode='bilinear', align_corners=True)

        f = torch.cat([f3, f4, f5], dim=1)
        f = torch.cat([self.d1(f), self.d2(f), self.d3(f), self.d4(f)], dim=1)

        return f


def ASPPConv(in_channels, out_channels, atrous_rate):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                  dilation=atrous_rate, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))


class ASPPPool(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.pool(x)
        return F.interpolate(x, (h, w),
                             mode='bilinear',
                             align_corners=True)


class ASPP(nn.Module):
    def __init__(self, in_dim, atrous_rates=(6, 12, 18)):
        super().__init__()
        out_dim = in_dim // 8
        if len(atrous_rates) == 3:
            rate1, rate2, rate3 = tuple(atrous_rates)
            rate4 = None
        elif len(atrous_rates) == 4:
            rate1, rate2, rate3, rate4 = tuple(atrous_rates)

        self.b0 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        self.b1 = ASPPConv(in_dim, out_dim, rate1)
        self.b2 = ASPPConv(in_dim, out_dim, rate2)
        self.b3 = ASPPConv(in_dim, out_dim, rate3)
        if rate4:
            self.b4 = ASPPConv(in_dim, out_dim, rate4)
        else:
            self.b4 = ASPPPool(in_dim, out_dim)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5, inplace=False)
        )

    def forward(self, x):
        x = torch.cat([
            self.b0(x),
            self.b1(x),
            self.b2(x),
            self.b3(x),
            self.b4(x),
        ], dim=1)
        return self.project(x)


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


class BengaliResNet34JPU(nn.Module):
    def __init__(self,
                 pretrained=True,
                 pooling='gap',
                 **kwargs):
        super().__init__()
        self.backend = make_backend_resnet34(pretrained=pretrained)
        self.jpu = JPU(in_channels=[128, 256, 512], width=128)
        self.aspp = ASPP(512)
        self.multihead = MultiHeadClassifier(512 // 8, pooling=pooling)

    def forward(self, x):
        x = self.backend[0](x)
        x = self.backend[1](x)
        x = self.backend[2](x)
        x = self.backend[3](x)
        x = self.backend[4](x)
        c2 = self.backend[5](x)
        c3 = self.backend[6](c2)
        c4 = self.backend[7](c3)
        x = self.jpu(c2, c3, c4)
        x = self.aspp(x)

        logit_g, logit_v, logit_c = self.multihead(x)
        return logit_g, logit_v, logit_c


class BengaliResNet34NS(nn.Module):
    def __init__(self,
                 pretrained=True,
                 pooling='gap',
                 dim=64,
                 use_maxblurpool=False,
                 **kwargs):
        super().__init__()
        self.backend = make_backend_resnet34(pretrained=pretrained,
                                             use_maxblurpool=use_maxblurpool)
        self.multihead = MultiHeadCenterClassifier2(512, dim=dim, pooling=pooling)

    def forward(self, x):
        x = self.backend(x)
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
        logit_g, logit_v, logit_c = self.multihead(x)
        return logit_g, logit_v, logit_c


def create_init_model(arch, **kwargs):
    if arch == 'BengaliResNet34':
        model = BengaliResNet34(**kwargs)
    elif arch == 'BengaliSEResNeXt50':
        model = BengaliSEResNeXt50(**kwargs)
    elif arch == 'BengaliResNet34JPU':
        model = BengaliResNet34JPU(**kwargs)
    elif arch == 'BengaliResNet34NS':
        model = BengaliResNet34NS(**kwargs)
    elif arch == 'BengaliSEResNeXt50NS':
        model = BengaliSEResNeXt50NS(**kwargs)
    else:
        raise ValueError(arch)

    return model