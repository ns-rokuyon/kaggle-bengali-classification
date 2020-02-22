import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
from typing import Tuple, List


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
