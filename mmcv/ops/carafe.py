# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import normal_init, xavier_init
from mmengine.registry import MODELS
from torch import Tensor
from torch.autograd import Function
from torch.nn.modules.module import Module

from ..utils import ext_loader


def _op_args(features: Tensor, masks: Tensor, kernel_size: int,
             group_size: int, scale_factor: int) -> Tuple[int, int, int]:
    """Check the inputs; the arguments of ``mmcv::carafe[_naive]_forward``
    after ``features`` and ``masks``."""
    assert scale_factor >= 1
    assert masks.size(1) == kernel_size * kernel_size * group_size
    assert masks.size(-1) == features.size(-1) * scale_factor
    assert masks.size(-2) == features.size(-2) * scale_factor
    assert features.size(1) % group_size == 0
    assert (kernel_size - 1) % 2 == 0 and kernel_size >= 1
    return int(kernel_size), int(group_size), int(scale_factor)


def _save(ctx, features, masks, params):
    ctx.save_for_backward(features, masks)
    ctx.params = params


def _setup_context(ctx, inputs, output):
    features, masks, *params = inputs
    _save(ctx, features, masks, params)


def _make_backward(name):
    """Backward of ``mmcv::<name>_forward``: gradients of features, masks."""

    def backward(ctx, grad_output):
        features, masks = ctx.saved_tensors
        backward_op = getattr(torch.ops.mmcv, f'{name}_backward')
        return backward_op(grad_output, features, masks, *ctx.params)

    return backward


@ext_loader.on_load
def _register():
    for name in ('carafe_naive', 'carafe'):

        @torch.library.register_fake(f'mmcv::{name}_forward')
        def _(features, masks, kernel_size, group_size, scale_factor):
            n, c, h, w = features.shape
            return features.new_empty(
                (n, c, h * scale_factor, w * scale_factor))

        @torch.library.register_fake(f'mmcv::{name}_backward')
        def _(top_grad, features, masks, *_):
            return (features.new_empty(features.shape),
                    masks.new_empty(masks.shape))

        backward = _make_backward(name)
        torch.library.register_autograd(
            f'mmcv::{name}_forward',
            lambda ctx, grad, backward=backward: backward(ctx, grad) +
            (None, ) * 3,
            setup_context=_setup_context)


class CARAFENaiveFunction(Function):
    """:func:`carafe_naive` as an autograd function, kept for the TorchScript
    ONNX exporter (``symbolic``) and for ``CARAFENaiveFunction.apply``."""

    @staticmethod
    def symbolic(g, features: Tensor, masks: Tensor, kernel_size: int,
                 group_size: int, scale_factor: int) -> Tensor:
        return g.op(
            'mmcv::MMCVCARAFENaive',
            features,
            masks,
            kernel_size_i=kernel_size,
            group_size_i=group_size,
            scale_factor_f=scale_factor)

    @staticmethod
    def forward(ctx, features: Tensor, masks: Tensor, kernel_size: int,
                group_size: int, scale_factor: int) -> Tensor:
        params = _op_args(features, masks, kernel_size, group_size,
                          scale_factor)
        ext_loader.load()
        _save(ctx, features, masks, params)
        return torch.ops.mmcv.carafe_naive_forward(features, masks, *params)

    @staticmethod
    def backward(
            ctx,
            grad_output: Tensor) -> Tuple[Tensor, Tensor, None, None, None]:
        return _make_backward('carafe_naive')(ctx, grad_output) + (None, ) * 3


def carafe_naive(features: Tensor, masks: Tensor, kernel_size: int,
                 group_size: int, scale_factor: int) -> Tensor:
    if torch.onnx.is_in_onnx_export():
        return CARAFENaiveFunction.apply(features, masks, kernel_size,
                                         group_size, scale_factor)
    params = _op_args(features, masks, kernel_size, group_size, scale_factor)
    ext_loader.load()
    return torch.ops.mmcv.carafe_naive_forward(features, masks, *params)


class CARAFENaive(Module):

    def __init__(self, kernel_size: int, group_size: int, scale_factor: int):
        super().__init__()

        assert isinstance(kernel_size, int) and isinstance(
            group_size, int) and isinstance(scale_factor, int)
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.scale_factor = scale_factor

    def forward(self, features: Tensor, masks: Tensor) -> Tensor:
        return carafe_naive(features, masks, self.kernel_size, self.group_size,
                            self.scale_factor)


class CARAFEFunction(Function):
    """:func:`carafe` as an autograd function, kept for the TorchScript ONNX
    exporter (``symbolic``) and for ``CARAFEFunction.apply``."""

    @staticmethod
    def symbolic(g, features: Tensor, masks: Tensor, kernel_size: int,
                 group_size: int, scale_factor: int) -> Tensor:
        return g.op(
            'mmcv::MMCVCARAFE',
            features,
            masks,
            kernel_size_i=kernel_size,
            group_size_i=group_size,
            scale_factor_f=scale_factor)

    @staticmethod
    def forward(ctx, features: Tensor, masks: Tensor, kernel_size: int,
                group_size: int, scale_factor: int) -> Tensor:
        params = _op_args(features, masks, kernel_size, group_size,
                          scale_factor)
        ext_loader.load()
        _save(ctx, features, masks, params)
        return torch.ops.mmcv.carafe_forward(features, masks, *params)

    @staticmethod
    def backward(
            ctx,
            grad_output: Tensor) -> Tuple[Tensor, Tensor, None, None, None]:
        return _make_backward('carafe')(ctx, grad_output) + (None, ) * 3


def carafe(features: Tensor, masks: Tensor, kernel_size: int, group_size: int,
           scale_factor: int) -> Tensor:
    if torch.onnx.is_in_onnx_export():
        return CARAFEFunction.apply(features, masks, kernel_size, group_size,
                                    scale_factor)
    params = _op_args(features, masks, kernel_size, group_size, scale_factor)
    ext_loader.load()
    return torch.ops.mmcv.carafe_forward(features, masks, *params)


class CARAFE(Module):
    """ CARAFE: Content-Aware ReAssembly of FEatures

    Please refer to `CARAFE: Content-Aware ReAssembly of FEatures
    <https://arxiv.org/abs/1905.02188>`_ for more details.

    Args:
        kernel_size (int): reassemble kernel size
        group_size (int): reassemble group size
        scale_factor (int): upsample ratio

    Returns:
        upsampled feature map
    """

    def __init__(self, kernel_size: int, group_size: int, scale_factor: int):
        super().__init__()

        assert isinstance(kernel_size, int) and isinstance(
            group_size, int) and isinstance(scale_factor, int)
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.scale_factor = scale_factor

    def forward(self, features: Tensor, masks: Tensor) -> Tensor:
        return carafe(features, masks, self.kernel_size, self.group_size,
                      self.scale_factor)


@MODELS.register_module(name='carafe')
class CARAFEPack(nn.Module):
    """A unified package of CARAFE upsampler that contains: 1) channel
    compressor 2) content encoder 3) CARAFE op.

    Official implementation of ICCV 2019 paper
    `CARAFE: Content-Aware ReAssembly of FEatures
    <https://arxiv.org/abs/1905.02188>`_.

    Args:
        channels (int): input feature channels
        scale_factor (int): upsample ratio
        up_kernel (int): kernel size of CARAFE op
        up_group (int): group size of CARAFE op
        encoder_kernel (int): kernel size of content encoder
        encoder_dilation (int): dilation of content encoder
        compressed_channels (int): output channels of channels compressor

    Returns:
        upsampled feature map
    """

    def __init__(self,
                 channels: int,
                 scale_factor: int,
                 up_kernel: int = 5,
                 up_group: int = 1,
                 encoder_kernel: int = 3,
                 encoder_dilation: int = 1,
                 compressed_channels: int = 64):
        super().__init__()
        self.channels = channels
        self.scale_factor = scale_factor
        self.up_kernel = up_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        self.channel_compressor = nn.Conv2d(channels, self.compressed_channels,
                                            1)
        self.content_encoder = nn.Conv2d(
            self.compressed_channels,
            self.up_kernel * self.up_kernel * self.up_group *
            self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder, std=0.001)

    def kernel_normalizer(self, mask: Tensor) -> Tensor:
        mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        # use float division explicitly,
        # to void inconsistency while exporting to onnx
        mask_channel = int(mask_c / float(self.up_kernel**2))
        mask = mask.view(n, mask_channel, -1, h, w)

        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_c, h, w).contiguous()

        return mask

    def feature_reassemble(self, x: Tensor, mask: Tensor) -> Tensor:
        x = carafe(x, mask, self.up_kernel, self.up_group, self.scale_factor)
        return x

    def forward(self, x: Tensor) -> Tensor:
        compressed_x = self.channel_compressor(x)
        mask = self.content_encoder(compressed_x)
        mask = self.kernel_normalizer(mask)

        x = self.feature_reassemble(x, mask)
        return x
