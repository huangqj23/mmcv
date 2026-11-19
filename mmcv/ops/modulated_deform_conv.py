# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.logging import print_log
from mmengine.registry import MODELS
from mmengine.utils import deprecated_api_warning
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair, _single

from ..utils import ext_loader


def _prepare(input, offset, mask, weight, bias):
    if input is not None and input.dim() != 4:
        raise ValueError(
            f'Expected 4D tensor as input, got {input.dim()}D tensor instead.')
    # Under AMP "offset" comes from an nn.Conv2d and is cast to float16
    # while the rest may stay float32, so the type of offset decides the
    # precision.
    input = input.type_as(offset)
    weight = weight.type_as(input)
    if bias is not None:
        bias = bias.type_as(input)
    return input, mask.type_as(input), weight, bias


def _op_args(weight, stride, padding, dilation, groups, deform_groups):
    """Arguments of ``mmcv::modulated_deform_conv_forward`` after the
    tensors."""
    stride, padding, dilation = _pair(stride), _pair(padding), _pair(dilation)
    return (weight.size(2), weight.size(3), stride[0], stride[1], padding[0],
            padding[1], dilation[0], dilation[1], int(groups),
            int(deform_groups))


def _save(ctx, input, weight, bias, offset, mask, params):
    ctx.save_for_backward(input, weight, bias, offset, mask)
    ctx.params = params


def _grads(ctx, grad_output):
    """Gradients of input, weight, bias (None without a bias), offset and
    mask."""
    input, weight, bias, offset, mask = ctx.saved_tensors
    grads = torch.ops.mmcv.modulated_deform_conv_backward(
        input, weight, bias, offset, mask, grad_output, *ctx.params)
    if bias is None:
        grads = grads[:2] + (None, ) + grads[3:]
    return grads


def _setup_context(ctx, inputs, output):
    input, weight, bias, offset, mask, *params = inputs
    _save(ctx, input, weight, bias, offset, mask, params)


def _backward(ctx, grad_output):
    # schema order: input, weight, bias, offset, mask, then 10 ints
    return tuple(_grads(ctx, grad_output)) + (None, ) * 10


@ext_loader.on_load
def _register():

    @torch.library.register_fake('mmcv::modulated_deform_conv_forward')
    def _(input, weight, bias, offset, mask, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
          deformable_group):
        n, _, h, w = input.shape
        out_h = (h + 2 * pad_h -
                 (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
        out_w = (w + 2 * pad_w -
                 (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
        return input.new_empty((n, weight.size(0), out_h, out_w))

    @torch.library.register_fake('mmcv::modulated_deform_conv_backward')
    def _(input, weight, bias, offset, mask, grad_output, *_):
        grad_bias = (
            bias.new_empty(bias.shape)
            if bias is not None else input.new_empty((0, )))
        return (input.new_empty(input.shape), weight.new_empty(weight.shape),
                grad_bias, offset.new_empty(offset.shape),
                mask.new_empty(mask.shape))

    torch.library.register_autograd(
        'mmcv::modulated_deform_conv_forward',
        _backward,
        setup_context=_setup_context)


class ModulatedDeformConv2dFunction(Function):
    """:func:`modulated_deform_conv2d` as an autograd function, kept for the
    TorchScript ONNX exporter (``symbolic``) and for
    ``ModulatedDeformConv2dFunction.apply``."""

    @staticmethod
    def symbolic(g, input, offset, mask, weight, bias, stride, padding,
                 dilation, groups, deform_groups):
        input_tensors = [input, offset, mask, weight]
        if bias is not None:
            input_tensors.append(bias)
        return g.op(
            'mmcv::MMCVModulatedDeformConv2d',
            *input_tensors,
            stride_i=stride,
            padding_i=padding,
            dilation_i=dilation,
            groups_i=groups,
            deform_groups_i=deform_groups)

    @staticmethod
    def forward(ctx,
                input: torch.Tensor,
                offset: torch.Tensor,
                mask: torch.Tensor,
                weight: nn.Parameter,
                bias: Optional[nn.Parameter] = None,
                stride: int = 1,
                padding: int = 0,
                dilation: int = 1,
                groups: int = 1,
                deform_groups: int = 1) -> torch.Tensor:
        input, mask, weight, bias = _prepare(input, offset, mask, weight, bias)
        params = _op_args(weight, stride, padding, dilation, groups,
                          deform_groups)
        ext_loader.load()
        _save(ctx, input, weight, bias, offset, mask, params)
        return torch.ops.mmcv.modulated_deform_conv_forward(
            input, weight, bias, offset, mask, *params)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        grad_input, grad_weight, grad_bias, grad_offset, grad_mask = _grads(
            ctx, grad_output)
        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
                None, None, None, None, None)


def modulated_deform_conv2d(input: torch.Tensor,
                            offset: torch.Tensor,
                            mask: torch.Tensor,
                            weight: nn.Parameter,
                            bias: Optional[nn.Parameter] = None,
                            stride: int = 1,
                            padding: int = 0,
                            dilation: int = 1,
                            groups: int = 1,
                            deform_groups: int = 1) -> torch.Tensor:
    if torch.onnx.is_in_onnx_export():
        return ModulatedDeformConv2dFunction.apply(input, offset, mask, weight,
                                                   bias, stride, padding,
                                                   dilation, groups,
                                                   deform_groups)
    input, mask, weight, bias = _prepare(input, offset, mask, weight, bias)
    ext_loader.load()
    return torch.ops.mmcv.modulated_deform_conv_forward(
        input, weight, bias, offset, mask,
        *_op_args(weight, stride, padding, dilation, groups, deform_groups))


class ModulatedDeformConv2d(nn.Module):

    @deprecated_api_warning({'deformable_groups': 'deform_groups'},
                            cls_name='ModulatedDeformConv2d')
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]],
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 deform_groups: int = 1,
                 bias: Union[bool, str] = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups,
                         *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x: torch.Tensor, offset: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


@MODELS.register_module('DCNv2')
class ModulatedDeformConv2dPack(ModulatedDeformConv2d):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True)
        self.init_weights()

    def init_weights(self) -> None:
        super().init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if version is None or version < 2:
            # the key is different in early versions
            # In version < 2, ModulatedDeformConvPack
            # loads previous benchmark models.
            if (prefix + 'conv_offset.weight' not in state_dict
                    and prefix[:-1] + '_offset.weight' in state_dict):
                state_dict[prefix + 'conv_offset.weight'] = state_dict.pop(
                    prefix[:-1] + '_offset.weight')
            if (prefix + 'conv_offset.bias' not in state_dict
                    and prefix[:-1] + '_offset.bias' in state_dict):
                state_dict[prefix +
                           'conv_offset.bias'] = state_dict.pop(prefix[:-1] +
                                                                '_offset.bias')

        if version is not None and version > 1:
            print_log(
                f'ModulatedDeformConvPack {prefix.rstrip(".")} is upgraded to '
                'version 2.',
                logger='current')

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

