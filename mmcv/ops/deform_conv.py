# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from mmengine.registry import MODELS
from mmengine.utils import deprecated_api_warning
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair, _single

from ..utils import ext_loader

# How DeformConv2d and ModulatedDeformConv2d export to ONNX:
# - 'grid_sample' (default): deform_conv2d_pytorch, i.e. GridSample (opset
#   16) and MatMul. TensorRT, OpenVINO and ONNX Runtime run it, no plugin.
# - 'deform_conv': one ONNX DeformConv node (opset 19). TensorRT 10 maps it
#   to its stock ModulatedDeformConv2d plugin (trtexec loads it; an API user
#   calls initLibNvInferPlugins); OpenVINO 2025.4 and ONNX Runtime 1.23
#   cannot read it.
ONNX_EXPORT_MODE = 'grid_sample'


def deform_conv2d_pytorch(input: Tensor,
                          offset: Tensor,
                          weight: Tensor,
                          mask: Optional[Tensor] = None,
                          bias: Optional[Tensor] = None,
                          stride: Union[int, Tuple[int, ...]] = 1,
                          padding: Union[int, Tuple[int, ...]] = 0,
                          dilation: Union[int, Tuple[int, ...]] = 1,
                          groups: int = 1,
                          deform_groups: int = 1) -> Tensor:
    """DCN, and with ``mask`` DCNv2, in plain PyTorch.

    Every kernel point samples the input with ``grid_sample`` (bilinear,
    zero outside the input, as in the mmcv kernels), and the samples are
    multiplied with the weight. Needs no compiled ops and exports to
    standard ONNX ops. The samples take kernel_h * kernel_w times the memory
    of the input, so this is the export path, not a training path.
    """
    stride, padding, dilation = _pair(stride), _pair(padding), _pair(dilation)
    n, c, h, w = input.shape
    out_c, _, kh, kw = weight.shape
    out_h, out_w = offset.shape[-2:]
    g = deform_groups
    # sampling positions in input pixels, (n, g, kh, kw, out_h, out_w)
    offset = offset.view(n, g, kh, kw, 2, out_h, out_w)
    arange = dict(device=input.device, dtype=input.dtype)
    ys = torch.arange(out_h, **arange) * stride[0] - padding[0]
    xs = torch.arange(out_w, **arange) * stride[1] - padding[1]
    ky = torch.arange(kh, **arange) * dilation[0]
    kx = torch.arange(kw, **arange) * dilation[1]
    py = (
        ys.view(1, 1, 1, 1, -1, 1) + ky.view(1, 1, -1, 1, 1, 1) +
        offset[:, :, :, :, 0])
    px = (
        xs.view(1, 1, 1, 1, 1, -1) + kx.view(1, 1, 1, -1, 1, 1) +
        offset[:, :, :, :, 1])
    # grid_sample coordinates (align_corners=False): pixel i is centred at
    # (2 * i + 1) / size - 1
    grid = torch.stack([(2 * px + 1) / w - 1, (2 * py + 1) / h - 1], dim=-1)
    samples = F.grid_sample(
        input.reshape(n * g, c // g, h, w),
        grid.view(n * g, kh * kw * out_h, out_w, 2),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False)
    samples = samples.view(n, g, c // g, kh * kw, out_h * out_w)
    if mask is not None:
        samples = samples * mask.view(n, g, 1, kh * kw, out_h * out_w)
    # grouped convolution as a batched matrix product
    samples = samples.reshape(n, groups, c // groups * kh * kw, out_h * out_w)
    out = torch.matmul(weight.view(groups, out_c // groups, -1), samples)
    out = out.view(n, out_c, out_h, out_w)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out


def deform_conv_symbolic(g, input, weight, offset, bias, mask, stride, padding,
                         dilation, groups, deform_groups):
    """ONNX ``DeformConv`` (opset 19) for DCN and, with ``mask``, DCNv2.

    Used when ``ONNX_EXPORT_MODE == 'deform_conv'``. The offset layout
    (offset group, kernel row, kernel column, then y and x) and the zero
    padding outside the input are the same as in the mmcv kernels.
    """
    opset = getattr(g, 'opset', None)
    if opset is None:
        from torch.onnx._globals import GLOBALS
        opset = GLOBALS.export_onnx_opset_version
    if opset < 19:
        raise RuntimeError(
            'DeformConv2d and ModulatedDeformConv2d export to ONNX '
            f'DeformConv, which needs opset_version >= 19 (got {opset})')
    stride, padding, dilation = _pair(stride), _pair(padding), _pair(dilation)
    inputs = [input, weight, offset]
    if mask is not None:
        if bias is None:
            from torch.onnx.symbolic_helper import \
                _optional_input_placeholder_tensor
            bias = _optional_input_placeholder_tensor(g)
        inputs += [bias, mask]
    elif bias is not None:
        inputs.append(bias)
    return g.op(
        'DeformConv',
        *inputs,
        dilations_i=list(dilation),
        group_i=groups,
        offset_group_i=deform_groups,
        pads_i=[padding[0], padding[1], padding[0], padding[1]],
        strides_i=list(stride))


def _prepare(input: Tensor, offset: Tensor, weight: Tensor,
             im2col_step: int) -> Tuple[Tensor, Tensor]:
    if input is not None and input.dim() != 4:
        raise ValueError(
            f'Expected 4D tensor as input, got {input.dim()}D tensor instead.')
    cur_im2col_step = min(im2col_step, input.size(0))
    assert input.size(0) % cur_im2col_step == 0, \
        'batch size must be divisible by im2col_step'
    # Under AMP "offset" comes from an nn.Conv2d and is cast to float16
    # while input and weight may stay float32, so the type of offset decides
    # the precision.
    input = input.type_as(offset)
    return input, weight.type_as(input)


def _op_args(weight, stride, padding, dilation, groups, deform_groups,
             im2col_step):
    """Arguments of ``mmcv::deform_conv_forward`` after the tensors."""
    stride, padding, dilation = _pair(stride), _pair(padding), _pair(dilation)
    return (weight.size(3), weight.size(2), stride[1], stride[0], padding[1],
            padding[0], dilation[1], dilation[0], int(groups),
            int(deform_groups), int(im2col_step))


def _save(ctx, input, offset, weight, params):
    ctx.save_for_backward(input, offset, weight)
    ctx.params = params


def _grads(ctx, grad_output, need_input_offset, need_weight):
    """Gradients of input, offset and weight (None where not needed)."""
    input, offset, weight = ctx.saved_tensors
    grad_input = grad_offset = grad_weight = None
    if need_input_offset:
        grad_input, grad_offset = torch.ops.mmcv.deform_conv_backward_input(
            input, offset, grad_output, weight, *ctx.params)
    if need_weight:
        *params, im2col_step = ctx.params
        grad_weight = torch.ops.mmcv.deform_conv_backward_parameters(
            input, offset, grad_output, weight, *params, 1.0, im2col_step)
    return grad_input, grad_offset, grad_weight


def _setup_context(ctx, inputs, output):
    input, weight, offset, *params = inputs
    _save(ctx, input, offset, weight, params)


def _backward(ctx, grad_output):
    # schema order: input, weight, offset, then 11 ints
    need = ctx.needs_input_grad
    grad_input, grad_offset, grad_weight = _grads(ctx, grad_output, need[0]
                                                  or need[2], need[1])
    return (grad_input, grad_weight, grad_offset) + (None, ) * 11


@ext_loader.on_load
def _register():

    @torch.library.register_fake('mmcv::deform_conv_forward')
    def _(input, weight, offset, kW, kH, dW, dH, padW, padH, dilationW,
          dilationH, group, deformable_group, im2col_step):
        n, _, h, w = input.shape
        out_h = (h + 2 * padH - (dilationH * (kH - 1) + 1)) // dH + 1
        out_w = (w + 2 * padW - (dilationW * (kW - 1) + 1)) // dW + 1
        return input.new_empty((n, weight.size(0), out_h, out_w))

    @torch.library.register_fake('mmcv::deform_conv_backward_input')
    def _(input, offset, grad_output, weight, *_):
        return input.new_empty(input.shape), offset.new_empty(offset.shape)

    @torch.library.register_fake('mmcv::deform_conv_backward_parameters')
    def _(input, offset, grad_output, weight, *_):
        return weight.new_empty(weight.shape)

    torch.library.register_autograd(
        'mmcv::deform_conv_forward', _backward, setup_context=_setup_context)


class DeformConv2dFunction(Function):
    """:func:`deform_conv2d` as an autograd function.

    :func:`deform_conv2d` calls the registered op
    ``mmcv::deform_conv_forward``. This class remains for the TorchScript
    ONNX exporter (``symbolic``) and for code that calls
    ``DeformConv2dFunction.apply``.
    """

    @staticmethod
    def symbolic(g,
                 input,
                 offset,
                 weight,
                 stride,
                 padding,
                 dilation,
                 groups,
                 deform_groups,
                 bias=False,
                 im2col_step=32):
        return deform_conv_symbolic(g, input, weight, offset, None, None,
                                    stride, padding, dilation, groups,
                                    deform_groups)

    @staticmethod
    def forward(ctx,
                input: Tensor,
                offset: Tensor,
                weight: Tensor,
                stride: Union[int, Tuple[int, ...]] = 1,
                padding: Union[int, Tuple[int, ...]] = 0,
                dilation: Union[int, Tuple[int, ...]] = 1,
                groups: int = 1,
                deform_groups: int = 1,
                bias: bool = False,
                im2col_step: int = 32) -> Tensor:
        assert bias is False, 'Only support bias is False.'
        input, weight = _prepare(input, offset, weight, im2col_step)
        params = _op_args(weight, stride, padding, dilation, groups,
                          deform_groups, im2col_step)
        ext_loader.load()
        _save(ctx, input, offset, weight, params)
        return torch.ops.mmcv.deform_conv_forward(input, weight, offset,
                                                  *params)

    @staticmethod
    @once_differentiable
    def backward(
        ctx, grad_output: Tensor
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], None,
               None, None, None, None, None, None]:
        # Function order: input, offset, weight
        need = ctx.needs_input_grad
        grads = _grads(ctx, grad_output, need[0] or need[1], need[2])
        return grads + (None, ) * 7


def deform_conv2d(input: Tensor,
                  offset: Tensor,
                  weight: Tensor,
                  stride: Union[int, Tuple[int, ...]] = 1,
                  padding: Union[int, Tuple[int, ...]] = 0,
                  dilation: Union[int, Tuple[int, ...]] = 1,
                  groups: int = 1,
                  deform_groups: int = 1,
                  bias: bool = False,
                  im2col_step: int = 32) -> Tensor:
    if torch.onnx.is_in_onnx_export():
        if ONNX_EXPORT_MODE == 'deform_conv':
            return DeformConv2dFunction.apply(input, offset, weight, stride,
                                              padding, dilation, groups,
                                              deform_groups, bias, im2col_step)
        assert bias is False, 'Only support bias is False.'
        return deform_conv2d_pytorch(input, offset, weight, None, None, stride,
                                     padding, dilation, groups, deform_groups)
    assert bias is False, 'Only support bias is False.'
    input, weight = _prepare(input, offset, weight, im2col_step)
    ext_loader.load()
    return torch.ops.mmcv.deform_conv_forward(
        input, weight, offset,
        *_op_args(weight, stride, padding, dilation, groups, deform_groups,
                  im2col_step))


class DeformConv2d(nn.Module):
    r"""Deformable 2D convolution.

    Applies a deformable 2D convolution over an input signal composed of
    several input planes. DeformConv2d was described in the paper
    `Deformable Convolutional Networks
    <https://arxiv.org/pdf/1703.06211.pdf>`_

    Note:
        The argument ``im2col_step`` was added in version 1.3.17, which means
        number of samples processed by the ``im2col_cuda_kernel`` per call.
        It enables users to define ``batch_size`` and ``im2col_step`` more
        flexibly and solved `issue mmcv#1440
        <https://github.com/open-mmlab/mmcv/issues/1440>`_.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size(int, tuple): Size of the convolving kernel.
        stride(int, tuple): Stride of the convolution. Default: 1.
        padding (int or tuple): Zero-padding added to both sides of the input.
            Default: 0.
        dilation (int or tuple): Spacing between kernel elements. Default: 1.
        groups (int): Number of blocked connections from input.
            channels to output channels. Default: 1.
        deform_groups (int): Number of deformable group partitions.
        bias (bool): If True, adds a learnable bias to the output.
            Default: False.
        im2col_step (int): Number of samples processed by im2col_cuda_kernel
            per call. It will work when ``batch_size`` > ``im2col_step``, but
            ``batch_size`` must be divisible by ``im2col_step``. Default: 32.
            `New in version 1.3.17.`
    """

    @deprecated_api_warning({'deformable_groups': 'deform_groups'},
                            cls_name='DeformConv2d')
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 groups: int = 1,
                 deform_groups: int = 1,
                 bias: bool = False,
                 im2col_step: int = 32) -> None:
        super().__init__()

        assert not bias, \
            f'bias={bias} is not supported in DeformConv2d.'
        assert in_channels % groups == 0, \
            f'in_channels {in_channels} cannot be divisible by groups {groups}'
        assert out_channels % groups == 0, \
            f'out_channels {out_channels} cannot be divisible by groups \
              {groups}'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        self.im2col_step = im2col_step
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        # only weight, no bias
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups,
                         *self.kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        # switch the initialization of `self.weight` to the standard kaiming
        # method described in `Delving deep into rectifiers: Surpassing
        # human-level performance on ImageNet classification` - He, K. et al.
        # (2015), using a uniform distribution
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

    def forward(self, x: Tensor, offset: Tensor) -> Tensor:
        """Deformable Convolutional forward function.

        Args:
            x (Tensor): Input feature, shape (B, C_in, H_in, W_in)
            offset (Tensor): Offset for deformable convolution, shape
                (B, deform_groups*kernel_size[0]*kernel_size[1]*2,
                H_out, W_out), H_out, W_out are equal to the output's.

                An offset is like `[y0, x0, y1, x1, y2, x2, ..., y8, x8]`.
                The spatial arrangement is like:

                .. code:: text

                    (x0, y0) (x1, y1) (x2, y2)
                    (x3, y3) (x4, y4) (x5, y5)
                    (x6, y6) (x7, y7) (x8, y8)

        Returns:
            Tensor: Output of the layer.
        """
        # To fix an assert error in deform_conv_cuda.cpp:128
        # input image is smaller than kernel
        input_pad = (x.size(2) < self.kernel_size[0]) or (x.size(3) <
                                                          self.kernel_size[1])
        if input_pad:
            pad_h = max(self.kernel_size[0] - x.size(2), 0)
            pad_w = max(self.kernel_size[1] - x.size(3), 0)
            x = F.pad(x, (0, pad_w, 0, pad_h), 'constant', 0).contiguous()
            offset = F.pad(offset, (0, pad_w, 0, pad_h), 'constant', 0)
            offset = offset.contiguous()
        out = deform_conv2d(x, offset, self.weight, self.stride, self.padding,
                            self.dilation, self.groups, self.deform_groups,
                            False, self.im2col_step)
        if input_pad:
            out = out[:, :, :out.size(2) - pad_h, :out.size(3) -
                      pad_w].contiguous()
        return out

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(in_channels={self.in_channels},\n'
        s += f'out_channels={self.out_channels},\n'
        s += f'kernel_size={self.kernel_size},\n'
        s += f'stride={self.stride},\n'
        s += f'padding={self.padding},\n'
        s += f'dilation={self.dilation},\n'
        s += f'groups={self.groups},\n'
        s += f'deform_groups={self.deform_groups},\n'
        # bias is not supported in DeformConv2d.
        s += 'bias=False)'
        return s


@MODELS.register_module('DCN')
class DeformConv2dPack(DeformConv2d):
    """A Deformable Conv Encapsulation that acts as normal Conv layers.

    The offset tensor is like `[y0, x0, y1, x1, y2, x2, ..., y8, x8]`.
    The spatial arrangement is like:

    .. code:: text

        (x0, y0) (x1, y1) (x2, y2)
        (x3, y3) (x4, y4) (x5, y5)
        (x6, y6) (x7, y7) (x8, y8)

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
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
            self.deform_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            dilation=_pair(self.dilation),
            bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        offset = self.conv_offset(x)
        return deform_conv2d(x, offset, self.weight, self.stride, self.padding,
                             self.dilation, self.groups, self.deform_groups,
                             False, self.im2col_step)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if version is None or version < 2:
            # the key is different in early versions
            # In version < 2, DeformConvPack loads previous benchmark models.
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
                f'DeformConv2dPack {prefix.rstrip(".")} is upgraded to '
                'version 2.',
                logger='current')

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)
