# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from ..utils import ext_loader


def _op_args(output_size, spatial_scale, sampling_ratio, gamma):
    """Arguments of ``mmcv::deform_roi_pool_forward`` after the tensors."""
    output_size = _pair(output_size)
    return (output_size[0], output_size[1], float(spatial_scale),
            int(sampling_ratio), float(gamma))


def _save(ctx, input, rois, offset, params):
    ctx.save_for_backward(input, rois, offset)
    ctx.params = params


def _grads(ctx, grad_output):
    """Gradients of input and offset (None without an offset)."""
    input, rois, offset = ctx.saved_tensors
    grad_input, grad_offset = torch.ops.mmcv.deform_roi_pool_backward(
        grad_output, input, rois, offset, *ctx.params)
    return grad_input, grad_offset if offset is not None else None


def _setup_context(ctx, inputs, output):
    input, rois, offset, *params = inputs
    _save(ctx, input, rois, offset, params)


def _backward(ctx, grad_output):
    grad_input, grad_offset = _grads(ctx, grad_output)
    return (grad_input, None, grad_offset) + (None, ) * 5


@ext_loader.on_load
def _register():

    @torch.library.register_fake('mmcv::deform_roi_pool_forward')
    def _(input, rois, offset, pooled_height, pooled_width, *_):
        return input.new_empty(
            (rois.size(0), input.size(1), pooled_height, pooled_width))

    @torch.library.register_fake('mmcv::deform_roi_pool_backward')
    def _(grad_output, input, rois, offset, *_):
        grad_offset = (
            offset.new_empty(offset.shape)
            if offset is not None else input.new_empty((0, )))
        return input.new_empty(input.shape), grad_offset

    torch.library.register_autograd(
        'mmcv::deform_roi_pool_forward',
        _backward,
        setup_context=_setup_context)


class DeformRoIPoolFunction(Function):
    """:func:`deform_roi_pool` as an autograd function, kept for the
    TorchScript ONNX exporter (``symbolic``) and for
    ``DeformRoIPoolFunction.apply``."""

    @staticmethod
    def symbolic(g, input, rois, offset, output_size, spatial_scale,
                 sampling_ratio, gamma):
        inputs = [input, rois]
        if offset is not None:
            inputs = [input, rois, offset]
        return g.op(
            'mmcv::MMCVDeformRoIPool',
            *inputs,
            pooled_height_i=output_size[0],
            pooled_width_i=output_size[1],
            spatial_scale_f=spatial_scale,
            sampling_ratio_f=sampling_ratio,
            gamma_f=gamma,
        )

    @staticmethod
    def forward(ctx,
                input: Tensor,
                rois: Tensor,
                offset: Optional[Tensor],
                output_size: Tuple[int, ...],
                spatial_scale: float = 1.0,
                sampling_ratio: int = 0,
                gamma: float = 0.1) -> Tensor:
        params = _op_args(output_size, spatial_scale, sampling_ratio, gamma)
        ext_loader.load()
        _save(ctx, input, rois, offset, params)
        return torch.ops.mmcv.deform_roi_pool_forward(input, rois, offset,
                                                      *params)

    @staticmethod
    @once_differentiable
    def backward(
        ctx, grad_output: Tensor
    ) -> Tuple[Tensor, None, Tensor, None, None, None, None]:
        grad_input, grad_offset = _grads(ctx, grad_output)
        return grad_input, None, grad_offset, None, None, None, None


def deform_roi_pool(input: Tensor,
                    rois: Tensor,
                    offset: Optional[Tensor],
                    output_size: Tuple[int, ...],
                    spatial_scale: float = 1.0,
                    sampling_ratio: int = 0,
                    gamma: float = 0.1) -> Tensor:
    if torch.onnx.is_in_onnx_export():
        return DeformRoIPoolFunction.apply(input, rois, offset, output_size,
                                           spatial_scale, sampling_ratio,
                                           gamma)
    ext_loader.load()
    return torch.ops.mmcv.deform_roi_pool_forward(
        input, rois, offset,
        *_op_args(output_size, spatial_scale, sampling_ratio, gamma))


class DeformRoIPool(nn.Module):

    def __init__(self,
                 output_size: Tuple[int, ...],
                 spatial_scale: float = 1.0,
                 sampling_ratio: int = 0,
                 gamma: float = 0.1):
        super().__init__()
        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.gamma = float(gamma)

    def forward(self,
                input: Tensor,
                rois: Tensor,
                offset: Optional[Tensor] = None) -> Tensor:
        return deform_roi_pool(input, rois, offset, self.output_size,
                               self.spatial_scale, self.sampling_ratio,
                               self.gamma)


class DeformRoIPoolPack(DeformRoIPool):

    def __init__(self,
                 output_size: Tuple[int, ...],
                 output_channels: int,
                 deform_fc_channels: int = 1024,
                 spatial_scale: float = 1.0,
                 sampling_ratio: int = 0,
                 gamma: float = 0.1):
        super().__init__(output_size, spatial_scale, sampling_ratio, gamma)

        self.output_channels = output_channels
        self.deform_fc_channels = deform_fc_channels

        self.offset_fc = nn.Sequential(
            nn.Linear(
                self.output_size[0] * self.output_size[1] *
                self.output_channels, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels,
                      self.output_size[0] * self.output_size[1] * 2))
        self.offset_fc[-1].weight.data.zero_()
        self.offset_fc[-1].bias.data.zero_()

    def forward(self, input: Tensor, rois: Tensor) -> Tensor:  # type: ignore
        assert input.size(1) == self.output_channels
        x = deform_roi_pool(input, rois, None, self.output_size,
                            self.spatial_scale, self.sampling_ratio,
                            self.gamma)
        rois_num = rois.size(0)
        offset = self.offset_fc(x.view(rois_num, -1))
        offset = offset.view(rois_num, 2, self.output_size[0],
                             self.output_size[1])
        return deform_roi_pool(input, rois, offset, self.output_size,
                               self.spatial_scale, self.sampling_ratio,
                               self.gamma)


class ModulatedDeformRoIPoolPack(DeformRoIPool):

    def __init__(self,
                 output_size: Tuple[int, ...],
                 output_channels: int,
                 deform_fc_channels: int = 1024,
                 spatial_scale: float = 1.0,
                 sampling_ratio: int = 0,
                 gamma: float = 0.1):
        super().__init__(output_size, spatial_scale, sampling_ratio, gamma)

        self.output_channels = output_channels
        self.deform_fc_channels = deform_fc_channels

        self.offset_fc = nn.Sequential(
            nn.Linear(
                self.output_size[0] * self.output_size[1] *
                self.output_channels, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels,
                      self.output_size[0] * self.output_size[1] * 2))
        self.offset_fc[-1].weight.data.zero_()
        self.offset_fc[-1].bias.data.zero_()

        self.mask_fc = nn.Sequential(
            nn.Linear(
                self.output_size[0] * self.output_size[1] *
                self.output_channels, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels,
                      self.output_size[0] * self.output_size[1] * 1),
            nn.Sigmoid())
        self.mask_fc[2].weight.data.zero_()
        self.mask_fc[2].bias.data.zero_()

    def forward(self, input: Tensor, rois: Tensor) -> Tensor:  # type: ignore
        assert input.size(1) == self.output_channels
        x = deform_roi_pool(input, rois, None, self.output_size,
                            self.spatial_scale, self.sampling_ratio,
                            self.gamma)
        rois_num = rois.size(0)
        offset = self.offset_fc(x.view(rois_num, -1))
        offset = offset.view(rois_num, 2, self.output_size[0],
                             self.output_size[1])
        mask = self.mask_fc(x.view(rois_num, -1))
        mask = mask.view(rois_num, 1, self.output_size[0], self.output_size[1])
        d = deform_roi_pool(input, rois, offset, self.output_size,
                            self.spatial_scale, self.sampling_ratio,
                            self.gamma)
        return d * mask
