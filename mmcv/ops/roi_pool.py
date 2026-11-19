# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Tuple, Union

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from ..utils import ext_loader


def _op_args(output_size, spatial_scale):
    """Arguments of ``mmcv::roi_pool_forward`` after ``input`` and ``rois``."""
    output_size = _pair(output_size)
    return output_size[0], output_size[1], float(spatial_scale)


def _save(ctx, input, rois, params, argmax):
    ctx.save_for_backward(rois, argmax)
    ctx.input_shape = list(input.shape)
    ctx.params = params


def _input_grad(ctx, grad_output):
    rois, argmax = ctx.saved_tensors
    return torch.ops.mmcv.roi_pool_backward(grad_output, rois, argmax,
                                            ctx.input_shape, *ctx.params)


def _setup_context(ctx, inputs, output):
    input, rois, *params = inputs
    ctx.mark_non_differentiable(output[1])
    _save(ctx, input, rois, params, output[1])


def _backward(ctx, grad_output, _):
    return (_input_grad(ctx, grad_output), ) + (None, ) * 4


@ext_loader.on_load
def _register():

    @torch.library.register_fake('mmcv::roi_pool_forward')
    def _(input, rois, pooled_height, pooled_width, spatial_scale):
        shape = (rois.size(0), input.size(1), pooled_height, pooled_width)
        return input.new_empty(shape), input.new_empty(shape, dtype=torch.int)

    @torch.library.register_fake('mmcv::roi_pool_backward')
    def _(grad_output, rois, argmax, input_shape, *_):
        return grad_output.new_empty(input_shape)

    torch.library.register_autograd(
        'mmcv::roi_pool_forward', _backward, setup_context=_setup_context)


class RoIPoolFunction(Function):
    """:func:`roi_pool` as an autograd function.

    Kept for the TorchScript ONNX exporter (``symbolic``) and for code that
    calls ``RoIPoolFunction.apply``.
    """

    @staticmethod
    def symbolic(g, input, rois, output_size, spatial_scale):
        return g.op(
            'MaxRoiPool',
            input,
            rois,
            pooled_shape_i=output_size,
            spatial_scale_f=spatial_scale)

    @staticmethod
    def forward(ctx: Any,
                input: torch.Tensor,
                rois: torch.Tensor,
                output_size: Union[int, tuple],
                spatial_scale: float = 1.0) -> torch.Tensor:
        params = _op_args(output_size, spatial_scale)
        ext_loader.load()
        output, argmax = torch.ops.mmcv.roi_pool_forward(input, rois, *params)
        _save(ctx, input, rois, params, argmax)
        return output

    @staticmethod
    @once_differentiable
    def backward(
            ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None, None]:
        return _input_grad(ctx, grad_output), None, None, None


def roi_pool(input: torch.Tensor,
             rois: torch.Tensor,
             output_size: Union[int, tuple],
             spatial_scale: float = 1.0) -> torch.Tensor:
    if torch.onnx.is_in_onnx_export():
        return RoIPoolFunction.apply(input, rois, output_size, spatial_scale)
    ext_loader.load()
    return torch.ops.mmcv.roi_pool_forward(
        input, rois, *_op_args(output_size, spatial_scale))[0]


class RoIPool(nn.Module):

    def __init__(self,
                 output_size: Union[int, tuple],
                 spatial_scale: float = 1.0):
        super().__init__()

        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)

    def forward(self, input: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        return roi_pool(input, rois, self.output_size, self.spatial_scale)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(output_size={self.output_size}, '
        s += f'spatial_scale={self.spatial_scale})'
        return s
