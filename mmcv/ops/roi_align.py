# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

import torch
import torch.nn as nn
from mmengine.utils import deprecated_api_warning
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from ..utils import ext_loader


def _op_args(output_size, spatial_scale, sampling_ratio, pool_mode, aligned):
    """Arguments of ``mmcv::roi_align_forward`` after input and rois."""
    assert pool_mode in ('max', 'avg')
    output_size = _pair(output_size)
    return (output_size[0], output_size[1], float(spatial_scale),
            int(sampling_ratio), 0 if pool_mode == 'max' else 1, bool(aligned))


def _save(ctx, input, rois, params, argmax_y, argmax_x):
    ctx.save_for_backward(rois, argmax_y, argmax_x)
    ctx.input_shape = list(input.shape)
    ctx.params = params


def _input_grad(ctx, grad_output):
    rois, argmax_y, argmax_x = ctx.saved_tensors
    return torch.ops.mmcv.roi_align_backward(grad_output, rois, argmax_y,
                                             argmax_x, ctx.input_shape,
                                             *ctx.params)


def _setup_context(ctx, inputs, output):
    input, rois, *params = inputs
    _, argmax_y, argmax_x = output
    ctx.mark_non_differentiable(argmax_y, argmax_x)
    _save(ctx, input, rois, params, argmax_y, argmax_x)


def _backward(ctx, grad_output, *_):
    return (_input_grad(ctx, grad_output), ) + (None, ) * 7


@ext_loader.on_load
def _register():

    @torch.library.register_fake('mmcv::roi_align_forward')
    def _(input, rois, aligned_height, aligned_width, spatial_scale,
          sampling_ratio, pool_mode, aligned):
        shape = (rois.size(0), input.size(1), aligned_height, aligned_width)
        argmax_shape = shape if pool_mode == 0 else (0, )
        return (input.new_empty(shape), input.new_empty(argmax_shape),
                input.new_empty(argmax_shape))

    @torch.library.register_fake('mmcv::roi_align_backward')
    def _(grad_output, rois, argmax_y, argmax_x, input_shape, *_):
        return grad_output.new_empty(input_shape)

    torch.library.register_autograd(
        'mmcv::roi_align_forward', _backward, setup_context=_setup_context)


class RoIAlignFunction(Function):
    """:func:`roi_align` as an autograd function.

    :func:`roi_align` calls the registered op ``mmcv::roi_align_forward``.
    This class remains for the TorchScript ONNX exporter, which needs
    ``symbolic``, and for code that calls ``RoIAlignFunction.apply``.
    """

    @staticmethod
    def symbolic(g, input, rois, output_size, spatial_scale, sampling_ratio,
                 pool_mode, aligned):
        from torch.onnx import TensorProtoDataType
        from torch.onnx.symbolic_opset9 import sub

        def _select(g, self, dim, index):
            return g.op('Gather', self, index, axis_i=dim)

        # batch_indices = rois[:, 0].long(); a scalar index drops the axis,
        # which Squeeze would do with an attribute up to opset 12 and an
        # input from opset 13
        batch_indices = _select(
            g, rois, 1,
            g.op('Constant', value_t=torch.tensor(0, dtype=torch.long)))
        batch_indices = g.op(
            'Cast', batch_indices, to_i=TensorProtoDataType.INT64)
        # rois = rois[:, 1:]
        rois = _select(
            g, rois, 1,
            g.op(
                'Constant',
                value_t=torch.tensor([1, 2, 3, 4], dtype=torch.long)))

        attrs = {}
        if g.opset >= 16:
            # RoiAlign-16 shifts the scaled rois by -0.5 itself unless told
            # otherwise (its default is half_pixel); output_half_pixel is the
            # behaviour of opset 10, which aligned=False matches
            attrs['coordinate_transformation_mode_s'] = (
                'half_pixel' if aligned else 'output_half_pixel')
        elif aligned:
            # rois -= 0.5/spatial_scale
            aligned_offset = g.op(
                'Constant',
                value_t=torch.tensor([0.5 / spatial_scale],
                                     dtype=torch.float32))
            rois = sub(g, rois, aligned_offset)
        # roi align
        return g.op(
            'RoiAlign',
            input,
            rois,
            batch_indices,
            output_height_i=output_size[0],
            output_width_i=output_size[1],
            spatial_scale_f=spatial_scale,
            sampling_ratio_i=max(0, sampling_ratio),
            mode_s=pool_mode,
            **attrs)

    @staticmethod
    def forward(ctx: Any,
                input: torch.Tensor,
                rois: torch.Tensor,
                output_size: int,
                spatial_scale: float = 1.0,
                sampling_ratio: int = 0,
                pool_mode: str = 'avg',
                aligned: bool = True) -> torch.Tensor:
        params = _op_args(output_size, spatial_scale, sampling_ratio,
                          pool_mode, aligned)
        ext_loader.load()
        output, argmax_y, argmax_x = torch.ops.mmcv.roi_align_forward(
            input, rois, *params)
        _save(ctx, input, rois, params, argmax_y, argmax_x)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        return (_input_grad(ctx, grad_output), ) + (None, ) * 6


def roi_align(input: torch.Tensor,
              rois: torch.Tensor,
              output_size: int,
              spatial_scale: float = 1.0,
              sampling_ratio: int = 0,
              pool_mode: str = 'avg',
              aligned: bool = True) -> torch.Tensor:
    if torch.onnx.is_in_onnx_export():
        return RoIAlignFunction.apply(input, rois, output_size, spatial_scale,
                                      sampling_ratio, pool_mode, aligned)
    params = _op_args(output_size, spatial_scale, sampling_ratio, pool_mode,
                      aligned)
    ext_loader.load()
    return torch.ops.mmcv.roi_align_forward(input, rois, *params)[0]


class RoIAlign(nn.Module):
    """RoI align pooling layer.

    Args:
        output_size (tuple): h, w
        spatial_scale (float): scale the input boxes by this number
        sampling_ratio (int): number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
        pool_mode (str, 'avg' or 'max'): pooling mode in each bin.
        aligned (bool): if False, use the legacy implementation in
            MMDetection. If True, align the results more perfectly.
        use_torchvision (bool): whether to use roi_align from torchvision.

    Note:
        The implementation of RoIAlign when aligned=True is modified from
        https://github.com/facebookresearch/detectron2/

        The meaning of aligned=True:

        Given a continuous coordinate c, its two neighboring pixel
        indices (in our pixel model) are computed by floor(c - 0.5) and
        ceil(c - 0.5). For example, c=1.3 has pixel neighbors with discrete
        indices [0] and [1] (which are sampled from the underlying signal
        at continuous coordinates 0.5 and 1.5). But the original roi_align
        (aligned=False) does not subtract the 0.5 when computing
        neighboring pixel indices and therefore it uses pixels with a
        slightly incorrect alignment (relative to our pixel model) when
        performing bilinear interpolation.

        With `aligned=True`,
        we first appropriately scale the ROI and then shift it by -0.5
        prior to calling roi_align. This produces the correct neighbors;

        The difference does not make a difference to the model's
        performance if ROIAlign is used together with conv layers.
    """

    @deprecated_api_warning(
        {
            'out_size': 'output_size',
            'sample_num': 'sampling_ratio'
        },
        cls_name='RoIAlign')
    def __init__(self,
                 output_size: tuple,
                 spatial_scale: float = 1.0,
                 sampling_ratio: int = 0,
                 pool_mode: str = 'avg',
                 aligned: bool = True,
                 use_torchvision: bool = False):
        super().__init__()

        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.pool_mode = pool_mode
        self.aligned = aligned
        self.use_torchvision = use_torchvision

    def forward(self, input: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N.\
                The other 4 columns are xyxy.
        """
        if self.use_torchvision:
            from torchvision.ops import roi_align as tv_roi_align
            if 'aligned' in tv_roi_align.__code__.co_varnames:
                return tv_roi_align(input, rois, self.output_size,
                                    self.spatial_scale, self.sampling_ratio,
                                    self.aligned)
            else:
                if self.aligned:
                    rois -= rois.new_tensor([0.] +
                                            [0.5 / self.spatial_scale] * 4)
                return tv_roi_align(input, rois, self.output_size,
                                    self.spatial_scale, self.sampling_ratio)
        else:
            return roi_align(input, rois, self.output_size, self.spatial_scale,
                             self.sampling_ratio, self.pool_mode, self.aligned)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(output_size={self.output_size}, '
        s += f'spatial_scale={self.spatial_scale}, '
        s += f'sampling_ratio={self.sampling_ratio}, '
        s += f'pool_mode={self.pool_mode}, '
        s += f'aligned={self.aligned}, '
        s += f'use_torchvision={self.use_torchvision})'
        return s
