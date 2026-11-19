# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ..utils import ext_loader


def _check(input, target, weight, reduction):
    assert target.dtype == torch.long
    assert input.dim() == 2
    assert target.dim() == 1
    assert input.size(0) == target.size(0)
    if weight is not None:
        assert weight.dim() == 1
        assert input.size(1) == weight.size(0)
    assert reduction in ('none', 'mean', 'sum')


def _reduce(loss, reduction, num):
    if reduction == 'mean':
        return loss.sum() / num
    if reduction == 'sum':
        return loss.sum()
    return loss


# The registered ops compute the loss of each element (sigmoid) or sample
# (softmax); the reduction is ordinary PyTorch code, so autograd handles it
# for every reduction, 'none' included.


def _sigmoid_setup_context(ctx, inputs, output):
    input, target, weight, gamma, alpha = inputs
    ctx.save_for_backward(input, target, weight)
    ctx.params = (gamma, alpha)


def _sigmoid_backward(ctx, grad_output):
    input, target, weight = ctx.saved_tensors
    grad_input = torch.ops.mmcv.sigmoid_focal_loss_backward(
        input, target, weight, *ctx.params)
    return grad_input * grad_output, None, None, None, None


def _softmax_setup_context(ctx, inputs, output):
    input, target, weight, gamma, alpha = inputs
    softmax = output[1]
    ctx.mark_non_differentiable(softmax)
    ctx.save_for_backward(softmax, target, weight)
    ctx.params = (gamma, alpha)


def _softmax_backward(ctx, grad_output, _):
    softmax, target, weight = ctx.saved_tensors
    grad_input = torch.ops.mmcv.softmax_focal_loss_backward(
        softmax, target, weight, *ctx.params)
    return grad_input * grad_output[:, None], None, None, None, None


@ext_loader.on_load
def _register():

    @torch.library.register_fake('mmcv::sigmoid_focal_loss_forward')
    def _(input, target, weight, gamma, alpha):
        return input.new_empty(input.shape)

    @torch.library.register_fake('mmcv::sigmoid_focal_loss_backward')
    def _(input, target, weight, gamma, alpha):
        return input.new_empty(input.shape)

    @torch.library.register_fake('mmcv::softmax_focal_loss_forward')
    def _(input, target, weight, gamma, alpha):
        return input.new_empty((input.size(0), )), input.new_empty(input.shape)

    @torch.library.register_fake('mmcv::softmax_focal_loss_backward')
    def _(softmax, target, weight, gamma, alpha):
        return softmax.new_empty(softmax.shape)

    torch.library.register_autograd(
        'mmcv::sigmoid_focal_loss_forward',
        _sigmoid_backward,
        setup_context=_sigmoid_setup_context)
    torch.library.register_autograd(
        'mmcv::softmax_focal_loss_forward',
        _softmax_backward,
        setup_context=_softmax_setup_context)


class SigmoidFocalLossFunction(Function):
    """Kept for code that calls ``SigmoidFocalLossFunction.apply``;
    :func:`sigmoid_focal_loss` uses the registered op."""

    @staticmethod
    def forward(ctx,
                input: torch.Tensor,
                target: Union[torch.LongTensor, torch.cuda.LongTensor],
                gamma: float = 2.0,
                alpha: float = 0.25,
                weight: Optional[torch.Tensor] = None,
                reduction: str = 'mean') -> torch.Tensor:
        _check(input, target, weight, reduction)
        ext_loader.load()
        _sigmoid_setup_context(
            ctx, (input, target, weight, float(gamma), float(alpha)), None)
        ctx.reduction = reduction
        loss = torch.ops.mmcv.sigmoid_focal_loss_forward(
            input, target, weight, *ctx.params)
        return _reduce(loss, reduction, input.size(0))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        grad_input = _sigmoid_backward(ctx, grad_output)[0]
        if ctx.reduction == 'mean':
            grad_input /= grad_input.size(0)
        return grad_input, None, None, None, None, None


def sigmoid_focal_loss(input: torch.Tensor,
                       target: Union[torch.LongTensor, torch.cuda.LongTensor],
                       gamma: float = 2.0,
                       alpha: float = 0.25,
                       weight: Optional[torch.Tensor] = None,
                       reduction: str = 'mean') -> torch.Tensor:
    _check(input, target, weight, reduction)
    ext_loader.load()
    loss = torch.ops.mmcv.sigmoid_focal_loss_forward(input, target, weight,
                                                     float(gamma),
                                                     float(alpha))
    return _reduce(loss, reduction, input.size(0))


class SigmoidFocalLoss(nn.Module):

    def __init__(self,
                 gamma: float,
                 alpha: float,
                 weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer('weight', weight)
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: Union[torch.LongTensor, torch.cuda.LongTensor],
    ) -> torch.Tensor:
        return sigmoid_focal_loss(input, target, self.gamma, self.alpha,
                                  self.weight, self.reduction)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(gamma={self.gamma}, '
        s += f'alpha={self.alpha}, '
        s += f'reduction={self.reduction})'
        return s


class SoftmaxFocalLossFunction(Function):
    """Kept for code that calls ``SoftmaxFocalLossFunction.apply``;
    :func:`softmax_focal_loss` uses the registered op."""

    @staticmethod
    def forward(ctx,
                input: torch.Tensor,
                target: Union[torch.LongTensor, torch.cuda.LongTensor],
                gamma: float = 2.0,
                alpha: float = 0.25,
                weight: Optional[torch.Tensor] = None,
                reduction='mean') -> torch.Tensor:
        _check(input, target, weight, reduction)
        ext_loader.load()
        params = (float(gamma), float(alpha))
        output = torch.ops.mmcv.softmax_focal_loss_forward(
            input, target, weight, *params)
        ctx.save_for_backward(output[1], target, weight)
        ctx.params = params
        ctx.reduction = reduction
        return _reduce(output[0], reduction, input.size(0))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        if grad_output.dim() == 0:  # 'mean' or 'sum'
            grad_output = grad_output.expand(ctx.saved_tensors[0].size(0))
        grad_input = _softmax_backward(ctx, grad_output, None)[0]
        if ctx.reduction == 'mean':
            grad_input /= grad_input.size(0)
        return grad_input, None, None, None, None, None


def softmax_focal_loss(input: torch.Tensor,
                       target: Union[torch.LongTensor, torch.cuda.LongTensor],
                       gamma: float = 2.0,
                       alpha: float = 0.25,
                       weight: Optional[torch.Tensor] = None,
                       reduction='mean') -> torch.Tensor:
    _check(input, target, weight, reduction)
    ext_loader.load()
    loss = torch.ops.mmcv.softmax_focal_loss_forward(input, target, weight,
                                                     float(gamma),
                                                     float(alpha))[0]
    return _reduce(loss, reduction, input.size(0))


class SoftmaxFocalLoss(nn.Module):

    def __init__(self,
                 gamma: float,
                 alpha: float,
                 weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer('weight', weight)
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: Union[torch.LongTensor, torch.cuda.LongTensor],
    ) -> torch.Tensor:
        return softmax_focal_loss(input, target, self.gamma, self.alpha,
                                  self.weight, self.reduction)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(gamma={self.gamma}, '
        s += f'alpha={self.alpha}, '
        s += f'reduction={self.reduction})'
        return s
