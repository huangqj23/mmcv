# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmengine.registry import MODELS
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from ..utils import ext_loader


@ext_loader.on_load
def _register():
    # The kernels behind SyncBatchNormFunction. They have no autograd of
    # their own: the function all-reduces between them.

    @torch.library.register_fake('mmcv::sync_bn_forward_mean')
    def _(input):
        return input.new_empty((input.size(1), ), dtype=torch.float)

    @torch.library.register_fake('mmcv::sync_bn_forward_var')
    def _(input, mean):
        return input.new_empty((input.size(1), ), dtype=torch.float)

    @torch.library.register_fake('mmcv::sync_bn_forward_output')
    def _(input, mean, var, weight, bias, running_mean, running_var, eps,
          momentum, group_size):
        return (input.new_empty(input.shape),
                input.new_empty(input.shape, dtype=torch.float),
                input.new_empty((input.size(1), ), dtype=torch.float))

    @torch.library.register_fake('mmcv::sync_bn_backward_param')
    def _(grad_output, norm):
        channels = grad_output.size(1)
        return (norm.new_empty((channels, ), dtype=torch.float),
                norm.new_empty((channels, ), dtype=torch.float))

    @torch.library.register_fake('mmcv::sync_bn_backward_data')
    def _(grad_output, weight, grad_weight, grad_bias, norm, std):
        return grad_output.new_empty(grad_output.shape)


class SyncBatchNormFunction(Function):

    @staticmethod
    def symbolic(g, input, running_mean, running_var, weight, bias, momentum,
                 eps, group, group_size, stats_mode):
        return g.op(
            'mmcv::MMCVSyncBatchNorm',
            input,
            running_mean,
            running_var,
            weight,
            bias,
            momentum_f=momentum,
            eps_f=eps,
            group_i=group,
            group_size_i=group_size,
            stats_mode=stats_mode)

    @staticmethod
    def forward(self, input: torch.Tensor, running_mean: torch.Tensor,
                running_var: torch.Tensor, weight: torch.Tensor,
                bias: torch.Tensor, momentum: float, eps: float, group: int,
                group_size: int, stats_mode: str) -> torch.Tensor:
        self.momentum = momentum
        self.eps = eps
        self.group = group
        self.group_size = group_size
        self.stats_mode = stats_mode

        assert isinstance(
                   input, (torch.HalfTensor, torch.FloatTensor,
                           torch.cuda.HalfTensor, torch.cuda.FloatTensor)), \
               f'only support Half or Float Tensor, but {input.type()}'
        ext_loader.load()
        input3d = input.contiguous().flatten(start_dim=2)
        num_channels = input3d.size(1)

        # mean and var stay zeros when the input is empty
        mean = torch.zeros(
            num_channels, dtype=torch.float, device=input3d.device)
        var = torch.zeros(
            num_channels, dtype=torch.float, device=input3d.device)

        batch_size = input3d.size(0)
        if batch_size > 0:
            mean = torch.ops.mmcv.sync_bn_forward_mean(input3d)
            batch_flag = torch.ones([1], device=mean.device, dtype=mean.dtype)
        else:
            batch_flag = torch.zeros([1], device=mean.device, dtype=mean.dtype)

        # synchronize mean and the batch flag
        vec = torch.cat([mean, batch_flag])
        if self.stats_mode == 'N':
            vec *= batch_size
        if self.group_size > 1:
            dist.all_reduce(vec, group=self.group)
        total_batch = vec[-1].detach()
        mean = vec[:num_channels]

        if self.stats_mode == 'default':
            mean = mean / self.group_size
        elif self.stats_mode == 'N':
            mean = mean / total_batch.clamp(min=1)
        else:
            raise NotImplementedError

        if batch_size > 0:
            var = torch.ops.mmcv.sync_bn_forward_var(input3d, mean)

        if self.stats_mode == 'N':
            var *= batch_size
        if self.group_size > 1:
            dist.all_reduce(var, group=self.group)

        if self.stats_mode == 'default':
            var /= self.group_size
        elif self.stats_mode == 'N':
            var /= total_batch.clamp(min=1)
        else:
            raise NotImplementedError

        # if the total batch size over all the ranks is zero,
        # we should not update the statistics in the current batch
        update_flag = total_batch.clamp(max=1)
        momentum = float(update_flag) * self.momentum
        # updates running_mean and running_var in place
        output3d, norm, std = torch.ops.mmcv.sync_bn_forward_output(
            input3d, mean, var, weight, bias, running_mean, running_var,
            float(self.eps), momentum, self.group_size)
        self.save_for_backward(norm, std, weight)
        return output3d.view_as(input)

    @staticmethod
    @once_differentiable
    def backward(self, grad_output: torch.Tensor) -> tuple:
        norm, std, weight = self.saved_tensors
        grad_output3d = grad_output.contiguous().flatten(start_dim=2)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(weight)

        batch_size = grad_output3d.size(0)
        if batch_size > 0:
            grad_weight, grad_bias = torch.ops.mmcv.sync_bn_backward_param(
                grad_output3d, norm)

        # all reduce
        if self.group_size > 1:
            dist.all_reduce(grad_weight, group=self.group)
            dist.all_reduce(grad_bias, group=self.group)
            grad_weight /= self.group_size
            grad_bias /= self.group_size

        if batch_size > 0:
            grad_input = torch.ops.mmcv.sync_bn_backward_data(
                grad_output3d, weight, grad_weight, grad_bias, norm,
                std).view_as(grad_output)
        else:
            grad_input = torch.zeros_like(grad_output)

        return grad_input, None, None, grad_weight, grad_bias, \
            None, None, None, None, None


@MODELS.register_module(name='MMSyncBN')
class SyncBatchNorm(Module):
    """Synchronized Batch Normalization.

    Args:
        num_features (int): number of features/chennels in input tensor
        eps (float, optional): a value added to the denominator for numerical
            stability. Defaults to 1e-5.
        momentum (float, optional): the value used for the running_mean and
            running_var computation. Defaults to 0.1.
        affine (bool, optional): whether to use learnable affine parameters.
            Defaults to True.
        track_running_stats (bool, optional): whether to track the running
            mean and variance during training. When set to False, this
            module does not track such statistics, and initializes statistics
            buffers ``running_mean`` and ``running_var`` as ``None``. When
            these buffers are ``None``, this module always uses batch
            statistics in both training and eval modes. Defaults to True.
        group (int, optional): synchronization of stats happen within
            each process group individually. By default it is synchronization
            across the whole world. Defaults to None.
        stats_mode (str, optional): The statistical mode. Available options
            includes ``'default'`` and ``'N'``. Defaults to 'default'.
            When ``stats_mode=='default'``, it computes the overall statistics
            using those from each worker with equal weight, i.e., the
            statistics are synchronized and simply divied by ``group``. This
            mode will produce inaccurate statistics when empty tensors occur.
            When ``stats_mode=='N'``, it compute the overall statistics using
            the total number of batches in each worker ignoring the number of
            group, i.e., the statistics are synchronized and then divied by
            the total batch ``N``. This mode is beneficial when empty tensors
            occur during training, as it average the total mean by the real
            number of batch.
    """

    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 group: Optional[int] = None,
                 stats_mode: str = 'default'):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        group = dist.group.WORLD if group is None else group
        self.group = group
        self.group_size = dist.get_world_size(group)
        assert stats_mode in ['default', 'N'], \
            f'"stats_mode" only accepts "default" and "N", got "{stats_mode}"'
        self.stats_mode = stats_mode
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()  # pytorch use ones_()
            self.bias.data.zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() < 2:
            raise ValueError(
                f'expected at least 2D input, got {input.dim()}D input')
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or not self.track_running_stats:
            return SyncBatchNormFunction.apply(
                input, self.running_mean, self.running_var, self.weight,
                self.bias, exponential_average_factor, self.eps, self.group,
                self.group_size, self.stats_mode)
        else:
            return F.batch_norm(input, self.running_mean, self.running_var,
                                self.weight, self.bias, False,
                                exponential_average_factor, self.eps)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'({self.num_features}, '
        s += f'eps={self.eps}, '
        s += f'momentum={self.momentum}, '
        s += f'affine={self.affine}, '
        s += f'track_running_stats={self.track_running_stats}, '
        s += f'group_size={self.group_size},'
        s += f'stats_mode={self.stats_mode})'
        return s
