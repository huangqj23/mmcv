# Modified from https://github.com/hszhao/semseg/blob/master/lib/psa
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from ..utils import ext_loader

EXT_BUILT = ext_loader.check_ops_exist()


def _rel_to_abs(x: torch.Tensor, size: int, half: int) -> torch.Tensor:
    """Relative to absolute positions along the last axis.

    ``x[..., i, a]`` holds offset ``a - half`` from position ``i``; returns
    ``out[..., i, j] = x[..., i, j - i + half]``, 0 where that offset is not
    in ``x``. Only pad / slice / reshape, so it exports to plain ONNX ops
    without index tensors.
    """
    if size == 1:
        return x[..., half:half + 1] if 0 <= half < x.shape[-1] else \
            x.new_zeros(x.shape[:-1] + (1, ))
    # place offsets on the full range [-(size - 1), size - 1]
    left = size - 1 - half
    if left < 0:  # offsets beyond the map
        x = x[..., -left:]
        left = 0
    right = 2 * size - 1 - left - x.shape[-1]
    if right < 0:
        x = x[..., :right]
        right = 0
    # contiguous input: eager pad returns a contiguous tensor anyway, but
    # under torch.compile the padded copy of a permuted tensor can keep the
    # permuted layout, and the flatten below would become an invalid view
    x = F.pad(x.contiguous(), (left, right))
    # x[i, j - i + size - 1] sits at flat index size - 1 + i * (2 size - 2) + j
    flat = x.flatten(-2)[..., size - 1:size - 1 + size * (2 * size - 2)]
    return flat.unflatten(-1, (size, 2 * size - 2))[..., :size]


def psa_mask_pytorch(input: torch.Tensor, psa_type: int,
                     mask_size) -> torch.Tensor:
    """PSAMask in plain PyTorch: same output as the CPU / CUDA kernels.

    Used for ONNX export (TensorRT and OpenVINO get standard ops instead of
    ``mmcv::MMCVPSAMask``), under ``torch.compile`` and without the compiled
    extension.
    """
    n, c, h, w = input.shape
    h_mask, w_mask = _pair(mask_size)
    assert c == h_mask * w_mask
    x = input.view(n, h_mask, w_mask, h, w)
    # rows: (n, w_mask, w, h, h_mask) -> (n, w_mask, w, h, h')
    x = _rel_to_abs(x.permute(0, 2, 4, 3, 1), h, (h_mask - 1) // 2)
    # columns: (n, h, h', w, w_mask) -> (n, h, h', w, w')
    x = _rel_to_abs(x.permute(0, 3, 4, 2, 1), w, (w_mask - 1) // 2)
    if psa_type == 0:  # collect: out[n, h' * w + w', h, w]
        x = x.permute(0, 2, 4, 1, 3)
    else:  # distribute: out[n, h * w + w, h', w']
        x = x.permute(0, 1, 3, 2, 4)
    return x.reshape(n, h * w, h, w)


def _setup_context(ctx, inputs, output):
    _, *params = inputs  # psa_type, h_mask, w_mask
    ctx.params = params


def _backward(ctx, grad_output):
    return (torch.ops.mmcv.psamask_backward(grad_output,
                                            *ctx.params), None, None, None)


@ext_loader.on_load
def _register():

    @torch.library.register_fake('mmcv::psamask_forward')
    def _(input, psa_type, h_mask, w_mask):
        n, _, h, w = input.shape
        return input.new_empty((n, h * w, h, w))

    @torch.library.register_fake('mmcv::psamask_backward')
    def _(grad_output, psa_type, h_mask, w_mask):
        n, _, h, w = grad_output.shape
        return grad_output.new_empty((n, h_mask * w_mask, h, w))

    torch.library.register_autograd(
        'mmcv::psamask_forward', _backward, setup_context=_setup_context)


class PSAMaskFunction(Function):
    """Kept for code that calls ``PSAMaskFunction.apply``; :func:`psa_mask`
    uses the registered op ``mmcv::psamask_forward``."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, psa_type: str,
                mask_size: int) -> torch.Tensor:
        ext_loader.load()
        ctx.params = (int(psa_type), *_pair(mask_size))
        return torch.ops.mmcv.psamask_forward(input, *ctx.params)

    @staticmethod
    def backward(
            ctx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None, None]:
        return _backward(ctx, grad_output)


def psa_mask(input: torch.Tensor, psa_type: int, mask_size) -> torch.Tensor:
    if (not EXT_BUILT or torch.onnx.is_in_onnx_export()
            or torch.compiler.is_compiling()):
        return psa_mask_pytorch(input, psa_type, mask_size)
    ext_loader.load()
    return torch.ops.mmcv.psamask_forward(input, int(psa_type),
                                          *_pair(mask_size))


class PSAMask(nn.Module):

    def __init__(self, psa_type: str, mask_size: Optional[tuple] = None):
        super().__init__()
        assert psa_type in ['collect', 'distribute']
        if psa_type == 'collect':
            psa_type_enum = 0
        else:
            psa_type_enum = 1
        self.psa_type_enum = psa_type_enum
        self.mask_size = mask_size
        self.psa_type = psa_type

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return psa_mask(input, self.psa_type_enum, self.mask_size)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(psa_type={self.psa_type}, '
        s += f'mask_size={self.mask_size})'
        return s
