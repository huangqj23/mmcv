# Modified from https://github.com/hszhao/semseg/blob/master/lib/psa
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext',
                                 ['psamask_forward', 'psamask_backward'])
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


class PSAMaskFunction(Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor, psa_type: str,
                mask_size: int) -> torch.Tensor:
        ctx.psa_type = psa_type
        ctx.mask_size = _pair(mask_size)
        ctx.save_for_backward(input)

        h_mask, w_mask = ctx.mask_size
        batch_size, channels, h_feature, w_feature = input.size()
        assert channels == h_mask * w_mask
        output = input.new_zeros(
            (batch_size, h_feature * w_feature, h_feature, w_feature))

        ext_module.psamask_forward(
            input,
            output,
            psa_type=psa_type,
            num_=batch_size,
            h_feature=h_feature,
            w_feature=w_feature,
            h_mask=h_mask,
            w_mask=w_mask,
            half_h_mask=(h_mask - 1) // 2,
            half_w_mask=(w_mask - 1) // 2)
        return output

    @staticmethod
    def backward(
            ctx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None, None]:
        input = ctx.saved_tensors[0]
        psa_type = ctx.psa_type
        h_mask, w_mask = ctx.mask_size
        batch_size, channels, h_feature, w_feature = input.size()
        grad_input = grad_output.new_zeros(
            (batch_size, channels, h_feature, w_feature))
        ext_module.psamask_backward(
            grad_output,
            grad_input,
            psa_type=psa_type,
            num_=batch_size,
            h_feature=h_feature,
            w_feature=w_feature,
            h_mask=h_mask,
            w_mask=w_mask,
            half_h_mask=(h_mask - 1) // 2,
            half_w_mask=(w_mask - 1) // 2)
        return grad_input, None, None, None


def psa_mask(input: torch.Tensor, psa_type: int, mask_size) -> torch.Tensor:
    if (not EXT_BUILT or torch.onnx.is_in_onnx_export()
            or torch.compiler.is_compiling()):
        return psa_mask_pytorch(input, psa_type, mask_size)
    return PSAMaskFunction.apply(input, psa_type, mask_size)


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
