# Copyright (c) OpenMMLab. All rights reserved.
"""The kernels read raw memory as dense tensors, so the Python side has to hand
them contiguous tensors. Without that, channels-last feature maps (common with
AMP) gave silently wrong outputs, and a stride-0 ``grad_output`` (what
``out.sum().backward()`` produces) gave wrong gradients."""
import pytest
import torch

from mmcv.utils import IS_CUDA_AVAILABLE

# an unusual value, so stale memory left by an earlier dense gradient cannot
# make the expanded case pass by accident
GRAD_VALUE = 1.7


def strided(t):
    """Same values, non-contiguous layout (channels-last for 4-D maps)."""
    if t.dim() == 4:
        return t.contiguous(memory_format=torch.channels_last)
    return t.transpose(-1, -2).contiguous().transpose(-1, -2)


def check(fn, *inputs):
    with torch.no_grad():
        ref = fn(*inputs)
        out = fn(*[strided(t) if t.is_floating_point() else t for t in inputs])
    assert torch.allclose(ref, out, rtol=1e-5, atol=1e-5)

    inputs = [
        t.detach().clone().requires_grad_(t.is_floating_point())
        for t in inputs
    ]
    out = fn(*inputs)
    floats = [t for t in inputs if t.requires_grad]
    expanded = torch.full((), GRAD_VALUE).to(out).expand_as(out)
    # allow_unused: no gradient flows to e.g. the rois of RoIAlign
    grad_expanded = torch.autograd.grad(
        out, floats, expanded, retain_graph=True, allow_unused=True)
    grad_dense = torch.autograd.grad(
        out, floats, torch.full_like(out, GRAD_VALUE), allow_unused=True)
    assert any(g is not None for g in grad_dense)
    for a, b in zip(grad_expanded, grad_dense):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.allclose(a, b, rtol=1e-5, atol=1e-5)


def psa_mask(device):
    from mmcv.ops.psa_mask import PSAMaskFunction
    x = torch.randn(2, 15 * 31, 8, 16, device=device)
    return lambda t: PSAMaskFunction.apply(t, 0, (15, 31)), x


def roi_align(device):
    from mmcv.ops import RoIAlign
    feat = torch.randn(2, 8, 16, 16, device=device)
    rois = torch.tensor([[0, 1., 1., 9., 9.], [1, 2., 3., 12., 14.]],
                        device=device)
    return lambda f, r: RoIAlign((4, 4), 1.0, 2)(f, r), feat, rois


def roi_pool(device):
    from mmcv.ops import RoIPool
    feat = torch.randn(2, 8, 16, 16, device=device)
    rois = torch.tensor([[0, 1., 1., 9., 9.], [1, 2., 3., 12., 14.]],
                        device=device)
    return lambda f, r: RoIPool((4, 4), 1.0)(f, r), feat, rois


def deform_roi_pool(device):
    from mmcv.ops import DeformRoIPool
    feat = torch.randn(2, 8, 16, 16, device=device)
    rois = torch.tensor([[0, 1., 1., 9., 9.], [1, 2., 3., 12., 14.]],
                        device=device)
    offset = torch.randn(2, 2, 4, 4, device=device) * 0.1
    return (lambda f, o: DeformRoIPool((4, 4), 1.0, 2)(f, rois, o), feat,
            offset)


def carafe(device):
    from mmcv.ops import CARAFE
    feat = torch.randn(2, 8, 16, 16, device=device)
    masks = torch.randn(2, 25, 32, 32, device=device).softmax(1)
    return lambda f, m: CARAFE(5, 1, 2)(f, m), feat, masks


def modulated_deform_conv(device):
    from mmcv.ops import ModulatedDeformConv2d
    torch.manual_seed(0)
    conv = ModulatedDeformConv2d(4, 6, 3, padding=1).to(device)
    x = torch.randn(2, 4, 10, 10, device=device)
    offset = torch.randn(2, 18, 10, 10, device=device) * 0.5
    mask = torch.rand(2, 9, 10, 10, device=device)
    return lambda a, o, m: conv(a, o, m), x, offset, mask


def deform_conv(device):
    from mmcv.ops import DeformConv2d
    torch.manual_seed(0)
    conv = DeformConv2d(4, 6, 3, padding=1).to(device)
    x = torch.randn(2, 4, 10, 10, device=device)
    offset = torch.randn(2, 18, 10, 10, device=device) * 0.5
    return lambda a, o: conv(a, o), x, offset


def sigmoid_focal_loss(device):
    from mmcv.ops import sigmoid_focal_loss
    logits = torch.randn(64, 5, device=device)
    labels = torch.randint(0, 5, (64, ), device=device)
    return (lambda p: sigmoid_focal_loss(p, labels, 2.0, 0.25, None, 'none'),
            logits)


def ms_deform_attn(device):
    from mmcv.ops.multi_scale_deform_attn import \
        MultiScaleDeformableAttnFunction
    shapes = torch.tensor([[8, 8], [4, 4]], device=device)
    start = torch.tensor([0, 64], device=device)
    value = torch.randn(1, 80, 2, 8, device=device)
    loc = torch.rand(1, 10, 2, 2, 4, 2, device=device)
    attn = torch.rand(1, 10, 2, 2, 4, device=device).softmax(-1)
    return (lambda v, l, a: MultiScaleDeformableAttnFunction.apply(
        v, shapes, start, l, a, 64), value, loc, attn)


CUDA_CASES = [
    psa_mask, roi_align, roi_pool, deform_roi_pool, carafe,
    modulated_deform_conv, deform_conv, sigmoid_focal_loss, ms_deform_attn
]
CPU_CASES = [psa_mask, roi_align, modulated_deform_conv, deform_conv]


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason='requires CUDA support')
@pytest.mark.parametrize('case', CUDA_CASES, ids=lambda c: c.__name__)
def test_noncontiguous_cuda(case):
    check(*case('cuda'))


@pytest.mark.parametrize('case', CPU_CASES, ids=lambda c: c.__name__)
def test_noncontiguous_cpu(case):
    check(*case('cpu'))
