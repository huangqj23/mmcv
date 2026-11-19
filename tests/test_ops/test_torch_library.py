# Copyright (c) OpenMMLab. All rights reserved.
"""The compiled ops as registered with the PyTorch dispatcher.

* ``torch.library.opcheck``: the schema, the fake (shape) functions, the
  autograd registration and AOT dispatch agree with the CPU / CUDA kernels.
* ``gradcheck`` of the registered backward functions, in float64.
* ``torch.compile(fullgraph=True)`` gives the eager results without breaking
  the graph.
* ``import mmcv.ops`` leaves the compiled library unloaded.
"""
import subprocess
import sys

import pytest
import torch
from torch.autograd import gradcheck

from mmcv.utils import ext_loader

pytestmark = pytest.mark.skipif(
    not ext_loader.check_ops_exist(), reason='mmcv._ext is not built')

CUDA = pytest.param(
    'cuda',
    marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason='requires CUDA'))
DEVICES = ['cpu', CUDA]

ALL_TESTS = ('test_schema', 'test_autograd_registration', 'test_faketensor',
             'test_aot_dispatch_dynamic')


def _ops():
    ext_loader.load()
    return torch.ops.mmcv


def _rand(*shape, device, dtype=torch.float, grad=False):
    return torch.rand(*shape, device=device, dtype=dtype).requires_grad_(grad)


def _rois(n, batch, size, device, dtype=torch.float):
    xy = torch.rand(n, 2, device=device, dtype=dtype) * size / 2
    wh = torch.rand(n, 2, device=device, dtype=dtype) * size / 2 + 1
    idx = torch.randint(0, batch, (n, 1), device=device).to(dtype)
    return torch.cat([idx, xy, xy + wh], 1)


def _boxes(n, device):
    return _rois(n, 1, 16, device)[:, 1:].contiguous()


def _cases(device):
    """(name, op, args, tests) for every op that has a kernel on device."""
    torch.manual_seed(0)
    ops = _ops()
    d = dict(device=device)
    cases = []

    def add(name, op, args, tests=ALL_TESTS):
        cases.append((name, op, args, tests))

    # ops with CPU and CUDA kernels
    add('bbox_overlaps', ops.bbox_overlaps,
        (_boxes(5, device), _boxes(6, device), 0, False, 0))
    add('bbox_overlaps_aligned', ops.bbox_overlaps,
        (_boxes(5, device), _boxes(5, device), 1, True, 1))

    x = _rand(2, 4, 7, 9, **d, grad=True)
    weight = _rand(6, 4, 3, 3, **d, grad=True)
    offset = (_rand(2, 2 * 2 * 9, 4, 5, **d) - 0.5).requires_grad_()
    mask = _rand(2, 2 * 9, 4, 5, **d, grad=True)
    bias = _rand(6, **d, grad=True)
    conv = (3, 3, 2, 2, 1, 1, 1, 1, 1, 2)  # kW kH dW dH padW padH dilW dilH
    add('deform_conv_forward', ops.deform_conv_forward,
        (x, weight, offset, *conv, 2))
    grad_out = _rand(2, 6, 4, 5, **d)
    add('deform_conv_backward_input', ops.deform_conv_backward_input,
        (x.detach(), offset.detach(), grad_out, weight.detach(), *conv, 2))
    add('deform_conv_backward_parameters', ops.deform_conv_backward_parameters,
        (x.detach(), offset.detach(), grad_out, weight.detach(), *conv, 1.0,
         2))
    mconv = (3, 3, 2, 2, 1, 1, 1, 1, 1, 2)  # kh kw sh sw ph pw dh dw g dg
    for b in (bias, None):
        add(f'modulated_deform_conv_forward(bias={b is not None})',
            ops.modulated_deform_conv_forward,
            (x, weight, b, offset, mask, *mconv))
        add(f'modulated_deform_conv_backward(bias={b is not None})',
            ops.modulated_deform_conv_backward,
            (x.detach(), weight.detach(), None if b is None else b.detach(),
             offset.detach(), mask.detach(), grad_out, *mconv))

    boxes, scores = _boxes(20, device), torch.rand(20, **d)
    add('nms', ops.nms, (boxes, scores, 0.5, 0))

    feat = _rand(2, 9, 4, 5, **d, grad=True)
    for psa_type in (0, 1):
        add(f'psamask_forward({psa_type})', ops.psamask_forward,
            (feat, psa_type, 3, 3))
        add(f'psamask_backward({psa_type})', ops.psamask_backward,
            (_rand(2, 20, 4, 5, **d), psa_type, 3, 3))

    rois = _rois(4, 2, 8, device)
    for pool_mode in (0, 1):  # max, avg
        add(f'roi_align_forward({pool_mode})', ops.roi_align_forward, (_rand(
            2, 3, 8, 8, **d, grad=True), rois, 2, 3, 0.9, 2, pool_mode, True))
    out, argmax_y, argmax_x = ops.roi_align_forward(
        _rand(2, 3, 8, 8, **d), rois, 2, 3, 0.9, 2, 0, True)
    add('roi_align_backward', ops.roi_align_backward,
        (torch.rand_like(out), rois, argmax_y, argmax_x, [2, 3, 8, 8], 2, 3,
         0.9, 2, 0, True))

    if device == 'cpu':
        add('softnms', ops.softnms, (boxes, scores, 0.3, 0.5, 1e-3, 1, 0))
        dets = torch.cat([boxes, scores[:, None]], 1)
        # a list of index tensors whose length depends on the data: no fake
        add('nms_match', ops.nms_match, (dets, 0.5), ('test_schema', ))
        return cases

    # CUDA only
    feat = _rand(2, 4, 5, 6, **d, grad=True)
    masks = _rand(2, 9, 10, 12, **d, grad=True)
    for name in ('carafe_naive', 'carafe'):
        add(f'{name}_forward', getattr(ops, f'{name}_forward'),
            (feat, masks, 3, 1, 2))
        add(f'{name}_backward', getattr(ops, f'{name}_backward'),
            (_rand(2, 4, 10, 12, **d), feat.detach(), masks.detach(), 3, 1, 2))

    x = _rand(2, 3, 8, 8, **d, grad=True)
    pool_offset = (_rand(4, 2, 2, 3, **d) - 0.5).requires_grad_()
    for off in (pool_offset, None):
        add(f'deform_roi_pool_forward(offset={off is not None})',
            ops.deform_roi_pool_forward, (x, rois, off, 2, 3, 0.9, 2, 0.1))
        add(f'deform_roi_pool_backward(offset={off is not None})',
            ops.deform_roi_pool_backward,
            (_rand(4, 3, 2, 3, **d), x.detach(), rois,
             None if off is None else off.detach(), 2, 3, 0.9, 2, 0.1))

    logits = _rand(6, 5, **d, grad=True)
    target = torch.randint(0, 5, (6, ), **d)
    class_weight = _rand(5, **d)
    for w in (None, class_weight):
        tag = f'(weight={w is not None})'
        add(f'sigmoid_focal_loss_forward{tag}', ops.sigmoid_focal_loss_forward,
            (logits, target, w, 2.0, 0.25))
        add(f'sigmoid_focal_loss_backward{tag}',
            ops.sigmoid_focal_loss_backward,
            (logits.detach(), target, w, 2.0, 0.25))
        add(f'softmax_focal_loss_forward{tag}', ops.softmax_focal_loss_forward,
            (logits, target, w, 2.0, 0.25))
        add(f'softmax_focal_loss_backward{tag}',
            ops.softmax_focal_loss_backward,
            (logits.detach().softmax(1), target, w, 2.0, 0.25))

    im = _rand(1, 4, 6, 6, **d)
    h_idx = torch.tensor([0, 1, 3, 5, 5], **d)
    w_idx = torch.tensor([0, 2, 2, 1, 5], **d)
    add('masked_im2col_forward', ops.masked_im2col_forward,
        (im, h_idx, w_idx, 3, 3, 1, 1))
    add('masked_col2im_forward', ops.masked_col2im_forward,
        (_rand(3, 5, **d), h_idx, w_idx, 6, 6, 3))

    rois_pool = _rois(4, 2, 8, device)
    add('roi_pool_forward', ops.roi_pool_forward,
        (_rand(2, 3, 8, 8, **d, grad=True), rois_pool, 2, 3, 0.9))
    out, argmax = ops.roi_pool_forward(
        _rand(2, 3, 8, 8, **d), rois_pool, 2, 3, 0.9)
    add('roi_pool_backward', ops.roi_pool_backward,
        (torch.rand_like(out), rois_pool, argmax, [2, 3, 8, 8], 2, 3, 0.9))

    shapes = torch.tensor([[4, 5], [2, 3]], **d)
    starts = torch.tensor([0, 20], **d)
    value = _rand(1, 26, 2, 4, **d, grad=True)
    locs = _rand(1, 3, 2, 2, 2, 2, **d, grad=True)
    attn = _rand(1, 3, 2, 2, 2, **d, grad=True)
    add('ms_deform_attn_forward', ops.ms_deform_attn_forward,
        (value, shapes, starts, locs, attn, 64))
    add('ms_deform_attn_backward', ops.ms_deform_attn_backward,
        (value.detach(), shapes, starts, locs.detach(), attn.detach(),
         _rand(1, 3, 8, **d), 64))

    x3d = torch.randn(4, 3, 10, **d)
    mean = ops.sync_bn_forward_mean(x3d)
    var = ops.sync_bn_forward_var(x3d, mean)
    add('sync_bn_forward_mean', ops.sync_bn_forward_mean, (x3d, ))
    add('sync_bn_forward_var', ops.sync_bn_forward_var, (x3d, mean))
    add('sync_bn_forward_output', ops.sync_bn_forward_output,
        (x3d, mean, var, torch.rand(3, **d), torch.rand(
            3, **d), torch.zeros(3, **d), torch.ones(3, **d), 1e-5, 0.1, 1))
    _, norm, std = ops.sync_bn_forward_output(x3d, mean, var,
                                              torch.rand(3, **d),
                                              torch.rand(3, **d),
                                              torch.zeros(3, **d),
                                              torch.ones(3, **d), 1e-5, 0.1, 1)
    grad3d = torch.randn(4, 3, 10, **d)
    grad_weight, grad_bias = ops.sync_bn_backward_param(grad3d, norm)
    add('sync_bn_backward_param', ops.sync_bn_backward_param, (grad3d, norm))
    add('sync_bn_backward_data', ops.sync_bn_backward_data,
        (grad3d, torch.rand(3, **d), grad_weight, grad_bias, norm, std))
    return cases


def _case_ids(device):
    return [c[0] for c in _cases(device)]


@pytest.mark.parametrize('device', DEVICES)
def test_opcheck(device):
    failures = []
    for name, op, args, tests in _cases(device):
        try:
            torch.library.opcheck(op, args, test_utils=tests)
        except Exception as e:  # noqa: BLE001
            failures.append(f'{name}: {str(e).splitlines()[0][:300]}')
    assert not failures, '\n'.join(failures)


def test_every_op_is_covered():
    _ops()  # load the library
    names = {
        n[len('mmcv::'):]
        for n in torch._C._dispatch_get_all_op_names()
        if n.startswith('mmcv::')
    }
    covered = set()
    for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
        covered |= {
            c[1]._qualified_op_name.split('::')[1]
            for c in _cases(device)
        }
    untested = names - covered - {
        'get_compiler_version', 'get_compiling_cuda_version'
    }
    if not torch.cuda.is_available():
        pytest.skip(f'CUDA-only ops not checked: {sorted(untested)}')
    assert not untested


# ---- backward functions, through the public Python API, in float64 ----

# CUDA backward kernels that accumulate with atomicAdd give slightly
# different sums from run to run
NONDET = dict(nondet_tol=1e-10)


@pytest.mark.parametrize('device', DEVICES)
def test_gradcheck_roi_align_psa_mask(device):
    from mmcv.ops import roi_align
    torch.manual_seed(0)
    x = torch.rand(2, 3, 8, 8, device=device, dtype=torch.double)
    rois = _rois(4, 2, 8, device, torch.double)
    assert gradcheck(lambda t: roi_align(t, rois, (2, 3), 0.9, 2, 'avg', True),
                     (x.requires_grad_(), ))
    feat = torch.rand(2, 9, 4, 5, device=device, dtype=torch.double)
    for psa_type in (0, 1):
        assert gradcheck(lambda t: _ops().psamask_forward(t, psa_type, 3, 3),
                         (feat.requires_grad_(), ))


@pytest.mark.parametrize('device', DEVICES)
def test_gradcheck_deform_conv(device):
    from mmcv.ops import deform_conv2d, modulated_deform_conv2d
    torch.manual_seed(0)
    dt = dict(device=device, dtype=torch.double)
    x = torch.rand(2, 4, 6, 7, **dt).requires_grad_()
    weight = torch.rand(6, 2, 3, 3, **dt).requires_grad_()
    # fractional offsets keep the samples away from the pixel grid, where
    # bilinear sampling is not differentiable
    offset = (torch.rand(2, 2 * 2 * 9, 3, 4, **dt) * 0.8 + 0.1)
    offset = offset.requires_grad_()
    mask = torch.rand(2, 2 * 9, 3, 4, **dt).requires_grad_()
    bias = torch.rand(6, **dt).requires_grad_()
    assert gradcheck(
        lambda i, o, w: deform_conv2d(i, o, w, 2, 1, 1, 2, 2, False, 2),
        (x, offset, weight), **NONDET)
    assert gradcheck(
        lambda i, o, m, w, b: modulated_deform_conv2d(
            i, o, m, w, b, 2, 1, 1, 2, 2), (x, offset, mask, weight, bias),
        **NONDET)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_gradcheck_cuda_only_ops():
    from mmcv.ops import (carafe, carafe_naive, deform_roi_pool,
                          sigmoid_focal_loss, softmax_focal_loss)
    from mmcv.ops.multi_scale_deform_attn import \
        multi_scale_deformable_attn_cuda
    torch.manual_seed(0)
    dt = dict(device='cuda', dtype=torch.double)
    feat = torch.rand(2, 4, 5, 6, **dt).requires_grad_()
    masks = torch.rand(2, 9, 10, 12, **dt).softmax(1).requires_grad_()
    for fn in (carafe, carafe_naive):
        # the tolerances of test_carafe.py: at gradcheck's defaults the fast
        # kernel's mask gradient is off by ~1e-5, the same before the ops
        # were registered with TORCH_LIBRARY
        assert gradcheck(
            lambda f, m: fn(f, m, 3, 1, 2), (feat, masks),
            eps=1e-4,
            atol=1e-4,
            **NONDET)

    x = torch.rand(2, 3, 8, 8, **dt).requires_grad_()
    rois = _rois(4, 2, 8, 'cuda', torch.double)
    offset = (torch.rand(4, 2, 2, 3, **dt) * 0.1).requires_grad_()
    assert gradcheck(
        lambda t, o: deform_roi_pool(t, rois, o, (2, 3), 0.9, 2, 0.1),
        (x, offset), **NONDET)

    # Focal loss against autograd of the same formulas in PyTorch, not
    # gradcheck: the sigmoid forward kernel uses expf, so its float64 loss
    # has float precision, too coarse for finite differences.
    logits = torch.randn(6, 5, **dt)
    target = torch.randint(0, 5, (6, ), device='cuda')
    onehot = torch.nn.functional.one_hot(target, 5).double()

    def sigmoid_ref(x):
        p = x.sigmoid()
        return (onehot * -0.25 * (1 - p)**2 * p.log() +
                (1 - onehot) * -0.75 * p**2 * (1 - p).log())

    def softmax_ref(x):
        p = x.softmax(1).gather(1, target[:, None])[:, 0]
        return -0.25 * (1 - p)**2 * p.log()

    for fn, ref in ((sigmoid_focal_loss, sigmoid_ref), (softmax_focal_loss,
                                                        softmax_ref)):
        # softmax with reduction='none' failed in the backward before the
        # reduction moved out of the autograd function
        for reduction in ('none', 'mean', 'sum'):
            x = logits.clone().requires_grad_()
            loss = fn(x, target, 2.0, 0.25, None, reduction)
            grad = torch.rand_like(loss)
            loss.backward(grad)
            x_ref = logits.clone().requires_grad_()
            loss_ref = ref(x_ref)
            if reduction != 'none':
                loss_ref = loss_ref.sum() / (6 if reduction == 'mean' else 1)
            loss_ref.backward(grad)
            torch.testing.assert_close(loss, loss_ref, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(x.grad, x_ref.grad)

    shapes = torch.tensor([[4, 5], [2, 3]], device='cuda')
    starts = torch.tensor([0, 20], device='cuda')
    value = torch.rand(1, 26, 2, 4, **dt).requires_grad_()
    locs = (torch.rand(1, 3, 2, 2, 2, 2, **dt) * 0.8 + 0.1).requires_grad_()
    attn = torch.rand(1, 3, 2, 2, 2, **dt).requires_grad_()
    assert gradcheck(
        lambda v, lo, a: multi_scale_deformable_attn_cuda(
            v, shapes, starts, lo, a, 64), (value, locs, attn), **NONDET)


# ---- torch.compile ----


def _compare_compiled(fn, inputs, backend='aot_eager'):
    torch._dynamo.reset()
    eager_in = [
        t.detach().clone().requires_grad_(t.requires_grad) for t in inputs
    ]
    out = fn(*eager_in)
    out.sum().backward()
    compiled = torch.compile(fn, fullgraph=True, backend=backend)
    comp_in = [
        t.detach().clone().requires_grad_(t.requires_grad) for t in inputs
    ]
    comp_out = compiled(*comp_in)
    comp_out.sum().backward()
    # inductor may reorder the float arithmetic around the opaque ops
    tol = dict(rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(comp_out, out, **tol)
    for a, b in zip(comp_in, eager_in):
        if a.requires_grad:
            torch.testing.assert_close(a.grad, b.grad, **tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('backend', ['aot_eager', 'inductor'])
def test_compile_fullgraph(backend):
    from mmcv.ops import (DeformConv2dPack, ModulatedDeformConv2dPack,
                          roi_align, sigmoid_focal_loss)
    from mmcv.ops.multi_scale_deform_attn import \
        multi_scale_deformable_attn_cuda
    torch.manual_seed(0)
    x = torch.rand(2, 4, 9, 9, device='cuda', requires_grad=True)
    for cls in (DeformConv2dPack, ModulatedDeformConv2dPack):
        model = cls(4, 6, 3, padding=1).cuda()
        torch.nn.init.normal_(model.conv_offset.weight, std=0.3)
        _compare_compiled(model, [x], backend)

    rois = _rois(4, 2, 9, 'cuda')
    _compare_compiled(lambda t: roi_align(t, rois, (2, 2), 1.0, 2), [x],
                      backend)

    target = torch.randint(0, 5, (6, ), device='cuda')
    _compare_compiled(lambda t: sigmoid_focal_loss(t, target, reduction='sum'),
                      [torch.randn(6, 5, device='cuda', requires_grad=True)],
                      backend)

    shapes = torch.tensor([[4, 5], [2, 3]], device='cuda')
    starts = torch.tensor([0, 20], device='cuda')
    _compare_compiled(
        lambda v, lo, a: multi_scale_deformable_attn_cuda(
            v, shapes, starts, lo, a, 64),
        [
            torch.rand(1, 26, 2, 4, device='cuda', requires_grad=True),
            torch.rand(1, 3, 2, 2, 2, 2, device='cuda', requires_grad=True),
            torch.rand(1, 3, 2, 2, 2, device='cuda', requires_grad=True)
        ], backend)


# ---- loading ----

_FRESH = """
import sys, torch
import mmcv.ops
assert 'mmcv._ext' not in sys.modules, 'import mmcv.ops loaded mmcv._ext'
if torch.cuda.is_available():
    # the first op call happens inside a compiled region
    from mmcv.ops import DeformConv2dPack
    model = DeformConv2dPack(2, 4, 3, padding=1).cuda()
    x = torch.rand(1, 2, 6, 6, device='cuda')
    out = torch.compile(model, fullgraph=True, backend='aot_eager')(x)
    torch.testing.assert_close(out, model(x))
    assert 'mmcv._ext' in sys.modules
print('ok')
"""


def test_lazy_load_and_compile_in_fresh_process():
    result = subprocess.run([sys.executable, '-c', _FRESH],
                            capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith('ok')
