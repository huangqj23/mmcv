# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import os
import warnings

import numpy as np
import onnx
import pytest
import torch
import torch.nn as nn

onnx_file = 'tmp.onnx'


def _export(model, inputs, **kwargs):
    """Export with the TorchScript exporter, which uses the ``symbolic`` of
    the ops' autograd functions.

    torch 2.9 made the dynamo exporter the default and deprecated this one.
    """
    if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
        kwargs['dynamo'] = False
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        with torch.no_grad():
            torch.onnx.export(model, inputs, onnx_file, **kwargs)


@pytest.fixture(autouse=True)
def run_before_and_after_test():
    # clear onnx_file before test
    if os.path.exists(onnx_file):
        os.remove(onnx_file)

    yield

    # clear onnx_file after test
    if os.path.exists(onnx_file):
        os.remove(onnx_file)


class WrapFunction(nn.Module):

    def __init__(self, wrapped_function):
        super().__init__()
        self.wrapped_function = wrapped_function

    def forward(self, *args, **kwargs):
        return self.wrapped_function(*args, **kwargs)


def test_roialign():
    rt = pytest.importorskip('onnxruntime')
    try:
        from mmcv.ops import roi_align
    except (ImportError, ModuleNotFoundError):
        pytest.skip('roi_align op is not successfully compiled')

    # roi align config
    pool_h = 2
    pool_w = 2
    spatial_scale = 1.0
    sampling_ratio = 2

    inputs = [([[[[1., 2.], [3., 4.]]]], [[0., 0., 0., 1., 1.]]),
              ([[[[1., 2.], [3., 4.]], [[4., 3.],
                                        [2., 1.]]]], [[0., 0., 0., 1., 1.]]),
              ([[[[1., 2., 5., 6.], [3., 4., 7., 8.], [9., 10., 13., 14.],
                  [11., 12., 15., 16.]]]], [[0., 0., 0., 3., 3.]])]

    def warpped_function(torch_input, torch_rois):
        return roi_align(torch_input, torch_rois, (pool_w, pool_h),
                         spatial_scale, sampling_ratio, 'avg', True)

    for case in inputs:
        np_input = np.array(case[0], dtype=np.float32)
        np_rois = np.array(case[1], dtype=np.float32)
        input = torch.from_numpy(np_input)
        rois = torch.from_numpy(np_rois)

        # compute pytorch_output
        with torch.no_grad():
            pytorch_output = roi_align(input, rois, (pool_w, pool_h),
                                       spatial_scale, sampling_ratio, 'avg',
                                       True)

        # export and load onnx model
        wrapped_model = WrapFunction(warpped_function)
        _export(
            wrapped_model, (input, rois),
            export_params=True,
            keep_initializers_as_inputs=True,
            input_names=['input', 'rois'],
            opset_version=11)

        onnx_model = onnx.load(onnx_file)
        session_options = rt.SessionOptions()

        # compute onnx_output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 2)
        sess = rt.InferenceSession(
            onnx_file, session_options, providers=['CPUExecutionProvider'])
        onnx_output = sess.run(None, {
            'input': input.detach().numpy(),
            'rois': rois.detach().numpy()
        })
        onnx_output = onnx_output[0]

        # allclose

        assert np.allclose(pytorch_output, onnx_output, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='test requires GPU')
def test_roipool():
    rt = pytest.importorskip('onnxruntime')
    from mmcv.ops import roi_pool

    # roi pool config
    pool_h = 2
    pool_w = 2
    spatial_scale = 1.0

    inputs = [([[[[1., 2.], [3., 4.]]]], [[0., 0., 0., 1., 1.]]),
              ([[[[1., 2.], [3., 4.]], [[4., 3.],
                                        [2., 1.]]]], [[0., 0., 0., 1., 1.]]),
              ([[[[1., 2., 5., 6.], [3., 4., 7., 8.], [9., 10., 13., 14.],
                  [11., 12., 15., 16.]]]], [[0., 0., 0., 3., 3.]])]

    def warpped_function(torch_input, torch_rois):
        return roi_pool(torch_input, torch_rois, (pool_w, pool_h),
                        spatial_scale)

    for case in inputs:
        np_input = np.array(case[0], dtype=np.float32)
        np_rois = np.array(case[1], dtype=np.float32)
        input = torch.from_numpy(np_input).cuda()
        rois = torch.from_numpy(np_rois).cuda()

        # compute pytorch_output
        with torch.no_grad():
            pytorch_output = roi_pool(input, rois, (pool_w, pool_h),
                                      spatial_scale)
            pytorch_output = pytorch_output.cpu()

        # export and load onnx model
        wrapped_model = WrapFunction(warpped_function)
        _export(
            wrapped_model, (input, rois),
            export_params=True,
            keep_initializers_as_inputs=True,
            input_names=['input', 'rois'],
            opset_version=11)
        onnx_model = onnx.load(onnx_file)

        # compute onnx_output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 2)
        sess = rt.InferenceSession(
            onnx_file, providers=['CPUExecutionProvider'])
        onnx_output = sess.run(
            None, {
                'input': input.detach().cpu().numpy(),
                'rois': rois.detach().cpu().numpy()
            })
        onnx_output = onnx_output[0]

        # allclose
        assert np.allclose(pytorch_output, onnx_output, atol=1e-3)


def _test_symbolic(model, inputs, symbol_name):
    _export(model, inputs, opset_version=11)

    model = onnx.load(onnx_file)
    nodes = model.graph.node

    symbol_exist = False
    for n in nodes:
        if n.op_type == symbol_name:
            symbol_exist = True
    assert symbol_exist


@pytest.mark.skipif(not torch.cuda.is_available(), reason='test requires GPU')
def test_carafe():
    from mmcv.ops import CARAFENaive
    feat = torch.randn(2, 64, 3, 3, device='cuda').double()
    mask = torch.randn(2, 100, 6, 6, device='cuda').sigmoid().double()
    _test_symbolic(CARAFENaive(5, 4, 2), (feat, mask), 'MMCVCARAFENaive')


DEVICES = [
    'cpu',
    pytest.param(
        'cuda',
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason='requires CUDA'))
]


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('stride, padding, dilation, groups, deform_groups',
                         [(1, 1, 1, 1, 1), (2, 1, 1, 2, 2), (1, 2, 2, 1, 4),
                          (2, 0, 1, 4, 1)])
def test_deform_conv2d_pytorch(device, stride, padding, dilation, groups,
                               deform_groups):
    # the export path of DCN / DCNv2 against the kernels
    from mmcv.ops import deform_conv2d, modulated_deform_conv2d
    from mmcv.ops.deform_conv import deform_conv2d_pytorch
    torch.manual_seed(0)
    x = torch.randn(2, 8, 11, 13, device=device)
    weight = torch.randn(12, 8 // groups, 3, 3, device=device)
    out_h = (11 + 2 * padding - dilation * 2 - 1) // stride + 1
    out_w = (13 + 2 * padding - dilation * 2 - 1) // stride + 1
    offset = torch.randn(
        2, deform_groups * 18, out_h, out_w, device=device) * 2
    mask = torch.rand(2, deform_groups * 9, out_h, out_w, device=device)
    bias = torch.randn(12, device=device)
    args = (stride, padding, dilation, groups, deform_groups)
    torch.testing.assert_close(
        deform_conv2d_pytorch(x, offset, weight, None, None, *args),
        deform_conv2d(x, offset, weight, *args),
        rtol=1e-4,
        atol=1e-4)
    torch.testing.assert_close(
        deform_conv2d_pytorch(x, offset, weight, mask, bias, *args),
        modulated_deform_conv2d(x, offset, mask, weight, bias, *args),
        rtol=1e-4,
        atol=1e-4)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('modulated', [False, True])
@pytest.mark.parametrize('mode', ['grid_sample', 'deform_conv'])
def test_deform_conv(device, modulated, mode, monkeypatch):
    from onnx.reference import ReferenceEvaluator

    from mmcv.ops import DeformConv2dPack, ModulatedDeformConv2dPack
    from mmcv.ops import deform_conv as deform_conv_module
    monkeypatch.setattr(deform_conv_module, 'ONNX_EXPORT_MODE', mode)
    torch.manual_seed(0)
    cls = ModulatedDeformConv2dPack if modulated else DeformConv2dPack
    model = cls(4, 6, 3, stride=2, padding=1, deform_groups=2).eval()
    # the offsets start at zero; random ones exercise the bilinear sampling
    nn.init.normal_(model.conv_offset.weight, std=0.5)
    model = model.to(device)
    x = torch.randn(2, 4, 7, 9, device=device)
    with torch.no_grad():
        expected = model(x).cpu().numpy()

    opset = 17 if mode == 'grid_sample' else 19
    _export(model, (x, ), opset_version=opset, input_names=['x'])
    onnx_model = onnx.load(onnx_file)
    nodes = onnx_model.graph.node
    assert all(n.domain in ('', 'ai.onnx') for n in nodes)
    deform = [n for n in nodes if n.op_type == 'DeformConv']
    if mode == 'grid_sample':
        # standard ops that ONNX Runtime, OpenVINO and TensorRT all run
        assert not deform
        assert any(n.op_type == 'GridSample' for n in nodes)
        rt = pytest.importorskip('onnxruntime')
        sess = rt.InferenceSession(
            onnx_file, providers=['CPUExecutionProvider'])
        out = sess.run(None, {'x': x.cpu().numpy()})[0]
    else:
        # ONNX DeformConv, checked against the ONNX reference implementation,
        # so the kernels and the ONNX definition agree on the offset / mask
        # layout and the sampling
        assert len(deform) == 1
        assert len(deform[0].input) == (5 if modulated else 3)
        out = ReferenceEvaluator(onnx_model).run(None,
                                                 {'x': x.cpu().numpy()})[0]
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)


def test_deform_conv_node_needs_opset_19(monkeypatch):
    from mmcv.ops import DeformConv2dPack
    from mmcv.ops import deform_conv as deform_conv_module
    monkeypatch.setattr(deform_conv_module, 'ONNX_EXPORT_MODE', 'deform_conv')
    model = DeformConv2dPack(2, 4, 3, padding=1).eval()
    with pytest.raises(Exception, match='opset_version >= 19'):
        _export(model, (torch.randn(1, 2, 4, 4), ), opset_version=17)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='test requires GPU')
def test_deform_roi_pool():
    from mmcv.ops import DeformRoIPoolPack
    x = torch.tensor([[[[1., 2.], [3., 4.]]]], device='cuda')
    rois = torch.tensor([[0., 0., 0., 1., 1.]], device='cuda')
    output_c = x.size(1)
    pool_h = 2
    pool_w = 2
    spatial_scale = 1.0
    sampling_ratio = 2
    model = DeformRoIPoolPack((pool_h, pool_w),
                              output_c,
                              spatial_scale=spatial_scale,
                              sampling_ratio=sampling_ratio).cuda()

    _test_symbolic(model, (x, rois), 'MMCVDeformRoIPool')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='test requires GPU')
def test_masked_conv():
    from mmcv.ops import MaskedConv2d
    x = torch.rand(1, 2, 4, 4, device='cuda')
    mask = torch.rand(1, 4, 4, device='cuda')
    _test_symbolic(
        MaskedConv2d(2, 4, 3, 1, 1).cuda(), (x, mask), 'MMCVMaskedConv2d')


@pytest.mark.parametrize('psa_type', ['collect', 'distribute'])
def test_psa_mask(psa_type):
    # exported as standard ops (pad / slice / reshape), so TensorRT and
    # OpenVINO run it without a plugin
    onnxruntime = pytest.importorskip('onnxruntime')
    from mmcv.ops import PSAMask
    from mmcv.ops.psa_mask import PSAMaskFunction
    input = torch.rand(2, 63, 6, 10)
    model = PSAMask(psa_type, (7, 9))
    _export(model, input, opset_version=17)
    onnx_model = onnx.load(onnx_file)
    assert all(n.domain in ('', 'ai.onnx') for n in onnx_model.graph.node)
    sess = onnxruntime.InferenceSession(
        onnx_file, providers=['CPUExecutionProvider'])
    out = sess.run(None, {sess.get_inputs()[0].name: input.numpy()})[0]
    ref = PSAMaskFunction.apply(input, model.psa_type_enum, (7, 9))
    assert np.array_equal(out, ref.numpy())
