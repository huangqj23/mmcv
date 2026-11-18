# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
import torch.nn as nn

from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE, IS_NPU_AVAILABLE


class Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.view(-1)
        target = target.view(-1)
        return torch.mean(input - target)


class TestPSAMask:

    @pytest.mark.parametrize('device', [
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
        pytest.param(
            'mlu',
            marks=pytest.mark.skipif(
                not IS_MLU_AVAILABLE, reason='requires MLU support')),
        pytest.param(
            'npu',
            marks=pytest.mark.skipif(
                not IS_NPU_AVAILABLE, reason='requires NPU support'))
    ])
    def test_psa_mask_collect(self, device):
        from mmcv.ops import PSAMask
        test_loss = Loss()

        input = np.fromfile(
            'tests/data/for_psa_mask/psa_input.bin', dtype=np.float32)
        output_collect = np.fromfile(
            'tests/data/for_psa_mask/psa_output_collect.bin', dtype=np.float32)

        input = input.reshape((4, 16, 8, 8))
        output_collect = output_collect.reshape((4, 64, 8, 8))
        label = torch.ones((4, 64, 8, 8))

        input = torch.FloatTensor(input)
        input.requires_grad = True

        psamask_collect = PSAMask('collect', (4, 4))

        # test collect cpu
        test_output = psamask_collect(input)
        loss = test_loss(test_output, label)
        loss.backward()
        test_output = test_output.detach().numpy()
        assert np.allclose(test_output, output_collect)
        assert test_output.shape == output_collect.shape

        psamask_collect.to(device)
        input = input.to(device)
        label = label.to(device)

        # test collect on device
        test_output = psamask_collect(input)
        loss = test_loss(test_output, label)
        loss.backward()
        test_output = test_output.detach().cpu().numpy()
        assert np.allclose(test_output, output_collect)
        assert test_output.shape == output_collect.shape

    @pytest.mark.parametrize('device', [
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
        pytest.param(
            'mlu',
            marks=pytest.mark.skipif(
                not IS_MLU_AVAILABLE, reason='requires MLU support')),
        pytest.param(
            'npu',
            marks=pytest.mark.skipif(
                not IS_NPU_AVAILABLE, reason='requires NPU support'))
    ])
    def test_psa_mask_distribute(self, device):
        from mmcv.ops import PSAMask
        test_loss = Loss()

        input = np.fromfile(
            'tests/data/for_psa_mask/psa_input.bin', dtype=np.float32)
        output_distribute = np.fromfile(
            'tests/data/for_psa_mask/psa_output_distribute.bin',
            dtype=np.float32)

        input = input.reshape((4, 16, 8, 8))
        output_distribute = output_distribute.reshape((4, 64, 8, 8))
        label = torch.ones((4, 64, 8, 8))

        input = torch.FloatTensor(input)
        input.requires_grad = True

        psamask_distribute = PSAMask('distribute', (4, 4))

        # test distribute cpu
        test_output = psamask_distribute(input)
        loss = test_loss(test_output, label)
        loss.backward()
        test_output = test_output.detach().numpy()
        assert np.allclose(test_output, output_distribute)
        assert test_output.shape == output_distribute.shape

        psamask_distribute.to(device)
        input = input.to(device)
        label = label.to(device)

        # test distribute on device
        test_output = psamask_distribute(input)
        loss = test_loss(test_output, label)
        loss.backward()
        test_output = test_output.detach().cpu().numpy()
        assert np.allclose(test_output, output_distribute)
        assert test_output.shape == output_distribute.shape


@pytest.mark.parametrize('device', [
    'cpu',
    pytest.param(
        'cuda',
        marks=pytest.mark.skipif(
            not IS_CUDA_AVAILABLE, reason='requires CUDA support'))
])
@pytest.mark.parametrize('psa_type', [0, 1])
@pytest.mark.parametrize(
    'h, w, h_mask, w_mask',
    [
        (8, 8, 4, 4),  # the shapes of the tests above
        (6, 10, 7, 9),  # window smaller than the map
        (4, 5, 9, 11),  # window covers more than the map
        (5, 7, 13, 3),  # mixed
        (1, 3, 3, 5),  # a single row
    ])
def test_psa_mask_pytorch(device, psa_type, h, w, h_mask, w_mask):
    """The plain-PyTorch PSAMask (export path) equals the kernels."""
    from mmcv.ops.psa_mask import PSAMaskFunction, psa_mask_pytorch
    torch.manual_seed(0)
    x = torch.randn(
        2, h_mask * w_mask, h, w, device=device, requires_grad=True)
    ref = PSAMaskFunction.apply(x, psa_type, (h_mask, w_mask))
    out = psa_mask_pytorch(x, psa_type, (h_mask, w_mask))
    assert torch.equal(out, ref)
    grad = torch.randn_like(ref)
    grad_ref, = torch.autograd.grad(ref, x, grad)
    grad_out, = torch.autograd.grad(out, x, grad)
    assert torch.equal(grad_out, grad_ref)
