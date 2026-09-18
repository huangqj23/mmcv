# Copyright (c) OpenMMLab. All rights reserved.
"""``type`` written as a class must behave exactly like the registered name."""
import pytest
import torch
import torch.nn as nn
from mmengine.utils.dl_utils.parrots_wrapper import SyncBatchNorm

from mmcv.cnn import (ConvModule, build_norm_layer, build_upsample_layer,
                      upsample_method)
from mmcv.cnn.bricks import HSigmoid, Swish
from mmcv.cnn.bricks.wrappers import ConvTranspose2d
from mmcv.ops import CARAFEPack


def _same_init(a: nn.Module, b: nn.Module) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


@pytest.mark.parametrize('name,cls', [('Tanh', nn.Tanh), ('Sigmoid', nn.Sigmoid),
                                      ('PReLU', nn.PReLU), ('GELU', nn.GELU),
                                      ('HSigmoid', HSigmoid), ('Swish', Swish),
                                      ('ReLU', nn.ReLU), ('SiLU', nn.SiLU)])
def test_conv_module_act_inplace(name, cls):
    # Tanh & co. take no ``inplace``; a class type must not get one injected
    by_str = ConvModule(3, 8, 3, act_cfg=dict(type=name))
    by_cls = ConvModule(3, 8, 3, act_cfg=dict(type=cls))
    assert repr(by_str) == repr(by_cls)


def test_conv_module_leaky_relu_init():
    torch.manual_seed(0)
    by_str = ConvModule(3, 8, 3, act_cfg=dict(type='LeakyReLU', negative_slope=0.2))
    torch.manual_seed(0)
    by_cls = ConvModule(3, 8, 3, act_cfg=dict(type=nn.LeakyReLU, negative_slope=0.2))
    assert _same_init(by_str, by_cls)


def test_norm_sync_bn_and_gn():
    name_s, bn_s = build_norm_layer(dict(type='SyncBN'), 8)
    name_c, bn_c = build_norm_layer(dict(type=SyncBatchNorm), 8)
    assert name_s == name_c and type(bn_s) is type(bn_c)
    name_s, gn_s = build_norm_layer(dict(type='GN', num_groups=2), 8)
    name_c, gn_c = build_norm_layer(dict(type=nn.GroupNorm, num_groups=2), 8)
    assert name_s == name_c and repr(gn_s) == repr(gn_c)


@pytest.mark.parametrize('mode', ['nearest', 'bilinear'])
def test_upsample_mode_is_preserved(mode):
    by_str = build_upsample_layer(dict(type=mode, scale_factor=2))
    by_cls = build_upsample_layer(dict(type=nn.Upsample, scale_factor=2, mode=mode))
    assert repr(by_str) == repr(by_cls)
    x = torch.randn(1, 2, 5, 7)
    assert torch.equal(by_str(x), by_cls(x))


def test_upsample_class_default_mode():
    layer = build_upsample_layer(dict(type=nn.Upsample, scale_factor=2))
    assert layer.mode == 'nearest'


def test_upsample_method():
    assert upsample_method(None) is None
    assert upsample_method(dict(type=None)) is None
    for s in ('deconv', 'nearest', 'bilinear', 'carafe', 'pixel_shuffle'):
        assert upsample_method(dict(type=s)) == s
    assert upsample_method(dict(type=nn.Upsample)) == 'nearest'
    assert upsample_method(dict(type=nn.Upsample, mode='bilinear')) == 'bilinear'
    assert upsample_method(dict(type=ConvTranspose2d)) == 'deconv'
    assert upsample_method(dict(type=CARAFEPack)) == 'carafe'
