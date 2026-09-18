# Copyright (c) OpenMMLab. All rights reserved.
import inspect
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import xavier_init
from mmengine.registry import (MODELS, cfg_type_matches, cfg_type_name,
                               resolve_cfg_type)

MODELS.register_module('nearest', module=nn.Upsample)
MODELS.register_module('bilinear', module=nn.Upsample)


@MODELS.register_module(name='pixel_shuffle')
class PixelShufflePack(nn.Module):
    """Pixel Shuffle upsample layer.

    This module packs `F.pixel_shuffle()` and a nn.Conv2d module together to
    achieve a simple upsampling with pixel shuffle.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of the conv layer to expand the
            channels.
    """

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int,
                 upsample_kernel: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weights(self):
        xavier_init(self.upsample_conv, distribution='uniform')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


def build_upsample_layer(cfg: Dict, *args, **kwargs) -> nn.Module:
    """Build upsample layer.

    Args:
        cfg (dict): The upsample layer config, which should contain:

            - type (str): Layer type.
            - scale_factor (int): Upsample ratio, which is not applicable to
              deconv.
            - layer args: Args needed to instantiate a upsample layer.
        args (argument list): Arguments passed to the ``__init__``
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the
            ``__init__`` method of the corresponding conv layer.

    Returns:
        nn.Module: Created upsample layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError(
            f'the cfg dict must contain the key "type", but got {cfg}')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')

    if isinstance(layer_type, str):
        # Switch registry to the target scope. If `upsample` cannot be found
        # in the registry, fallback to search `upsample` in the
        # mmengine.MODELS.
        with MODELS.switch_scope_and_registry(None) as registry:
            upsample = registry.get(layer_type)
        if upsample is None:
            raise KeyError(f'Cannot find {upsample} in registry under scope '
                           f'name {registry.scope}')
        if upsample is nn.Upsample:
            # 'nearest' / 'bilinear' are both registered as nn.Upsample; the
            # interpolation mode lives in the name.
            cfg_['mode'] = layer_type
    else:
        # ``type`` given as a class (pure-Python style config). nn.Upsample
        # then takes its interpolation mode from the ``mode`` key, e.g.
        # ``dict(type=nn.Upsample, scale_factor=2, mode='bilinear')``;
        # without it PyTorch's default 'nearest' applies.
        upsample = resolve_cfg_type(layer_type, MODELS)
        if upsample is nn.Upsample:
            cfg_.setdefault('mode', 'nearest')
    layer = upsample(*args, **kwargs, **cfg_)
    return layer


def upsample_method(cfg: Optional[Dict]) -> Optional[str]:
    """Traditional upsample method name of an ``upsample_cfg``.

    Code that dispatches on the method name (``'deconv'``, ``'carafe'``,
    ``'nearest'``, ...) should call this instead of reading ``cfg['type']`` so
    that ``type`` may also be written as a class:

    - string ``type``: returned unchanged;
    - ``nn.Upsample``: the ``mode`` key (default ``'nearest'``);
    - classes registered as ``'deconv'`` / ``'pixel_shuffle'`` / ``'carafe'``:
      that name;
    - anything else: its registered name, or ``__name__``.
    """
    if cfg is None:
        return None
    t = cfg.get('type')
    if t is None or isinstance(t, str):
        return t
    t = resolve_cfg_type(t, MODELS)
    if t is nn.Upsample:
        return cfg.get('mode', 'nearest')
    for name in ('deconv', 'pixel_shuffle', 'carafe'):
        if cfg_type_matches(t, name):
            return name
    return cfg_type_name(t)
