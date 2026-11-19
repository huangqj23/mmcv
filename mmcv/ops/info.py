# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..utils import ext_loader


def get_compiler_version():
    ext_loader.load()
    return torch.ops.mmcv.get_compiler_version()


def get_compiling_cuda_version():
    ext_loader.load()
    return torch.ops.mmcv.get_compiling_cuda_version()
