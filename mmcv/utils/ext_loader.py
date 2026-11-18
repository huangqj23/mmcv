# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import importlib.util
from typing import Any, List


class _LazyExt:
    """The compiled extension ``mmcv.<name>``, imported on first use.

    ``import mmcv.ops`` therefore works without the extension: the pure
    PyTorch ops (``point_sample``, ``CrissCrossAttention``, the PyTorch path
    of ``MultiScaleDeformableAttention``, ...) keep working, and a compiled op
    only raises when it is called.
    """

    def __init__(self, name: str, funcs: List[str]):
        self._name = name
        self._funcs = funcs
        self._ext = None

    def _load(self):
        if self._ext is None:
            try:
                ext = importlib.import_module('mmcv.' + self._name)
            except ImportError as e:
                raise ImportError(
                    f'mmcv.{self._name} is not available, so the compiled ops '
                    f'{self._funcs} cannot run. Build mmcv with '
                    'MMCV_WITH_OPS=1 (and FORCE_CUDA=1 for the CUDA kernels)'
                ) from e
            for fun in self._funcs:
                assert hasattr(ext, fun), f'{fun} miss in module {self._name}'
            self._ext = ext
        return self._ext

    def __getattr__(self, attr: str) -> Any:
        value = getattr(self._load(), attr)
        setattr(self, attr, value)  # later lookups skip __getattr__
        return value


def load_ext(name: str, funcs: List[str]) -> Any:
    return _LazyExt(name, funcs)


def check_ops_exist() -> bool:
    return importlib.util.find_spec('mmcv._ext') is not None
