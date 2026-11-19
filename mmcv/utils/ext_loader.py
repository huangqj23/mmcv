# Copyright (c) OpenMMLab. All rights reserved.
"""Loading of the compiled ops.

``mmcv._ext`` registers its ops with the PyTorch dispatcher as
``torch.ops.mmcv.*``. It is imported on first use, so ``import mmcv.ops``
works without it: the pure PyTorch ops (``point_sample``,
``CrissCrossAttention``, the PyTorch path of ``MultiScaleDeformableAttention``,
...) keep working, and a compiled op only raises when it is called.
"""
import importlib
import importlib.util
from typing import Any, Callable, List

import torch

_loaded = False
_callbacks: List[Callable[[], None]] = []
_seen = set()


def _constant_under_compile(fn):
    # torch.compile calls a function marked this way once while tracing and
    # keeps the result, so loading the library inside a compiled region does
    # not break the graph
    mark = getattr(torch.compiler, 'assume_constant_result', None)
    return mark(fn) if mark is not None else fn


@_constant_under_compile
def load() -> bool:
    """Import ``mmcv._ext`` once, then run the :func:`on_load` callbacks.

    Op modules call this right before ``torch.ops.mmcv.<op>``.

    Raises:
        ImportError: mmcv was installed without its compiled ops.
    """
    global _loaded
    if not _loaded:
        try:
            importlib.import_module('mmcv._ext')
        except ImportError as e:
            raise ImportError(
                'mmcv._ext is not available, so the compiled ops cannot run. '
                'Build mmcv with MMCV_WITH_OPS=1 (and FORCE_CUDA=1 for the '
                'CUDA kernels)') from e
        _loaded = True
        # the op modules register the fake and autograd functions; import
        # them all, so torch.ops.mmcv is complete however the library was
        # first reached
        importlib.import_module('mmcv.ops')
        while _callbacks:
            _callbacks.pop(0)()
    return True


def on_load(fn: Callable[[], None]) -> Callable[[], None]:
    """Run ``fn`` once ``mmcv._ext`` is loaded, or now if it already is.

    ``torch.library.register_fake`` and ``register_autograd`` need the op to
    exist, so op modules register them through this decorator.
    """
    key = f'{fn.__module__}.{fn.__qualname__}'
    if key in _seen:  # the module was reloaded; the op keeps its old ones
        return fn
    _seen.add(key)
    if _loaded:
        fn()
    else:
        _callbacks.append(fn)
    return fn


class _LazyExt:
    """``torch.ops.mmcv`` that loads the library on first attribute access.

    Only for code written against the old ``load_ext`` interface; the op
    signatures are the ones registered in ``mmcv/ops/csrc``.
    """

    def __init__(self, funcs: List[str]):
        self._funcs = funcs

    def __getattr__(self, attr: str) -> Any:
        load()
        return getattr(torch.ops.mmcv, attr)


def load_ext(name: str, funcs: List[str]) -> Any:
    assert name == '_ext', f'unknown extension mmcv.{name}'
    return _LazyExt(funcs)


def check_ops_exist() -> bool:
    return importlib.util.find_spec('mmcv._ext') is not None
