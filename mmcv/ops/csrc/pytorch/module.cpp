// Copyright (c) OpenMMLab. All rights reserved
// mmcv._ext defines no Python functions. Importing it loads this library, and
// the TORCH_LIBRARY_FRAGMENT blocks next to each op register the ops with the
// PyTorch dispatcher as torch.ops.mmcv.*. Only the Python limited API is used
// here, so with Py_LIMITED_API one build serves every CPython 3 version that
// the torch build supports.
#include <Python.h>

PyMODINIT_FUNC PyInit__ext(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT, "_ext", nullptr, -1, nullptr,
  };
  return PyModule_Create(&module_def);
}
