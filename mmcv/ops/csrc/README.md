# Code Structure of CUDA operators

This folder contains all non-python code for MMCV custom ops. Please follow the same architecture if you want to add new ops.

## Directories Tree

```folder
.
├── common
│   ├── pytorch_cpp_helper.hpp
│   ├── pytorch_cuda_helper.hpp
│   └── cuda
│       ├── common_cuda_helper.hpp
│       ├── ...
│       └── ops_cuda_kernel.cuh
└── pytorch
    ├── module.cpp
    ├── info.cpp
    ├── ...
    ├── ops.cpp
    ├── cuda
    │   ├── ...
    │   └── ops_cuda.cu
    └── cpu
        ├── ...
        └── ops.cpp
```

## Components

This fork (branch `nv`) builds for NVIDIA GPUs and CPU only.

- `common`: This directory contains all tools and shared codes.
  - `cuda`: The cuda kernels.
- `pytorch`: The ops are registered with the PyTorch dispatcher (`TORCH_LIBRARY`) and reach Python as `torch.ops.mmcv.<op>`. Each `ops.cpp` defines the op signatures, allocates the outputs and calls the CPU or CUDA kernel. `module.cpp` turns the library into the importable module `mmcv._ext`, using only the Python limited API.
  - `cuda`: This directory contains cuda kernel launchers, which feed memory pointers of tensor to the cuda kernel in `common/cuda`. The launchers provide c++ interface of cuda implementation of corresponding custom ops.
  - `cpu`: This directory contain cpu implementations of corresponding custom ops.

## How to add new PyTorch ops?

1. Add the cuda kernel in `common/cuda` and its launcher in `pytorch/cuda`. A launcher writes into the output tensors it is given.

   ```c++
   // src/common/cuda/new_ops_cuda_kernel.cuh
   template <typename T>
   __global__ void new_ops_forward_cuda_kernel(const T* input, T* output, ...) {
       // forward here
   }

   // src/pytorch/cuda/new_ops_cuda.cu
   #include "new_ops_cuda_kernel.cuh"
   #include "pytorch_cuda_helper.hpp"

   void NewOpsForwardCUDAKernelLauncher(const Tensor input, Tensor output, ...) {
       at::cuda::CUDAGuard device_guard(input.device());
       cudaStream_t stream = at::cuda::getCurrentCUDAStream();
       AT_DISPATCH_FLOATING_TYPES_AND_HALF(
           input.scalar_type(), "new_ops_forward_cuda_kernel", ([&] {
               new_ops_forward_cuda_kernel<scalar_t>
                   <<<GET_BLOCKS(output.numel()), THREADS_PER_BLOCK, 0, stream>>>(
                       input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), ...);
           }));
       AT_CUDA_CHECK(cudaGetLastError());
   }
   ```

   A CPU kernel goes in `pytorch/cpu` the same way.

2. Register the op in `pytorch/new_ops.cpp`. The entry function takes the inputs, allocates the outputs and returns them (functional: no output arguments). Make the inputs contiguous there, since the kernels read raw memory.

   ```c++
   #include "pytorch_cpp_helper.hpp"

   #ifdef MMCV_WITH_CUDA
   void NewOpsForwardCUDAKernelLauncher(const Tensor input, Tensor output, ...);
   #endif
   void new_ops_forward_cpu(const Tensor input, Tensor output, ...);

   Tensor new_ops_forward(const Tensor &input_, int64_t param) {
       at::DeviceGuard guard(input_.device());
       const Tensor input = input_.contiguous();
       Tensor output = at::zeros_like(input);
       if (input.is_cuda()) {
           MMCV_CUDA_ONLY(NewOpsForwardCUDAKernelLauncher(input, output, ...);
                          return output;);
       }
       new_ops_forward_cpu(input, output, ...);
       return output;
   }

   TORCH_LIBRARY_FRAGMENT(mmcv, m) {
       m.def("new_ops_forward(Tensor input, int param) -> Tensor");
   }
   TORCH_LIBRARY_IMPL(mmcv, CPU, m) {
       m.impl("new_ops_forward", TORCH_FN(new_ops_forward));
   }
   #ifdef MMCV_WITH_CUDA
   TORCH_LIBRARY_IMPL(mmcv, CUDA, m) {
       m.impl("new_ops_forward", TORCH_FN(new_ops_forward));
   }
   #endif
   ```

   Define the schema outside `#ifdef MMCV_WITH_CUDA` even for a CUDA-only op, so that a CPU-only build still knows the op.

3. In the Python module, register the fake function (output shapes, for `torch.compile` and export) and the backward. They need the op to exist, so they run when the library is loaded.

   ```python
   import torch
   from ..utils import ext_loader

   @ext_loader.on_load
   def _register():

       @torch.library.register_fake('mmcv::new_ops_forward')
       def _(input, param):
           return input.new_empty(input.shape)

       torch.library.register_autograd(
           'mmcv::new_ops_forward', _backward, setup_context=_setup_context)


   def new_ops(input, param):
       ext_loader.load()  # imports mmcv._ext on first use
       return torch.ops.mmcv.new_ops_forward(input, param)
   ```

4. Add the op to `tests/test_ops/test_torch_library.py`, which runs `torch.library.opcheck` on every op.
