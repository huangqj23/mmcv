# Code Structure of CUDA operators

This folder contains all non-python code for MMCV custom ops. Please follow the same architecture if you want to add new ops.

## Directories Tree

```folder
.
├── common
│   ├── pytorch_cpp_helper.hpp
│   ├── pytorch_cuda_helper.hpp
│   ├── pytorch_device_registry.hpp
│   └── cuda
│       ├── common_cuda_helper.hpp
│       ├── ...
│       └── ops_cuda_kernel.cuh
└── pytorch
    ├── info.cpp
    ├── pybind.cpp
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
- `pytorch`: **PyTorch** custom ops are supported by binding C++ to Python with **pybind11**. The ops implementation and binding codes are placed in this directory.
  - `cuda`: This directory contains cuda kernel launchers, which feed memory pointers of tensor to the cuda kernel in `common/cuda`. The launchers provide c++ interface of cuda implementation of corresponding custom ops.
  - `cpu`: This directory contain cpu implementations of corresponding custom ops.

## How to add new PyTorch ops?

1. Add the cuda kernel in `common/cuda`.

   ```c++
   // src/common/cuda/new_ops_cuda_kernel.cuh

   template <typename T>
   __global__ void new_ops_forward_cuda_kernel(const T* input, T* output, ...) {
       // forward here
   }

   ```

   Add cuda kernel launcher in `pytorch/cuda`.

   ```c++
   // src/pytorch/cuda
   #include <new_ops_cuda_kernel.cuh>

   void NewOpsForwardCUDAKernelLauncher(Tensor input, Tensor output, ...){
       // initialize
       at::cuda::CUDAGuard device_guard(input.device());
       cudaStream_t stream = at::cuda::getCurrentCUDAStream();
       ...
       AT_DISPATCH_FLOATING_TYPES_AND_HALF(
           input.scalar_type(), "new_ops_forward_cuda_kernel", ([&] {
               new_ops_forward_cuda_kernel<scalar_t>
                   <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                       input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),...);
           }));
       AT_CUDA_CHECK(cudaGetLastError());
   }
   ```

2. Register implementation for different devices.

   ```c++
   // src/pytorch/cuda/cudabind.cpp
   ...

   Tensor new_ops_forward_cuda(Tensor input, Tensor output, ...){
       // implement cuda forward here
       // use `NewOpsForwardCUDAKernelLauncher` here
   }
   // declare interface here.
   Tensor new_ops_forward_impl(Tensor input, Tensor output, ...);
   // register the implementation for given device (CUDA here).
   REGISTER_DEVICE_IMPL(new_ops_forward_impl, CUDA, new_ops_forward_cuda);
   ```

3. Add ops implementation in `pytorch` directory. Select different implementations according to device type.

   ```c++
   // src/pytorch/new_ops.cpp
   Tensor new_ops_forward_impl(Tensor input, Tensor output, ...){
       // dispatch the implementation according to the device type of input.
       DISPATCH_DEVICE_IMPL(new_ops_forward_impl, input, output, ...);
   }
   ...

   Tensor new_ops_forward(Tensor input, Tensor output, ...){
       return new_ops_forward_impl(input, output, ...);
   }
   ```

4. Binding the implementation in `pytorch/pybind.cpp`

   ```c++
   // src/pytorch/pybind.cpp

   ...

   Tensor new_ops_forward(Tensor input, Tensor output, ...);

   ...

   // bind with pybind11
   m.def("new_ops_forward", &new_ops_forward, "new_ops_forward",
           py::arg("input"), py::arg("output"), ...);

   ...

   ```

5. Build MMCV again. Enjoy new ops in python

   ```python
   from ..utils import ext_loader
   ext_module = ext_loader.load_ext('_ext', ['new_ops_forward'])

   ...

   ext_module.new_ops_forward(input, output, ...)

   ```
