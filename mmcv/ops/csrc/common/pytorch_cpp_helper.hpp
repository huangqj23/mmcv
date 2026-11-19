#ifndef PYTORCH_CPP_HELPER
#define PYTORCH_CPP_HELPER
#include <torch/library.h>
#include <torch/types.h>

#include <vector>

using namespace at;

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) \
  TORCH_CHECK(x.device().type() == at::kCPU, #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA_INPUT(x) \
  CHECK_CUDA(x);            \
  CHECK_CONTIGUOUS(x)
#define CHECK_CPU_INPUT(x) \
  CHECK_CPU(x);            \
  CHECK_CONTIGUOUS(x)

// Each op is defined with TORCH_LIBRARY_FRAGMENT(mmcv, m) in the file of its
// entry function, and the dispatcher picks the CPU or CUDA kernel. An entry
// function registered for both keys wraps its CUDA branch in MMCV_CUDA_ONLY,
// so a CPU-only build fails with a clear message there.
#ifdef MMCV_WITH_CUDA
#define MMCV_CUDA_ONLY(...) __VA_ARGS__
#else
#define MMCV_CUDA_ONLY(...) \
  TORCH_CHECK(false, "mmcv was built without CUDA support")
#endif

#endif  // PYTORCH_CPP_HELPER
