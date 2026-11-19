// Copyright (c) OpenMMLab. All rights reserved
// Modified from
// https://github.com/hszhao/semseg/blob/master/lib/psa/src
#include "pytorch_cpp_helper.hpp"

void psamask_forward_cpu(const int psa_type, const Tensor input, Tensor output,
                         const int num_, const int h_feature,
                         const int w_feature, const int h_mask,
                         const int w_mask, const int half_h_mask,
                         const int half_w_mask);
void psamask_backward_cpu(const int psa_type, const Tensor grad_output,
                          Tensor grad_input, const int num_,
                          const int h_feature, const int w_feature,
                          const int h_mask, const int w_mask,
                          const int half_h_mask, const int half_w_mask);
#ifdef MMCV_WITH_CUDA
void PSAMaskForwardCUDAKernelLauncher(const int psa_type, const Tensor input,
                                      Tensor output, const int num_,
                                      const int h_feature, const int w_feature,
                                      const int h_mask, const int w_mask,
                                      const int half_h_mask,
                                      const int half_w_mask);
void PSAMaskBackwardCUDAKernelLauncher(
    const int psa_type, const Tensor grad_output, Tensor grad_input,
    const int num_, const int h_feature, const int w_feature, const int h_mask,
    const int w_mask, const int half_h_mask, const int half_w_mask);
#endif

// psa_type: 0 = collect, 1 = distribute.
// input: (N, h_mask * w_mask, H, W) -> output: (N, H * W, H, W)
Tensor psamask_forward(const Tensor &input_, int64_t psa_type, int64_t h_mask,
                       int64_t w_mask) {
  TORCH_CHECK(input_.dim() == 4 && input_.size(1) == h_mask * w_mask,
              "psamask: input must be (N, h_mask * w_mask, H, W)");
  at::DeviceGuard guard(input_.device());
  const Tensor input = input_.contiguous();
  const int64_t num = input.size(0), h = input.size(2), w = input.size(3);
  Tensor output = at::zeros({num, h * w, h, w}, input.options());
  if (input.is_cuda()) {
    MMCV_CUDA_ONLY(PSAMaskForwardCUDAKernelLauncher(
                       psa_type, input, output, num, h, w, h_mask, w_mask,
                       (h_mask - 1) / 2, (w_mask - 1) / 2);
                   return output;);
  }
  psamask_forward_cpu(psa_type, input, output, num, h, w, h_mask, w_mask,
                      (h_mask - 1) / 2, (w_mask - 1) / 2);
  return output;
}

// grad_output: (N, H * W, H, W) -> grad_input: (N, h_mask * w_mask, H, W)
Tensor psamask_backward(const Tensor &grad_output_, int64_t psa_type,
                        int64_t h_mask, int64_t w_mask) {
  at::DeviceGuard guard(grad_output_.device());
  const Tensor grad_output = grad_output_.contiguous();
  const int64_t num = grad_output.size(0), h = grad_output.size(2),
                w = grad_output.size(3);
  Tensor grad_input =
      at::zeros({num, h_mask * w_mask, h, w}, grad_output.options());
  if (grad_output.is_cuda()) {
    MMCV_CUDA_ONLY(PSAMaskBackwardCUDAKernelLauncher(
                       psa_type, grad_output, grad_input, num, h, w, h_mask,
                       w_mask, (h_mask - 1) / 2, (w_mask - 1) / 2);
                   return grad_input;);
  }
  psamask_backward_cpu(psa_type, grad_output, grad_input, num, h, w, h_mask,
                       w_mask, (h_mask - 1) / 2, (w_mask - 1) / 2);
  return grad_input;
}

TORCH_LIBRARY_FRAGMENT(mmcv, m) {
  m.def(
      "psamask_forward(Tensor input, int psa_type, int h_mask, int w_mask) "
      "-> Tensor");
  m.def(
      "psamask_backward(Tensor grad_output, int psa_type, int h_mask, "
      "int w_mask) -> Tensor");
}

TORCH_LIBRARY_IMPL(mmcv, CPU, m) {
  m.impl("psamask_forward", TORCH_FN(psamask_forward));
  m.impl("psamask_backward", TORCH_FN(psamask_backward));
}

#ifdef MMCV_WITH_CUDA
TORCH_LIBRARY_IMPL(mmcv, CUDA, m) {
  m.impl("psamask_forward", TORCH_FN(psamask_forward));
  m.impl("psamask_backward", TORCH_FN(psamask_backward));
}
#endif
