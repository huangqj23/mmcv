// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void CARAFENAIVEForwardCUDAKernelLauncher(const Tensor features,
                                          const Tensor masks, Tensor output,
                                          const int kernel_size,
                                          const int group_size,
                                          const int scale_factor);

void CARAFENAIVEBackwardCUDAKernelLauncher(
    const Tensor top_grad, const Tensor features, const Tensor masks,
    Tensor bottom_grad, Tensor mask_grad, const int kernel_size,
    const int group_size, const int scale_factor);

Tensor carafe_naive_forward(const Tensor &features_, const Tensor &masks_,
                            int64_t kernel_size, int64_t group_size,
                            int64_t scale_factor) {
  at::DeviceGuard guard(features_.device());
  const Tensor features = features_.contiguous(), masks = masks_.contiguous();
  Tensor output =
      at::zeros({features.size(0), features.size(1),
                 features.size(2) * scale_factor,
                 features.size(3) * scale_factor},
                features.options());
  CARAFENAIVEForwardCUDAKernelLauncher(features, masks, output, kernel_size,
                                       group_size, scale_factor);
  return output;
}

std::tuple<Tensor, Tensor> carafe_naive_backward(
    const Tensor &top_grad_, const Tensor &features_, const Tensor &masks_,
    int64_t kernel_size, int64_t group_size, int64_t scale_factor) {
  at::DeviceGuard guard(features_.device());
  const Tensor top_grad = top_grad_.contiguous(),
               features = features_.contiguous(), masks = masks_.contiguous();
  Tensor bottom_grad = at::zeros_like(features);
  Tensor mask_grad = at::zeros_like(masks);
  CARAFENAIVEBackwardCUDAKernelLauncher(top_grad, features, masks, bottom_grad,
                                        mask_grad, kernel_size, group_size,
                                        scale_factor);
  return {bottom_grad, mask_grad};
}

TORCH_LIBRARY_IMPL(mmcv, CUDA, m) {
  m.impl("carafe_naive_forward", TORCH_FN(carafe_naive_forward));
  m.impl("carafe_naive_backward", TORCH_FN(carafe_naive_backward));
}
#endif

TORCH_LIBRARY_FRAGMENT(mmcv, m) {
  m.def(
      "carafe_naive_forward(Tensor features, Tensor masks, int kernel_size, "
      "int group_size, int scale_factor) -> Tensor");
  m.def(
      "carafe_naive_backward(Tensor top_grad, Tensor features, Tensor masks, "
      "int kernel_size, int group_size, int scale_factor) -> (Tensor, Tensor)");
}
