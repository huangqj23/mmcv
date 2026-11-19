// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void CARAFEForwardCUDAKernelLauncher(const Tensor features, const Tensor masks,
                                     Tensor rfeatures, Tensor routput,
                                     Tensor rmasks, Tensor output,
                                     const int kernel_size,
                                     const int group_size,
                                     const int scale_factor);

void CARAFEBackwardCUDAKernelLauncher(
    const Tensor top_grad, const Tensor rfeatures, const Tensor masks,
    Tensor rtop_grad, Tensor rbottom_grad_hs, Tensor rbottom_grad,
    Tensor rmask_grad, Tensor bottom_grad, Tensor mask_grad,
    const int kernel_size, const int group_size, const int scale_factor);

Tensor carafe_forward(const Tensor &features_, const Tensor &masks_,
                      int64_t kernel_size, int64_t group_size,
                      int64_t scale_factor) {
  at::DeviceGuard guard(features_.device());
  const Tensor features = features_.contiguous(), masks = masks_.contiguous();
  Tensor output =
      at::zeros({features.size(0), features.size(1),
                 features.size(2) * scale_factor,
                 features.size(3) * scale_factor},
                features.options());
  // NHWC scratch buffers; the launcher resizes them
  Tensor routput = at::zeros_like(output);
  Tensor rfeatures = at::zeros_like(features);
  Tensor rmasks = at::zeros_like(masks);
  CARAFEForwardCUDAKernelLauncher(features, masks, rfeatures, routput, rmasks,
                                  output, kernel_size, group_size,
                                  scale_factor);
  return output;
}

std::tuple<Tensor, Tensor> carafe_backward(const Tensor &top_grad_,
                                           const Tensor &features_,
                                           const Tensor &masks_,
                                           int64_t kernel_size,
                                           int64_t group_size,
                                           int64_t scale_factor) {
  at::DeviceGuard guard(features_.device());
  const Tensor top_grad = top_grad_.contiguous(), masks = masks_.contiguous();
  // the forward launcher's NHWC copy of the features, rebuilt here instead of
  // being kept alive between forward and backward
  const Tensor rfeatures = features_.permute({0, 2, 3, 1}).contiguous();
  Tensor rtop_grad = at::zeros_like(top_grad);
  Tensor rbottom_grad_hs = at::zeros_like(top_grad);
  Tensor rbottom_grad = at::zeros_like(features_);
  Tensor rmask_grad = at::zeros_like(masks);
  Tensor bottom_grad = at::zeros_like(features_, at::MemoryFormat::Contiguous);
  Tensor mask_grad = at::zeros_like(masks);
  CARAFEBackwardCUDAKernelLauncher(top_grad, rfeatures, masks, rtop_grad,
                                   rbottom_grad_hs, rbottom_grad, rmask_grad,
                                   bottom_grad, mask_grad, kernel_size,
                                   group_size, scale_factor);
  return {bottom_grad, mask_grad};
}

TORCH_LIBRARY_IMPL(mmcv, CUDA, m) {
  m.impl("carafe_forward", TORCH_FN(carafe_forward));
  m.impl("carafe_backward", TORCH_FN(carafe_backward));
}
#endif

TORCH_LIBRARY_FRAGMENT(mmcv, m) {
  m.def(
      "carafe_forward(Tensor features, Tensor masks, int kernel_size, "
      "int group_size, int scale_factor) -> Tensor");
  m.def(
      "carafe_backward(Tensor top_grad, Tensor features, Tensor masks, "
      "int kernel_size, int group_size, int scale_factor) -> (Tensor, Tensor)");
}
