// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void ROIPoolForwardCUDAKernelLauncher(Tensor input, Tensor rois, Tensor output,
                                      Tensor argmax, int pooled_height,
                                      int pooled_width, float spatial_scale);

void ROIPoolBackwardCUDAKernelLauncher(Tensor grad_output, Tensor rois,
                                       Tensor argmax, Tensor grad_input,
                                       int pooled_height, int pooled_width,
                                       float spatial_scale);

// output and the int32 argmax the backward needs
std::tuple<Tensor, Tensor> roi_pool_forward(const Tensor &input_,
                                            const Tensor &rois_,
                                            int64_t pooled_height,
                                            int64_t pooled_width,
                                            double spatial_scale) {
  TORCH_CHECK(rois_.dim() == 2 && rois_.size(1) == 5,
              "roi_pool: rois must be (idx, x1, y1, x2, y2)");
  at::DeviceGuard guard(input_.device());
  const Tensor input = input_.contiguous(), rois = rois_.contiguous();
  Tensor output = at::zeros(
      {rois.size(0), input.size(1), pooled_height, pooled_width},
      input.options());
  Tensor argmax = at::zeros_like(output, output.options().dtype(at::kInt));
  ROIPoolForwardCUDAKernelLauncher(input, rois, output, argmax, pooled_height,
                                   pooled_width, spatial_scale);
  return {output, argmax};
}

Tensor roi_pool_backward(const Tensor &grad_output_, const Tensor &rois_,
                         const Tensor &argmax_, at::IntArrayRef input_shape,
                         int64_t pooled_height, int64_t pooled_width,
                         double spatial_scale) {
  at::DeviceGuard guard(grad_output_.device());
  const Tensor grad_output = grad_output_.contiguous(),
               rois = rois_.contiguous(), argmax = argmax_.contiguous();
  Tensor grad_input = at::zeros(input_shape, grad_output.options());
  ROIPoolBackwardCUDAKernelLauncher(grad_output, rois, argmax, grad_input,
                                    pooled_height, pooled_width,
                                    spatial_scale);
  return grad_input;
}

TORCH_LIBRARY_IMPL(mmcv, CUDA, m) {
  m.impl("roi_pool_forward", TORCH_FN(roi_pool_forward));
  m.impl("roi_pool_backward", TORCH_FN(roi_pool_backward));
}
#endif

TORCH_LIBRARY_FRAGMENT(mmcv, m) {
  m.def(
      "roi_pool_forward(Tensor input, Tensor rois, int pooled_height, "
      "int pooled_width, float spatial_scale) -> (Tensor, Tensor)");
  m.def(
      "roi_pool_backward(Tensor grad_output, Tensor rois, Tensor argmax, "
      "int[] input_shape, int pooled_height, int pooled_width, "
      "float spatial_scale) -> Tensor");
}
