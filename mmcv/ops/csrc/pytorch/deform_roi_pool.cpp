// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void DeformRoIPoolForwardCUDAKernelLauncher(Tensor input, Tensor rois,
                                            Tensor offset, Tensor output,
                                            int pooled_height, int pooled_width,
                                            float spatial_scale,
                                            int sampling_ratio, float gamma);

void DeformRoIPoolBackwardCUDAKernelLauncher(
    Tensor grad_output, Tensor input, Tensor rois, Tensor offset,
    Tensor grad_input, Tensor grad_offset, int pooled_height, int pooled_width,
    float spatial_scale, int sampling_ratio, float gamma);

// Without an offset the kernels get an empty tensor (a null pointer).
static Tensor offset_or_empty(const std::optional<Tensor> &offset,
                              const Tensor &input) {
  return offset.has_value() ? offset->contiguous()
                            : at::zeros({0}, input.options());
}

Tensor deform_roi_pool_forward(const Tensor &input_, const Tensor &rois_,
                               const std::optional<Tensor> &offset_,
                               int64_t pooled_height, int64_t pooled_width,
                               double spatial_scale, int64_t sampling_ratio,
                               double gamma) {
  TORCH_CHECK(rois_.dim() == 2 && rois_.size(1) == 5,
              "deform_roi_pool: rois must be (idx, x1, y1, x2, y2)");
  at::DeviceGuard guard(input_.device());
  const Tensor input = input_.contiguous(), rois = rois_.contiguous(),
               offset = offset_or_empty(offset_, input);
  Tensor output = at::zeros(
      {rois.size(0), input.size(1), pooled_height, pooled_width},
      input.options());
  DeformRoIPoolForwardCUDAKernelLauncher(input, rois, offset, output,
                                         pooled_height, pooled_width,
                                         spatial_scale, sampling_ratio, gamma);
  return output;
}

// grad_offset is empty when there is no offset
std::tuple<Tensor, Tensor> deform_roi_pool_backward(
    const Tensor &grad_output_, const Tensor &input_, const Tensor &rois_,
    const std::optional<Tensor> &offset_, int64_t pooled_height,
    int64_t pooled_width, double spatial_scale, int64_t sampling_ratio,
    double gamma) {
  at::DeviceGuard guard(input_.device());
  const Tensor grad_output = grad_output_.contiguous(),
               input = input_.contiguous(), rois = rois_.contiguous(),
               offset = offset_or_empty(offset_, input);
  Tensor grad_input = at::zeros_like(input);
  Tensor grad_offset = at::zeros_like(offset);
  DeformRoIPoolBackwardCUDAKernelLauncher(
      grad_output, input, rois, offset, grad_input, grad_offset, pooled_height,
      pooled_width, spatial_scale, sampling_ratio, gamma);
  return {grad_input, grad_offset};
}

TORCH_LIBRARY_IMPL(mmcv, CUDA, m) {
  m.impl("deform_roi_pool_forward", TORCH_FN(deform_roi_pool_forward));
  m.impl("deform_roi_pool_backward", TORCH_FN(deform_roi_pool_backward));
}
#endif

TORCH_LIBRARY_FRAGMENT(mmcv, m) {
  m.def(
      "deform_roi_pool_forward(Tensor input, Tensor rois, Tensor? offset, "
      "int pooled_height, int pooled_width, float spatial_scale, "
      "int sampling_ratio, float gamma) -> Tensor");
  m.def(
      "deform_roi_pool_backward(Tensor grad_output, Tensor input, Tensor rois, "
      "Tensor? offset, int pooled_height, int pooled_width, "
      "float spatial_scale, int sampling_ratio, float gamma) "
      "-> (Tensor, Tensor)");
}
