// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"

void roi_align_forward_cpu(Tensor input, Tensor rois, Tensor output,
                           Tensor argmax_y, Tensor argmax_x, int aligned_height,
                           int aligned_width, float spatial_scale,
                           int sampling_ratio, int pool_mode, bool aligned);
void roi_align_backward_cpu(Tensor grad_output, Tensor rois, Tensor argmax_y,
                            Tensor argmax_x, Tensor grad_input,
                            int aligned_height, int aligned_width,
                            float spatial_scale, int sampling_ratio,
                            int pool_mode, bool aligned);
#ifdef MMCV_WITH_CUDA
void ROIAlignForwardCUDAKernelLauncher(Tensor input, Tensor rois, Tensor output,
                                       Tensor argmax_y, Tensor argmax_x,
                                       int aligned_height, int aligned_width,
                                       float spatial_scale, int sampling_ratio,
                                       int pool_mode, bool aligned);
void ROIAlignBackwardCUDAKernelLauncher(Tensor grad_output, Tensor rois,
                                        Tensor argmax_y, Tensor argmax_x,
                                        Tensor grad_input, int aligned_height,
                                        int aligned_width, float spatial_scale,
                                        int sampling_ratio, int pool_mode,
                                        bool aligned);
#endif

// pool_mode: 0 = max, 1 = avg. argmax_y / argmax_x are only filled (and only
// needed by the backward) in max mode; in avg mode they are empty.
std::tuple<Tensor, Tensor, Tensor> roi_align_forward(
    const Tensor &input_, const Tensor &rois_, int64_t aligned_height,
    int64_t aligned_width, double spatial_scale, int64_t sampling_ratio,
    int64_t pool_mode, bool aligned) {
  TORCH_CHECK(rois_.dim() == 2 && rois_.size(1) == 5,
              "roi_align: rois must be (idx, x1, y1, x2, y2)");
  at::DeviceGuard guard(input_.device());
  const Tensor input = input_.contiguous(), rois = rois_.contiguous();
  Tensor output = at::zeros(
      {rois.size(0), input.size(1), aligned_height, aligned_width},
      input.options());
  Tensor argmax_y = pool_mode == 0 ? at::zeros_like(output)
                                   : at::zeros({0}, input.options());
  Tensor argmax_x = pool_mode == 0 ? at::zeros_like(output)
                                   : at::zeros({0}, input.options());
  if (input.is_cuda()) {
    MMCV_CUDA_ONLY(ROIAlignForwardCUDAKernelLauncher(
                       input, rois, output, argmax_y, argmax_x, aligned_height,
                       aligned_width, spatial_scale, sampling_ratio, pool_mode,
                       aligned);
                   return {output, argmax_y, argmax_x};);
  }
  roi_align_forward_cpu(input, rois, output, argmax_y, argmax_x,
                        aligned_height, aligned_width, spatial_scale,
                        sampling_ratio, pool_mode, aligned);
  return {output, argmax_y, argmax_x};
}

Tensor roi_align_backward(const Tensor &grad_output_, const Tensor &rois_,
                          const Tensor &argmax_y, const Tensor &argmax_x,
                          at::IntArrayRef input_shape, int64_t aligned_height,
                          int64_t aligned_width, double spatial_scale,
                          int64_t sampling_ratio, int64_t pool_mode,
                          bool aligned) {
  at::DeviceGuard guard(grad_output_.device());
  const Tensor grad_output = grad_output_.contiguous(),
               rois = rois_.contiguous();
  Tensor grad_input = at::zeros(input_shape, grad_output.options());
  if (grad_output.is_cuda()) {
    MMCV_CUDA_ONLY(ROIAlignBackwardCUDAKernelLauncher(
                       grad_output, rois, argmax_y.contiguous(),
                       argmax_x.contiguous(), grad_input, aligned_height,
                       aligned_width, spatial_scale, sampling_ratio, pool_mode,
                       aligned);
                   return grad_input;);
  }
  roi_align_backward_cpu(grad_output, rois, argmax_y.contiguous(),
                         argmax_x.contiguous(), grad_input, aligned_height,
                         aligned_width, spatial_scale, sampling_ratio,
                         pool_mode, aligned);
  return grad_input;
}

TORCH_LIBRARY_FRAGMENT(mmcv, m) {
  m.def(
      "roi_align_forward(Tensor input, Tensor rois, int aligned_height, "
      "int aligned_width, float spatial_scale, int sampling_ratio, "
      "int pool_mode, bool aligned) -> (Tensor, Tensor, Tensor)");
  m.def(
      "roi_align_backward(Tensor grad_output, Tensor rois, Tensor argmax_y, "
      "Tensor argmax_x, int[] input_shape, int aligned_height, "
      "int aligned_width, float spatial_scale, int sampling_ratio, "
      "int pool_mode, bool aligned) -> Tensor");
}

TORCH_LIBRARY_IMPL(mmcv, CPU, m) {
  m.impl("roi_align_forward", TORCH_FN(roi_align_forward));
  m.impl("roi_align_backward", TORCH_FN(roi_align_backward));
}

#ifdef MMCV_WITH_CUDA
TORCH_LIBRARY_IMPL(mmcv, CUDA, m) {
  m.impl("roi_align_forward", TORCH_FN(roi_align_forward));
  m.impl("roi_align_backward", TORCH_FN(roi_align_backward));
}
#endif
