/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from
*https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
Tensor ms_deform_attn_cuda_forward(const Tensor &value,
                                   const Tensor &spatial_shapes,
                                   const Tensor &level_start_index,
                                   const Tensor &sampling_loc,
                                   const Tensor &attn_weight,
                                   const int im2col_step);

void ms_deform_attn_cuda_backward(
    const Tensor &value, const Tensor &spatial_shapes,
    const Tensor &level_start_index, const Tensor &sampling_loc,
    const Tensor &attn_weight, const Tensor &grad_output, Tensor &grad_value,
    Tensor &grad_sampling_loc, Tensor &grad_attn_weight, const int im2col_step);

Tensor ms_deform_attn_forward(const Tensor &value, const Tensor &spatial_shapes,
                              const Tensor &level_start_index,
                              const Tensor &sampling_loc,
                              const Tensor &attn_weight, int64_t im2col_step) {
  at::DeviceGuard guard(value.device());
  return ms_deform_attn_cuda_forward(
      value.contiguous(), spatial_shapes.contiguous(),
      level_start_index.contiguous(), sampling_loc.contiguous(),
      attn_weight.contiguous(), im2col_step);
}

// gradients of value, sampling_locations and attention_weights
std::tuple<Tensor, Tensor, Tensor> ms_deform_attn_backward(
    const Tensor &value_, const Tensor &spatial_shapes_,
    const Tensor &level_start_index_, const Tensor &sampling_loc_,
    const Tensor &attn_weight_, const Tensor &grad_output_,
    int64_t im2col_step) {
  at::DeviceGuard guard(value_.device());
  const Tensor value = value_.contiguous(),
               spatial_shapes = spatial_shapes_.contiguous(),
               level_start_index = level_start_index_.contiguous(),
               sampling_loc = sampling_loc_.contiguous(),
               attn_weight = attn_weight_.contiguous(),
               grad_output = grad_output_.contiguous();
  Tensor grad_value = at::zeros_like(value);
  Tensor grad_sampling_loc = at::zeros_like(sampling_loc);
  Tensor grad_attn_weight = at::zeros_like(attn_weight);
  ms_deform_attn_cuda_backward(value, spatial_shapes, level_start_index,
                               sampling_loc, attn_weight, grad_output,
                               grad_value, grad_sampling_loc, grad_attn_weight,
                               im2col_step);
  return {grad_value, grad_sampling_loc, grad_attn_weight};
}

TORCH_LIBRARY_IMPL(mmcv, CUDA, m) {
  m.impl("ms_deform_attn_forward", TORCH_FN(ms_deform_attn_forward));
  m.impl("ms_deform_attn_backward", TORCH_FN(ms_deform_attn_backward));
}
#endif

TORCH_LIBRARY_FRAGMENT(mmcv, m) {
  m.def(
      "ms_deform_attn_forward(Tensor value, Tensor value_spatial_shapes, "
      "Tensor value_level_start_index, Tensor sampling_locations, "
      "Tensor attention_weights, int im2col_step) -> Tensor");
  m.def(
      "ms_deform_attn_backward(Tensor value, Tensor value_spatial_shapes, "
      "Tensor value_level_start_index, Tensor sampling_locations, "
      "Tensor attention_weights, Tensor grad_output, int im2col_step) "
      "-> (Tensor, Tensor, Tensor)");
}
