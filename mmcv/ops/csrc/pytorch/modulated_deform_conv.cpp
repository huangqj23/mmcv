// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"

// im2col / col2im kernels: cpu/modulated_deform_conv.cpp and
// cuda/modulated_deform_conv_cuda.cu
#define MODULATED_IM2COL_ARGS(first, last)                                  \
  const Tensor first, const Tensor data_offset, const Tensor data_mask,     \
      const int batch_size, const int channels, const int height_im,        \
      const int width_im, const int height_col, const int width_col,        \
      const int kernel_h, const int kernel_w, const int pad_h,              \
      const int pad_w, const int stride_h, const int stride_w,              \
      const int dilation_h, const int dilation_w, const int deformable_group, \
      Tensor last
#define MODULATED_COL2IM_COORD_ARGS                                           \
  const Tensor data_col, const Tensor data_im, const Tensor data_offset,      \
      const Tensor data_mask, const int batch_size, const int channels,       \
      const int height_im, const int width_im, const int height_col,          \
      const int width_col, const int kernel_h, const int kernel_w,            \
      const int pad_h, const int pad_w, const int stride_h,                   \
      const int stride_w, const int dilation_h, const int dilation_w,         \
      const int deformable_group, Tensor grad_offset, Tensor grad_mask

void modulated_deformable_im2col_cpu(MODULATED_IM2COL_ARGS(data_im, data_col));
void modulated_deformable_col2im_cpu(MODULATED_IM2COL_ARGS(data_col, grad_im));
void modulated_deformable_col2im_coord_cpu(MODULATED_COL2IM_COORD_ARGS);
#ifdef MMCV_WITH_CUDA
void modulated_deformable_im2col_cuda(MODULATED_IM2COL_ARGS(data_im, data_col));
void modulated_deformable_col2im_cuda(MODULATED_IM2COL_ARGS(data_col, grad_im));
void modulated_deformable_col2im_coord_cuda(MODULATED_COL2IM_COORD_ARGS);
#endif

static void modulated_deformable_im2col_impl(
    MODULATED_IM2COL_ARGS(data_im, data_col)) {
  if (data_im.is_cuda()) {
    MMCV_CUDA_ONLY(modulated_deformable_im2col_cuda(
                       data_im, data_offset, data_mask, batch_size, channels,
                       height_im, width_im, height_col, width_col, kernel_h,
                       kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
                       dilation_w, deformable_group, data_col);
                   return;);
  }
  modulated_deformable_im2col_cpu(
      data_im, data_offset, data_mask, batch_size, channels, height_im,
      width_im, height_col, width_col, kernel_h, kernel_w, pad_h, pad_w,
      stride_h, stride_w, dilation_h, dilation_w, deformable_group, data_col);
}

static void modulated_deformable_col2im_impl(
    MODULATED_IM2COL_ARGS(data_col, grad_im)) {
  if (data_col.is_cuda()) {
    MMCV_CUDA_ONLY(modulated_deformable_col2im_cuda(
                       data_col, data_offset, data_mask, batch_size, channels,
                       height_im, width_im, height_col, width_col, kernel_h,
                       kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
                       dilation_w, deformable_group, grad_im);
                   return;);
  }
  modulated_deformable_col2im_cpu(
      data_col, data_offset, data_mask, batch_size, channels, height_im,
      width_im, height_col, width_col, kernel_h, kernel_w, pad_h, pad_w,
      stride_h, stride_w, dilation_h, dilation_w, deformable_group, grad_im);
}

static void modulated_deformable_col2im_coord_impl(
    MODULATED_COL2IM_COORD_ARGS) {
  if (data_col.is_cuda()) {
    MMCV_CUDA_ONLY(modulated_deformable_col2im_coord_cuda(
                       data_col, data_im, data_offset, data_mask, batch_size,
                       channels, height_im, width_im, height_col, width_col,
                       kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
                       dilation_h, dilation_w, deformable_group, grad_offset,
                       grad_mask);
                   return;);
  }
  modulated_deformable_col2im_coord_cpu(
      data_col, data_im, data_offset, data_mask, batch_size, channels,
      height_im, width_im, height_col, width_col, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, dilation_h, dilation_w, deformable_group,
      grad_offset, grad_mask);
}

// The two *_fallthrough functions write into the output tensors they are
// given; the registered ops at the end of the file allocate those outputs.
static void modulated_deform_conv_forward_fallthrough(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor output, Tensor columns, int kernel_h, int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, const int group,
    const int deformable_group, const bool with_bias) {
  at::DeviceGuard guard(input.device());

  const int batch = input.size(0);
  const int channels = input.size(1);
  const int height = input.size(2);
  const int width = input.size(3);

  const int channels_out = weight.size(0);
  const int channels_kernel = weight.size(1);
  const int kernel_h_ = weight.size(2);
  const int kernel_w_ = weight.size(3);

  if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
    AT_ERROR("Input shape and kernel shape won't match: (%d x %d vs %d x %d).",
             kernel_h_, kernel_w, kernel_h_, kernel_w_);
  if (channels != channels_kernel * group)
    AT_ERROR("Input shape and kernel channels won't match: (%d vs %d).",
             channels, channels_kernel * group);

  const int height_out =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_out =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  if (ones.ndimension() != 2 ||
      ones.size(0) * ones.size(1) < height_out * width_out) {
    // Resize plane and fill with ones...
    ones = at::ones({height_out, width_out}, input.options());
  }

  // resize output
  output = output.view({batch, channels_out, height_out, width_out}).zero_();
  // resize temporary columns
  columns =
      at::zeros({channels * kernel_h * kernel_w, 1 * height_out * width_out},
                input.options());

  output = output.view({output.size(0), group, output.size(1) / group,
                        output.size(2), output.size(3)});

  for (int b = 0; b < batch; b++) {
    modulated_deformable_im2col_impl(
        input[b], offset[b], mask[b], 1, channels, height, width, height_out,
        width_out, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, deformable_group, columns);

    // divide into group
    weight = weight.view({group, weight.size(0) / group, weight.size(1),
                          weight.size(2), weight.size(3)});
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});

    for (int g = 0; g < group; g++) {
      output[b][g] = output[b][g]
                         .flatten(1)
                         .addmm_(weight[g].flatten(1), columns[g])
                         .view_as(output[b][g]);
    }

    weight = weight.view({weight.size(0) * weight.size(1), weight.size(2),
                          weight.size(3), weight.size(4)});
    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
  }

  output = output.view({output.size(0), output.size(1) * output.size(2),
                        output.size(3), output.size(4)});

  if (with_bias) {
    output += bias.view({1, bias.size(0), 1, 1});
  }
}

static void modulated_deform_conv_backward_fallthrough(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor columns, Tensor grad_input, Tensor grad_weight,
    Tensor grad_bias, Tensor grad_offset, Tensor grad_mask, Tensor grad_output,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int dilation_h, int dilation_w, int group, int deformable_group,
    const bool with_bias) {
  at::DeviceGuard guard(input.device());

  const int batch = input.size(0);
  const int channels = input.size(1);
  const int height = input.size(2);
  const int width = input.size(3);

  const int channels_kernel = weight.size(1);
  const int kernel_h_ = weight.size(2);
  const int kernel_w_ = weight.size(3);
  if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
    AT_ERROR("Input shape and kernel shape won't match: (%d x %d vs %d x %d).",
             kernel_h_, kernel_w, kernel_h_, kernel_w_);
  if (channels != channels_kernel * group)
    AT_ERROR("Input shape and kernel channels won't match: (%d vs %d).",
             channels, channels_kernel * group);

  const int height_out =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_out =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  if (ones.ndimension() != 2 ||
      ones.size(0) * ones.size(1) < height_out * width_out) {
    // Resize plane and fill with ones...
    ones = at::ones({height_out, width_out}, input.options());
  }

  grad_input = grad_input.view({batch, channels, height, width});
  columns = at::zeros({channels * kernel_h * kernel_w, height_out * width_out},
                      input.options());

  grad_output =
      grad_output.view({grad_output.size(0), group, grad_output.size(1) / group,
                        grad_output.size(2), grad_output.size(3)});

  for (int b = 0; b < batch; b++) {
    // divide int group
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    weight = weight.view({group, weight.size(0) / group, weight.size(1),
                          weight.size(2), weight.size(3)});

    for (int g = 0; g < group; g++) {
      columns[g].addmm_(weight[g].flatten(1).transpose(0, 1),
                        grad_output[b][g].flatten(1), 0.0f, 1.0f);
    }

    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    weight = weight.view({weight.size(0) * weight.size(1), weight.size(2),
                          weight.size(3), weight.size(4)});

    // gradient w.r.t. input coordinate data
    modulated_deformable_col2im_coord_impl(
        columns, input[b], offset[b], mask[b], 1, channels, height, width,
        height_out, width_out, kernel_h, kernel_w, pad_h, pad_w, stride_h,
        stride_w, dilation_h, dilation_w, deformable_group, grad_offset[b],
        grad_mask[b]);
    // gradient w.r.t. input data
    modulated_deformable_col2im_impl(
        columns, offset[b], mask[b], 1, channels, height, width, height_out,
        width_out, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, deformable_group, grad_input[b]);

    // gradient w.r.t. weight, dWeight should accumulate across the batch and
    // group
    modulated_deformable_im2col_impl(
        input[b], offset[b], mask[b], 1, channels, height, width, height_out,
        width_out, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, deformable_group, columns);

    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    grad_weight = grad_weight.view({group, grad_weight.size(0) / group,
                                    grad_weight.size(1), grad_weight.size(2),
                                    grad_weight.size(3)});
    if (with_bias)
      grad_bias = grad_bias.view({group, grad_bias.size(0) / group});

    for (int g = 0; g < group; g++) {
      grad_weight[g] =
          grad_weight[g]
              .flatten(1)
              .addmm_(grad_output[b][g].flatten(1), columns[g].transpose(0, 1))
              .view_as(grad_weight[g]);
      if (with_bias) {
        grad_bias[g] =
            grad_bias[g]
                .view({-1, 1})
                .addmm_(grad_output[b][g].flatten(1), ones.view({-1, 1}))
                .view(-1);
      }
    }

    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    grad_weight = grad_weight.view({grad_weight.size(0) * grad_weight.size(1),
                                    grad_weight.size(2), grad_weight.size(3),
                                    grad_weight.size(4)});
    if (with_bias)
      grad_bias = grad_bias.view({grad_bias.size(0) * grad_bias.size(1)});
  }
  grad_output = grad_output.view({grad_output.size(0) * grad_output.size(1),
                                  grad_output.size(2), grad_output.size(3),
                                  grad_output.size(4)});
}

Tensor modulated_deform_conv_forward(
    const Tensor &input_, const Tensor &weight_,
    const std::optional<Tensor> &bias_, const Tensor &offset_,
    const Tensor &mask_, int64_t kernel_h, int64_t kernel_w, int64_t stride_h,
    int64_t stride_w, int64_t pad_h, int64_t pad_w, int64_t dilation_h,
    int64_t dilation_w, int64_t group, int64_t deformable_group) {
  TORCH_CHECK(input_.dim() == 4, "modulated_deform_conv: expected a 4D input, ",
              "got ", input_.dim(), "D");
  at::DeviceGuard guard(input_.device());
  const Tensor input = input_.contiguous(), weight = weight_.contiguous(),
               offset = offset_.contiguous(), mask = mask_.contiguous();
  const bool with_bias = bias_.has_value();
  const Tensor bias =
      with_bias ? bias_->contiguous() : at::empty({0}, input.options());
  const int64_t out_h =
      (input.size(2) + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
          stride_h +
      1;
  const int64_t out_w =
      (input.size(3) + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
          stride_w +
      1;
  TORCH_CHECK(out_h > 0 && out_w > 0,
              "modulated_deform_conv: the input is too small for the kernel");
  Tensor output =
      at::empty({input.size(0), weight.size(0), out_h, out_w}, input.options());
  modulated_deform_conv_forward_fallthrough(
      input, weight, bias, at::empty({0}, input.options()), offset, mask,
      output, at::empty({0}, input.options()), kernel_h, kernel_w, stride_h,
      stride_w, pad_h, pad_w, dilation_h, dilation_w, group, deformable_group,
      with_bias);
  return output;
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
modulated_deform_conv_backward(
    const Tensor &input_, const Tensor &weight_,
    const std::optional<Tensor> &bias_, const Tensor &offset_,
    const Tensor &mask_, const Tensor &grad_output_, int64_t kernel_h,
    int64_t kernel_w, int64_t stride_h, int64_t stride_w, int64_t pad_h,
    int64_t pad_w, int64_t dilation_h, int64_t dilation_w, int64_t group,
    int64_t deformable_group) {
  at::DeviceGuard guard(input_.device());
  const Tensor input = input_.contiguous(), weight = weight_.contiguous(),
               offset = offset_.contiguous(), mask = mask_.contiguous(),
               grad_output = grad_output_.contiguous();
  const bool with_bias = bias_.has_value();
  const Tensor bias =
      with_bias ? bias_->contiguous() : at::empty({0}, input.options());
  Tensor grad_input = at::zeros_like(input);
  Tensor grad_weight = at::zeros_like(weight);
  Tensor grad_bias = at::zeros_like(bias);  // stays empty without a bias
  Tensor grad_offset = at::zeros_like(offset);
  Tensor grad_mask = at::zeros_like(mask);
  modulated_deform_conv_backward_fallthrough(
      input, weight, bias, at::empty({0}, input.options()), offset, mask,
      at::empty({0}, input.options()), grad_input, grad_weight, grad_bias,
      grad_offset, grad_mask, grad_output, kernel_h, kernel_w, stride_h,
      stride_w, pad_h, pad_w, dilation_h, dilation_w, group, deformable_group,
      with_bias);
  return {grad_input, grad_weight, grad_bias, grad_offset, grad_mask};
}

TORCH_LIBRARY_FRAGMENT(mmcv, m) {
  m.def(
      "modulated_deform_conv_forward(Tensor input, Tensor weight, "
      "Tensor? bias, Tensor offset, Tensor mask, int kernel_h, int kernel_w, "
      "int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, "
      "int dilation_w, int group, int deformable_group) -> Tensor");
  m.def(
      "modulated_deform_conv_backward(Tensor input, Tensor weight, "
      "Tensor? bias, Tensor offset, Tensor mask, Tensor grad_output, "
      "int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, "
      "int pad_w, int dilation_h, int dilation_w, int group, "
      "int deformable_group) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(mmcv, CPU, m) {
  m.impl("modulated_deform_conv_forward",
         TORCH_FN(modulated_deform_conv_forward));
  m.impl("modulated_deform_conv_backward",
         TORCH_FN(modulated_deform_conv_backward));
}

#ifdef MMCV_WITH_CUDA
TORCH_LIBRARY_IMPL(mmcv, CUDA, m) {
  m.impl("modulated_deform_conv_forward",
         TORCH_FN(modulated_deform_conv_forward));
  m.impl("modulated_deform_conv_backward",
         TORCH_FN(modulated_deform_conv_backward));
}
#endif
