// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void MaskedIm2colForwardCUDAKernelLauncher(const Tensor bottom_data,
                                           const Tensor mask_h_idx,
                                           const Tensor mask_w_idx,
                                           Tensor top_data, const int kernel_h,
                                           const int kernel_w, const int pad_h,
                                           const int pad_w);

void MaskedCol2imForwardCUDAKernelLauncher(const Tensor bottom_data,
                                           const Tensor mask_h_idx,
                                           const Tensor mask_w_idx,
                                           Tensor top_data, const int height,
                                           const int width, const int channels);

// im: (1, C, H, W) -> col: (C * kernel_h * kernel_w, number of masked pixels)
Tensor masked_im2col_forward(const Tensor &im_, const Tensor &mask_h_idx_,
                             const Tensor &mask_w_idx_, int64_t kernel_h,
                             int64_t kernel_w, int64_t pad_h, int64_t pad_w) {
  at::DeviceGuard guard(im_.device());
  const Tensor im = im_.contiguous(), mask_h_idx = mask_h_idx_.contiguous(),
               mask_w_idx = mask_w_idx_.contiguous();
  Tensor col = at::zeros(
      {im.size(1) * kernel_h * kernel_w, mask_h_idx.size(0)}, im.options());
  MaskedIm2colForwardCUDAKernelLauncher(im, mask_h_idx, mask_w_idx, col,
                                        kernel_h, kernel_w, pad_h, pad_w);
  return col;
}

// col: (channels, number of masked pixels) -> im: (1, channels, height, width)
Tensor masked_col2im_forward(const Tensor &col_, const Tensor &mask_h_idx_,
                             const Tensor &mask_w_idx_, int64_t height,
                             int64_t width, int64_t channels) {
  at::DeviceGuard guard(col_.device());
  const Tensor col = col_.contiguous(), mask_h_idx = mask_h_idx_.contiguous(),
               mask_w_idx = mask_w_idx_.contiguous();
  Tensor im = at::zeros({1, channels, height, width}, col.options());
  MaskedCol2imForwardCUDAKernelLauncher(col, mask_h_idx, mask_w_idx, im, height,
                                        width, channels);
  return im;
}

TORCH_LIBRARY_IMPL(mmcv, CUDA, m) {
  m.impl("masked_im2col_forward", TORCH_FN(masked_im2col_forward));
  m.impl("masked_col2im_forward", TORCH_FN(masked_col2im_forward));
}
#endif

TORCH_LIBRARY_FRAGMENT(mmcv, m) {
  m.def(
      "masked_im2col_forward(Tensor im, Tensor mask_h_idx, Tensor mask_w_idx, "
      "int kernel_h, int kernel_w, int pad_h, int pad_w) -> Tensor");
  m.def(
      "masked_col2im_forward(Tensor col, Tensor mask_h_idx, Tensor mask_w_idx, "
      "int height, int width, int channels) -> Tensor");
}
