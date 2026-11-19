// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"

void bbox_overlaps_cpu(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                       const int mode, const bool aligned, const int offset);
#ifdef MMCV_WITH_CUDA
void BBoxOverlapsCUDAKernelLauncher(const Tensor bboxes1, const Tensor bboxes2,
                                    Tensor ious, const int mode,
                                    const bool aligned, const int offset);
#endif

Tensor bbox_overlaps(const Tensor &bboxes1_, const Tensor &bboxes2_,
                     int64_t mode, bool aligned, int64_t offset) {
  const int64_t rows = bboxes1_.size(0), cols = bboxes2_.size(0);
  TORCH_CHECK(!aligned || rows == cols,
              "bbox_overlaps: aligned mode needs as many bboxes1 as bboxes2");
  Tensor ious = aligned ? at::zeros({rows}, bboxes1_.options())
                        : at::zeros({rows, cols}, bboxes1_.options());
  if (rows * cols == 0) return ious;

  at::DeviceGuard guard(bboxes1_.device());
  const Tensor bboxes1 = bboxes1_.contiguous(), bboxes2 = bboxes2_.contiguous();
  if (bboxes1.is_cuda()) {
    MMCV_CUDA_ONLY(BBoxOverlapsCUDAKernelLauncher(bboxes1, bboxes2, ious, mode,
                                                  aligned, offset);
                   return ious;);
  }
  bbox_overlaps_cpu(bboxes1, bboxes2, ious, mode, aligned, offset);
  return ious;
}

TORCH_LIBRARY_FRAGMENT(mmcv, m) {
  m.def(
      "bbox_overlaps(Tensor bboxes1, Tensor bboxes2, int mode, bool aligned, "
      "int offset) -> Tensor");
}

TORCH_LIBRARY_IMPL(mmcv, CPU, m) {
  m.impl("bbox_overlaps", TORCH_FN(bbox_overlaps));
}

#ifdef MMCV_WITH_CUDA
TORCH_LIBRARY_IMPL(mmcv, CUDA, m) {
  m.impl("bbox_overlaps", TORCH_FN(bbox_overlaps));
}
#endif
