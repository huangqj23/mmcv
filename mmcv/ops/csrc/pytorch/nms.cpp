// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"

Tensor nms_cpu(Tensor boxes, Tensor scores, float iou_threshold, int offset);
Tensor softnms_cpu(Tensor boxes, Tensor scores, Tensor dets,
                   float iou_threshold, float sigma, float min_score,
                   int method, int offset);
std::vector<std::vector<int> > nms_match_cpu(Tensor dets, float iou_threshold);
#ifdef MMCV_WITH_CUDA
Tensor NMSCUDAKernelLauncher(Tensor boxes, Tensor scores, float iou_threshold,
                             int offset);
#endif

Tensor nms(const Tensor &boxes_, const Tensor &scores_, double iou_threshold,
           int64_t offset) {
  at::DeviceGuard guard(boxes_.device());
  const Tensor boxes = boxes_.contiguous(), scores = scores_.contiguous();
  if (boxes.is_cuda()) {
    MMCV_CUDA_ONLY(
        return NMSCUDAKernelLauncher(boxes, scores, iou_threshold, offset););
  }
  return nms_cpu(boxes, scores, iou_threshold, offset);
}

// CPU only. dets gets the boxes with their decayed scores; its first
// inds.size(0) rows are the kept boxes.
std::tuple<Tensor, Tensor> softnms(const Tensor &boxes_, const Tensor &scores_,
                                   double iou_threshold, double sigma,
                                   double min_score, int64_t method,
                                   int64_t offset) {
  const Tensor boxes = boxes_.contiguous(), scores = scores_.contiguous();
  Tensor dets = at::empty({boxes.size(0), 5}, boxes.options());
  Tensor inds = softnms_cpu(boxes, scores, dets, iou_threshold, sigma,
                            min_score, method, offset);
  return {dets, inds};
}

// CPU only. One int64 index tensor per matched group, in score order.
std::vector<Tensor> nms_match(const Tensor &dets, double iou_threshold) {
  const auto groups = nms_match_cpu(dets.contiguous(), iou_threshold);
  std::vector<Tensor> out;
  out.reserve(groups.size());
  for (const auto &group : groups) {
    out.push_back(at::tensor(at::ArrayRef<int>(group)).to(at::kLong));
  }
  return out;
}

TORCH_LIBRARY_FRAGMENT(mmcv, m) {
  m.def(
      "nms(Tensor boxes, Tensor scores, float iou_threshold, int offset) "
      "-> Tensor");
  m.def(
      "softnms(Tensor boxes, Tensor scores, float iou_threshold, float sigma, "
      "float min_score, int method, int offset) -> (Tensor, Tensor)");
  m.def("nms_match(Tensor dets, float iou_threshold) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(mmcv, CPU, m) {
  m.impl("nms", TORCH_FN(nms));
  m.impl("softnms", TORCH_FN(softnms));
  m.impl("nms_match", TORCH_FN(nms_match));
}

#ifdef MMCV_WITH_CUDA
TORCH_LIBRARY_IMPL(mmcv, CUDA, m) { m.impl("nms", TORCH_FN(nms)); }
#endif
