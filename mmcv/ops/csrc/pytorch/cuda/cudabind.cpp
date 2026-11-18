#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void BBoxOverlapsCUDAKernelLauncher(const Tensor bboxes1, const Tensor bboxes2,
                                    Tensor ious, const int mode,
                                    const bool aligned, const int offset);

void bbox_overlaps_cuda(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                        const int mode, const bool aligned, const int offset) {
  BBoxOverlapsCUDAKernelLauncher(bboxes1, bboxes2, ious, mode, aligned, offset);
}

void bbox_overlaps_impl(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                        const int mode, const bool aligned, const int offset);
REGISTER_DEVICE_IMPL(bbox_overlaps_impl, CUDA, bbox_overlaps_cuda);

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

void carafe_forward_cuda(Tensor features, Tensor masks, Tensor rfeatures,
                         Tensor routput, Tensor rmasks, Tensor output,
                         int kernel_size, int group_size, int scale_factor) {
  CARAFEForwardCUDAKernelLauncher(features, masks, rfeatures, routput, rmasks,
                                  output, kernel_size, group_size,
                                  scale_factor);
}

void carafe_backward_cuda(Tensor top_grad, Tensor rfeatures, Tensor masks,
                          Tensor rtop_grad, Tensor rbottom_grad_hs,
                          Tensor rbottom_grad, Tensor rmask_grad,
                          Tensor bottom_grad, Tensor mask_grad, int kernel_size,
                          int group_size, int scale_factor) {
  CARAFEBackwardCUDAKernelLauncher(top_grad, rfeatures, masks, rtop_grad,
                                   rbottom_grad_hs, rbottom_grad, rmask_grad,
                                   bottom_grad, mask_grad, kernel_size,
                                   group_size, scale_factor);
}

void carafe_forward_impl(Tensor features, Tensor masks, Tensor rfeatures,
                         Tensor routput, Tensor rmasks, Tensor output,
                         int kernel_size, int group_size, int scale_factor);

void carafe_backward_impl(Tensor top_grad, Tensor rfeatures, Tensor masks,
                          Tensor rtop_grad, Tensor rbottom_grad_hs,
                          Tensor rbottom_grad, Tensor rmask_grad,
                          Tensor bottom_grad, Tensor mask_grad, int kernel_size,
                          int group_size, int scale_factor);

REGISTER_DEVICE_IMPL(carafe_forward_impl, CUDA, carafe_forward_cuda);
REGISTER_DEVICE_IMPL(carafe_backward_impl, CUDA, carafe_backward_cuda);

void CARAFENAIVEForwardCUDAKernelLauncher(const Tensor features,
                                          const Tensor masks, Tensor output,
                                          const int kernel_size,
                                          const int group_size,
                                          const int scale_factor);

void CARAFENAIVEBackwardCUDAKernelLauncher(
    const Tensor top_grad, const Tensor features, const Tensor masks,
    Tensor bottom_grad, Tensor mask_grad, const int kernel_size,
    const int group_size, const int scale_factor);

void carafe_naive_forward_cuda(Tensor features, Tensor masks, Tensor output,
                               int kernel_size, int group_size,
                               int scale_factor) {
  CARAFENAIVEForwardCUDAKernelLauncher(features, masks, output, kernel_size,
                                       group_size, scale_factor);
}

void carafe_naive_backward_cuda(Tensor top_grad, Tensor features, Tensor masks,
                                Tensor bottom_grad, Tensor mask_grad,
                                int kernel_size, int group_size,
                                int scale_factor) {
  CARAFENAIVEBackwardCUDAKernelLauncher(top_grad, features, masks, bottom_grad,
                                        mask_grad, kernel_size, group_size,
                                        scale_factor);
}
void carafe_naive_forward_impl(Tensor features, Tensor masks, Tensor output,
                               int kernel_size, int group_size,
                               int scale_factor);

void carafe_naive_backward_impl(Tensor top_grad, Tensor features, Tensor masks,
                                Tensor bottom_grad, Tensor mask_grad,
                                int kernel_size, int group_size,
                                int scale_factor);

REGISTER_DEVICE_IMPL(carafe_naive_forward_impl, CUDA,
                     carafe_naive_forward_cuda);
REGISTER_DEVICE_IMPL(carafe_naive_backward_impl, CUDA,
                     carafe_naive_backward_cuda);

void deformable_im2col_cuda(Tensor data_im, Tensor data_offset,
                            const int channels, const int height,
                            const int width, const int ksize_h,
                            const int ksize_w, const int pad_h, const int pad_w,
                            const int stride_h, const int stride_w,
                            const int dilation_h, const int dilation_w,
                            const int parallel_imgs, const int deformable_group,
                            Tensor data_col);

void deformable_col2im_cuda(Tensor data_col, Tensor data_offset,
                            const int channels, const int height,
                            const int width, const int ksize_h,
                            const int ksize_w, const int pad_h, const int pad_w,
                            const int stride_h, const int stride_w,
                            const int dilation_h, const int dilation_w,
                            const int parallel_imgs, const int deformable_group,
                            Tensor grad_im);

void deformable_col2im_coord_cuda(
    Tensor data_col, Tensor data_im, Tensor data_offset, const int channels,
    const int height, const int width, const int ksize_h, const int ksize_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int parallel_imgs,
    const int deformable_group, Tensor grad_offset);

void deformable_im2col_impl(Tensor data_im, Tensor data_offset,
                            const int channels, const int height,
                            const int width, const int ksize_h,
                            const int ksize_w, const int pad_h, const int pad_w,
                            const int stride_h, const int stride_w,
                            const int dilation_h, const int dilation_w,
                            const int parallel_imgs, const int deformable_group,
                            Tensor data_col);

void deformable_col2im_impl(Tensor data_col, Tensor data_offset,
                            const int channels, const int height,
                            const int width, const int ksize_h,
                            const int ksize_w, const int pad_h, const int pad_w,
                            const int stride_h, const int stride_w,
                            const int dilation_h, const int dilation_w,
                            const int parallel_imgs, const int deformable_group,
                            Tensor grad_im);

void deformable_col2im_coord_impl(
    Tensor data_col, Tensor data_im, Tensor data_offset, const int channels,
    const int height, const int width, const int ksize_h, const int ksize_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int parallel_imgs,
    const int deformable_group, Tensor grad_offset);

REGISTER_DEVICE_IMPL(deformable_im2col_impl, CUDA, deformable_im2col_cuda);
REGISTER_DEVICE_IMPL(deformable_col2im_impl, CUDA, deformable_col2im_cuda);
REGISTER_DEVICE_IMPL(deformable_col2im_coord_impl, CUDA,
                     deformable_col2im_coord_cuda);

void DeformRoIPoolForwardCUDAKernelLauncher(Tensor input, Tensor rois,
                                            Tensor offset, Tensor output,
                                            int pooled_height, int pooled_width,
                                            float spatial_scale,
                                            int sampling_ratio, float gamma);

void DeformRoIPoolBackwardCUDAKernelLauncher(
    Tensor grad_output, Tensor input, Tensor rois, Tensor offset,
    Tensor grad_input, Tensor grad_offset, int pooled_height, int pooled_width,
    float spatial_scale, int sampling_ratio, float gamma);

void deform_roi_pool_forward_cuda(Tensor input, Tensor rois, Tensor offset,
                                  Tensor output, int pooled_height,
                                  int pooled_width, float spatial_scale,
                                  int sampling_ratio, float gamma) {
  DeformRoIPoolForwardCUDAKernelLauncher(input, rois, offset, output,
                                         pooled_height, pooled_width,
                                         spatial_scale, sampling_ratio, gamma);
}

void deform_roi_pool_backward_cuda(Tensor grad_output, Tensor input,
                                   Tensor rois, Tensor offset,
                                   Tensor grad_input, Tensor grad_offset,
                                   int pooled_height, int pooled_width,
                                   float spatial_scale, int sampling_ratio,
                                   float gamma) {
  DeformRoIPoolBackwardCUDAKernelLauncher(
      grad_output, input, rois, offset, grad_input, grad_offset, pooled_height,
      pooled_width, spatial_scale, sampling_ratio, gamma);
}

void deform_roi_pool_forward_impl(Tensor input, Tensor rois, Tensor offset,
                                  Tensor output, int pooled_height,
                                  int pooled_width, float spatial_scale,
                                  int sampling_ratio, float gamma);

void deform_roi_pool_backward_impl(Tensor grad_output, Tensor input,
                                   Tensor rois, Tensor offset,
                                   Tensor grad_input, Tensor grad_offset,
                                   int pooled_height, int pooled_width,
                                   float spatial_scale, int sampling_ratio,
                                   float gamma);

REGISTER_DEVICE_IMPL(deform_roi_pool_forward_impl, CUDA,
                     deform_roi_pool_forward_cuda);
REGISTER_DEVICE_IMPL(deform_roi_pool_backward_impl, CUDA,
                     deform_roi_pool_backward_cuda);

void SigmoidFocalLossForwardCUDAKernelLauncher(Tensor input, Tensor target,
                                               Tensor weight, Tensor output,
                                               const float gamma,
                                               const float alpha);

void SigmoidFocalLossBackwardCUDAKernelLauncher(Tensor input, Tensor target,
                                                Tensor weight,
                                                Tensor grad_input,
                                                const float gamma,
                                                const float alpha);

void SoftmaxFocalLossForwardCUDAKernelLauncher(Tensor softmax, Tensor target,
                                               Tensor weight, Tensor output,
                                               const float gamma,
                                               const float alpha);

void SoftmaxFocalLossBackwardCUDAKernelLauncher(Tensor softmax, Tensor target,
                                                Tensor weight, Tensor buff,
                                                Tensor grad_input,
                                                const float gamma,
                                                const float alpha);

void sigmoid_focal_loss_forward_cuda(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha) {
  SigmoidFocalLossForwardCUDAKernelLauncher(input, target, weight, output,
                                            gamma, alpha);
}

void sigmoid_focal_loss_backward_cuda(Tensor input, Tensor target,
                                      Tensor weight, Tensor grad_input,
                                      float gamma, float alpha) {
  SigmoidFocalLossBackwardCUDAKernelLauncher(input, target, weight, grad_input,
                                             gamma, alpha);
}

void softmax_focal_loss_forward_cuda(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha) {
  SoftmaxFocalLossForwardCUDAKernelLauncher(input, target, weight, output,
                                            gamma, alpha);
}

void softmax_focal_loss_backward_cuda(Tensor input, Tensor target,
                                      Tensor weight, Tensor buff,
                                      Tensor grad_input, float gamma,
                                      float alpha) {
  SoftmaxFocalLossBackwardCUDAKernelLauncher(input, target, weight, buff,
                                             grad_input, gamma, alpha);
}

void sigmoid_focal_loss_forward_impl(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha);

void sigmoid_focal_loss_backward_impl(Tensor input, Tensor target,
                                      Tensor weight, Tensor grad_input,
                                      float gamma, float alpha);

void softmax_focal_loss_forward_impl(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha);

void softmax_focal_loss_backward_impl(Tensor input, Tensor target,
                                      Tensor weight, Tensor buff,
                                      Tensor grad_input, float gamma,
                                      float alpha);

REGISTER_DEVICE_IMPL(sigmoid_focal_loss_forward_impl, CUDA,
                     sigmoid_focal_loss_forward_cuda);
REGISTER_DEVICE_IMPL(sigmoid_focal_loss_backward_impl, CUDA,
                     sigmoid_focal_loss_backward_cuda);
REGISTER_DEVICE_IMPL(softmax_focal_loss_forward_impl, CUDA,
                     softmax_focal_loss_forward_cuda);
REGISTER_DEVICE_IMPL(softmax_focal_loss_backward_impl, CUDA,
                     softmax_focal_loss_backward_cuda);

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

void masked_im2col_forward_cuda(const Tensor im, const Tensor mask_h_idx,
                                const Tensor mask_w_idx, Tensor col,
                                const int kernel_h, const int kernel_w,
                                const int pad_h, const int pad_w) {
  // im: (n, ic, h, w), kernel size (kh, kw)
  // kernel: (oc, ic * kh * kw), col: (kh * kw * ic, ow * oh)
  MaskedIm2colForwardCUDAKernelLauncher(im, mask_h_idx, mask_w_idx, col,
                                        kernel_h, kernel_w, pad_h, pad_w);
}

void masked_col2im_forward_cuda(const Tensor col, const Tensor mask_h_idx,
                                const Tensor mask_w_idx, Tensor im, int height,
                                int width, int channels) {
  // im: (n, ic, h, w), kernel size (kh, kw)
  // kernel: (oc, ic * kh * kh), col: (kh * kw * ic, ow * oh)
  MaskedCol2imForwardCUDAKernelLauncher(col, mask_h_idx, mask_w_idx, im, height,
                                        width, channels);
}

void masked_im2col_forward_impl(const Tensor im, const Tensor mask_h_idx,
                                const Tensor mask_w_idx, Tensor col,
                                const int kernel_h, const int kernel_w,
                                const int pad_h, const int pad_w);

void masked_col2im_forward_impl(const Tensor col, const Tensor mask_h_idx,
                                const Tensor mask_w_idx, Tensor im, int height,
                                int width, int channels);

REGISTER_DEVICE_IMPL(masked_im2col_forward_impl, CUDA,
                     masked_im2col_forward_cuda);
REGISTER_DEVICE_IMPL(masked_col2im_forward_impl, CUDA,
                     masked_col2im_forward_cuda);

void modulated_deformable_im2col_cuda(
    const Tensor data_im, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor data_col);

void modulated_deformable_col2im_cuda(
    const Tensor data_col, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im);

void modulated_deformable_col2im_coord_cuda(
    const Tensor data_col, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deformable_group,
    Tensor grad_offset, Tensor grad_mask);

void modulated_deformable_im2col_impl(
    const Tensor data_im, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor data_col);

void modulated_deformable_col2im_impl(
    const Tensor data_col, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im);

void modulated_deformable_col2im_coord_impl(
    const Tensor data_col, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deformable_group,
    Tensor grad_offset, Tensor grad_mask);

REGISTER_DEVICE_IMPL(modulated_deformable_im2col_impl, CUDA,
                     modulated_deformable_im2col_cuda);
REGISTER_DEVICE_IMPL(modulated_deformable_col2im_impl, CUDA,
                     modulated_deformable_col2im_cuda);
REGISTER_DEVICE_IMPL(modulated_deformable_col2im_coord_impl, CUDA,
                     modulated_deformable_col2im_coord_cuda);

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

Tensor ms_deform_attn_impl_forward(const Tensor &value,
                                   const Tensor &spatial_shapes,
                                   const Tensor &level_start_index,
                                   const Tensor &sampling_loc,
                                   const Tensor &attn_weight,
                                   const int im2col_step);

void ms_deform_attn_impl_backward(
    const Tensor &value, const Tensor &spatial_shapes,
    const Tensor &level_start_index, const Tensor &sampling_loc,
    const Tensor &attn_weight, const Tensor &grad_output, Tensor &grad_value,
    Tensor &grad_sampling_loc, Tensor &grad_attn_weight, const int im2col_step);

REGISTER_DEVICE_IMPL(ms_deform_attn_impl_forward, CUDA,
                     ms_deform_attn_cuda_forward);
REGISTER_DEVICE_IMPL(ms_deform_attn_impl_backward, CUDA,
                     ms_deform_attn_cuda_backward);

Tensor NMSCUDAKernelLauncher(Tensor boxes, Tensor scores, float iou_threshold,
                             int offset);

Tensor nms_cuda(Tensor boxes, Tensor scores, float iou_threshold, int offset) {
  return NMSCUDAKernelLauncher(boxes, scores, iou_threshold, offset);
}

Tensor nms_impl(Tensor boxes, Tensor scores, float iou_threshold, int offset);
REGISTER_DEVICE_IMPL(nms_impl, CUDA, nms_cuda);

void PSAMaskForwardCUDAKernelLauncher(const int psa_type, const Tensor input,
                                      Tensor output, const int num_,
                                      const int h_feature, const int w_feature,
                                      const int h_mask, const int w_mask,
                                      const int half_h_mask,
                                      const int half_w_mask);

void PSAMaskBackwardCUDAKernelLauncher(
    const int psa_type, const Tensor grad_output, Tensor grad_input,
    const int num_, const int h_feature, const int w_feature, const int h_mask,
    const int w_mask, const int half_h_mask, const int half_w_mask);

void psamask_forward_cuda(const int psa_type, const Tensor input, Tensor output,
                          const int num_, const int h_feature,
                          const int w_feature, const int h_mask,
                          const int w_mask, const int half_h_mask,
                          const int half_w_mask) {
  PSAMaskForwardCUDAKernelLauncher(psa_type, input, output, num_, h_feature,
                                   w_feature, h_mask, w_mask, half_h_mask,
                                   half_w_mask);
}

void psamask_backward_cuda(const int psa_type, const Tensor grad_output,
                           Tensor grad_input, const int num_,
                           const int h_feature, const int w_feature,
                           const int h_mask, const int w_mask,
                           const int half_h_mask, const int half_w_mask) {
  PSAMaskBackwardCUDAKernelLauncher(psa_type, grad_output, grad_input, num_,
                                    h_feature, w_feature, h_mask, w_mask,
                                    half_h_mask, half_w_mask);
}

void psamask_forward_impl(const int psa_type, const Tensor input, Tensor output,
                          const int num_, const int h_feature,
                          const int w_feature, const int h_mask,
                          const int w_mask, const int half_h_mask,
                          const int half_w_mask);

void psamask_backward_impl(const int psa_type, const Tensor grad_output,
                           Tensor grad_input, const int num_,
                           const int h_feature, const int w_feature,
                           const int h_mask, const int w_mask,
                           const int half_h_mask, const int half_w_mask);
REGISTER_DEVICE_IMPL(psamask_forward_impl, CUDA, psamask_forward_cuda);
REGISTER_DEVICE_IMPL(psamask_backward_impl, CUDA, psamask_backward_cuda);

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

void roi_align_forward_cuda(Tensor input, Tensor rois, Tensor output,
                            Tensor argmax_y, Tensor argmax_x,
                            int aligned_height, int aligned_width,
                            float spatial_scale, int sampling_ratio,
                            int pool_mode, bool aligned) {
  ROIAlignForwardCUDAKernelLauncher(
      input, rois, output, argmax_y, argmax_x, aligned_height, aligned_width,
      spatial_scale, sampling_ratio, pool_mode, aligned);
}

void roi_align_backward_cuda(Tensor grad_output, Tensor rois, Tensor argmax_y,
                             Tensor argmax_x, Tensor grad_input,
                             int aligned_height, int aligned_width,
                             float spatial_scale, int sampling_ratio,
                             int pool_mode, bool aligned) {
  ROIAlignBackwardCUDAKernelLauncher(
      grad_output, rois, argmax_y, argmax_x, grad_input, aligned_height,
      aligned_width, spatial_scale, sampling_ratio, pool_mode, aligned);
}

void roi_align_forward_impl(Tensor input, Tensor rois, Tensor output,
                            Tensor argmax_y, Tensor argmax_x,
                            int aligned_height, int aligned_width,
                            float spatial_scale, int sampling_ratio,
                            int pool_mode, bool aligned);

void roi_align_backward_impl(Tensor grad_output, Tensor rois, Tensor argmax_y,
                             Tensor argmax_x, Tensor grad_input,
                             int aligned_height, int aligned_width,
                             float spatial_scale, int sampling_ratio,
                             int pool_mode, bool aligned);

REGISTER_DEVICE_IMPL(roi_align_forward_impl, CUDA, roi_align_forward_cuda);
REGISTER_DEVICE_IMPL(roi_align_backward_impl, CUDA, roi_align_backward_cuda);

void ROIPoolForwardCUDAKernelLauncher(Tensor input, Tensor rois, Tensor output,
                                      Tensor argmax, int pooled_height,
                                      int pooled_width, float spatial_scale);

void ROIPoolBackwardCUDAKernelLauncher(Tensor grad_output, Tensor rois,
                                       Tensor argmax, Tensor grad_input,
                                       int pooled_height, int pooled_width,
                                       float spatial_scale);

void roi_pool_forward_cuda(Tensor input, Tensor rois, Tensor output,
                           Tensor argmax, int pooled_height, int pooled_width,
                           float spatial_scale) {
  ROIPoolForwardCUDAKernelLauncher(input, rois, output, argmax, pooled_height,
                                   pooled_width, spatial_scale);
}

void roi_pool_backward_cuda(Tensor grad_output, Tensor rois, Tensor argmax,
                            Tensor grad_input, int pooled_height,
                            int pooled_width, float spatial_scale) {
  ROIPoolBackwardCUDAKernelLauncher(grad_output, rois, argmax, grad_input,
                                    pooled_height, pooled_width, spatial_scale);
}

void roi_pool_forward_impl(Tensor input, Tensor rois, Tensor output,
                           Tensor argmax, int pooled_height, int pooled_width,
                           float spatial_scale);
void roi_pool_backward_impl(Tensor grad_output, Tensor rois, Tensor argmax,
                            Tensor grad_input, int pooled_height,
                            int pooled_width, float spatial_scale);
REGISTER_DEVICE_IMPL(roi_pool_forward_impl, CUDA, roi_pool_forward_cuda);
REGISTER_DEVICE_IMPL(roi_pool_backward_impl, CUDA, roi_pool_backward_cuda);

typedef enum { SUM = 0, MEAN = 1, MAX = 2 } reduce_t;

void SyncBNForwardMeanCUDAKernelLauncher(const Tensor input, Tensor mean);

void SyncBNForwardVarCUDAKernelLauncher(const Tensor input, const Tensor mean,
                                        Tensor var);

void SyncBNForwardOutputCUDAKernelLauncher(
    const Tensor input, const Tensor mean, const Tensor var,
    Tensor running_mean, Tensor running_var, const Tensor weight,
    const Tensor bias, Tensor norm, Tensor std, Tensor output, float eps,
    float momentum, int group_size);

void SyncBNBackwardParamCUDAKernelLauncher(const Tensor grad_output,
                                           const Tensor norm,
                                           Tensor grad_weight,
                                           Tensor grad_bias);

void SyncBNBackwardDataCUDAKernelLauncher(const Tensor grad_output,
                                          const Tensor weight,
                                          const Tensor grad_weight,
                                          const Tensor grad_bias,
                                          const Tensor norm, const Tensor std,
                                          Tensor grad_input);

void sync_bn_forward_mean_cuda(const Tensor input, Tensor mean) {
  SyncBNForwardMeanCUDAKernelLauncher(input, mean);
}

void sync_bn_forward_var_cuda(const Tensor input, const Tensor mean,
                              Tensor var) {
  SyncBNForwardVarCUDAKernelLauncher(input, mean, var);
}

void sync_bn_forward_output_cuda(const Tensor input, const Tensor mean,
                                 const Tensor var, Tensor running_mean,
                                 Tensor running_var, const Tensor weight,
                                 const Tensor bias, Tensor norm, Tensor std,
                                 Tensor output, float eps, float momentum,
                                 int group_size) {
  SyncBNForwardOutputCUDAKernelLauncher(input, mean, var, running_mean,
                                        running_var, weight, bias, norm, std,
                                        output, eps, momentum, group_size);
}

void sync_bn_backward_param_cuda(const Tensor grad_output, const Tensor norm,
                                 Tensor grad_weight, Tensor grad_bias) {
  SyncBNBackwardParamCUDAKernelLauncher(grad_output, norm, grad_weight,
                                        grad_bias);
}

void sync_bn_backward_data_cuda(const Tensor grad_output, const Tensor weight,
                                const Tensor grad_weight,
                                const Tensor grad_bias, const Tensor norm,
                                const Tensor std, Tensor grad_input) {
  SyncBNBackwardDataCUDAKernelLauncher(grad_output, weight, grad_weight,
                                       grad_bias, norm, std, grad_input);
}

void sync_bn_forward_mean_impl(const Tensor input, Tensor mean);

void sync_bn_forward_var_impl(const Tensor input, const Tensor mean,
                              Tensor var);

void sync_bn_forward_output_impl(const Tensor input, const Tensor mean,
                                 const Tensor var, Tensor running_mean,
                                 Tensor running_var, const Tensor weight,
                                 const Tensor bias, Tensor norm, Tensor std,
                                 Tensor output, float eps, float momentum,
                                 int group_size);

void sync_bn_backward_param_impl(const Tensor grad_output, const Tensor norm,
                                 Tensor grad_weight, Tensor grad_bias);

void sync_bn_backward_data_impl(const Tensor grad_output, const Tensor weight,
                                const Tensor grad_weight,
                                const Tensor grad_bias, const Tensor norm,
                                const Tensor std, Tensor grad_input);

REGISTER_DEVICE_IMPL(sync_bn_forward_mean_impl, CUDA,
                     sync_bn_forward_mean_cuda);
REGISTER_DEVICE_IMPL(sync_bn_forward_var_impl, CUDA, sync_bn_forward_var_cuda);
REGISTER_DEVICE_IMPL(sync_bn_forward_output_impl, CUDA,
                     sync_bn_forward_output_cuda);
REGISTER_DEVICE_IMPL(sync_bn_backward_param_impl, CUDA,
                     sync_bn_backward_param_cuda);
REGISTER_DEVICE_IMPL(sync_bn_backward_data_impl, CUDA,
                     sync_bn_backward_data_cuda);
