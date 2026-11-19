// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
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

// Without class weights the kernels get an empty tensor (a null pointer).
static Tensor weight_or_empty(const std::optional<Tensor> &weight,
                              const Tensor &input) {
  return weight.has_value() ? weight->contiguous()
                            : at::empty({0}, input.options());
}

// Element-wise loss, shape (N, C); the caller reduces it.
Tensor sigmoid_focal_loss_forward(const Tensor &input_, const Tensor &target_,
                                  const std::optional<Tensor> &weight_,
                                  double gamma, double alpha) {
  at::DeviceGuard guard(input_.device());
  const Tensor input = input_.contiguous(), target = target_.contiguous(),
               weight = weight_or_empty(weight_, input);
  Tensor output = at::zeros_like(input);
  SigmoidFocalLossForwardCUDAKernelLauncher(input, target, weight, output,
                                            gamma, alpha);
  return output;
}

// d loss[i, j] / d input[i, j], shape (N, C)
Tensor sigmoid_focal_loss_backward(const Tensor &input_, const Tensor &target_,
                                   const std::optional<Tensor> &weight_,
                                   double gamma, double alpha) {
  at::DeviceGuard guard(input_.device());
  const Tensor input = input_.contiguous(), target = target_.contiguous(),
               weight = weight_or_empty(weight_, input);
  Tensor grad_input = at::zeros_like(input);
  SigmoidFocalLossBackwardCUDAKernelLauncher(input, target, weight, grad_input,
                                             gamma, alpha);
  return grad_input;
}

// Per-sample loss, shape (N,), and the softmax of the logits, which the
// backward needs. The softmax is computed with the same steps as before the
// op was registered, so results are unchanged.
std::tuple<Tensor, Tensor> softmax_focal_loss_forward(
    const Tensor &input_, const Tensor &target_,
    const std::optional<Tensor> &weight_, double gamma, double alpha) {
  at::DeviceGuard guard(input_.device());
  const Tensor input = input_.contiguous(), target = target_.contiguous(),
               weight = weight_or_empty(weight_, input);
  const Tensor channel_stats = std::get<0>(input.max(1));
  Tensor softmax = (input - channel_stats.unsqueeze(1).expand_as(input)).exp();
  softmax /= softmax.sum(1).unsqueeze(1).expand_as(input);
  Tensor output = at::zeros({input.size(0)}, input.options());
  SoftmaxFocalLossForwardCUDAKernelLauncher(softmax, target, weight, output,
                                            gamma, alpha);
  return {output, softmax};
}

// d loss[i] / d input[i, j] from the forward's softmax, shape (N, C)
Tensor softmax_focal_loss_backward(const Tensor &softmax_,
                                   const Tensor &target_,
                                   const std::optional<Tensor> &weight_,
                                   double gamma, double alpha) {
  at::DeviceGuard guard(softmax_.device());
  const Tensor softmax = softmax_.contiguous(), target = target_.contiguous(),
               weight = weight_or_empty(weight_, softmax);
  Tensor buff = at::zeros({softmax.size(0)}, softmax.options());
  Tensor grad_input = at::zeros_like(softmax);
  SoftmaxFocalLossBackwardCUDAKernelLauncher(softmax, target, weight, buff,
                                             grad_input, gamma, alpha);
  return grad_input;
}

TORCH_LIBRARY_IMPL(mmcv, CUDA, m) {
  m.impl("sigmoid_focal_loss_forward", TORCH_FN(sigmoid_focal_loss_forward));
  m.impl("sigmoid_focal_loss_backward", TORCH_FN(sigmoid_focal_loss_backward));
  m.impl("softmax_focal_loss_forward", TORCH_FN(softmax_focal_loss_forward));
  m.impl("softmax_focal_loss_backward", TORCH_FN(softmax_focal_loss_backward));
}
#endif

TORCH_LIBRARY_FRAGMENT(mmcv, m) {
  m.def(
      "sigmoid_focal_loss_forward(Tensor input, Tensor target, Tensor? weight, "
      "float gamma, float alpha) -> Tensor");
  m.def(
      "sigmoid_focal_loss_backward(Tensor input, Tensor target, "
      "Tensor? weight, float gamma, float alpha) -> Tensor");
  m.def(
      "softmax_focal_loss_forward(Tensor input, Tensor target, Tensor? weight, "
      "float gamma, float alpha) -> (Tensor, Tensor)");
  m.def(
      "softmax_focal_loss_backward(Tensor softmax, Tensor target, "
      "Tensor? weight, float gamma, float alpha) -> Tensor");
}
