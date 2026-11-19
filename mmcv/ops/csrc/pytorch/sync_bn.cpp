// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"

// The per-step pieces of MMSyncBN. Its Python autograd function all-reduces
// between them, so they have no autograd of their own. Inputs are (N, C, L);
// statistics, norm and std are float32.
#ifdef MMCV_WITH_CUDA
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

static at::TensorOptions float_options(const Tensor &t) {
  return t.options().dtype(at::kFloat);
}

Tensor sync_bn_forward_mean(const Tensor &input_) {
  at::DeviceGuard guard(input_.device());
  const Tensor input = input_.contiguous();
  Tensor mean = at::zeros({input.size(1)}, float_options(input));
  SyncBNForwardMeanCUDAKernelLauncher(input, mean);
  return mean;
}

Tensor sync_bn_forward_var(const Tensor &input_, const Tensor &mean) {
  at::DeviceGuard guard(input_.device());
  const Tensor input = input_.contiguous();
  Tensor var = at::zeros({input.size(1)}, float_options(input));
  SyncBNForwardVarCUDAKernelLauncher(input, mean.contiguous(), var);
  return var;
}

// output, norm and std; updates running_mean and running_var in place
std::tuple<Tensor, Tensor, Tensor> sync_bn_forward_output(
    const Tensor &input_, const Tensor &mean, const Tensor &var,
    const Tensor &weight, const Tensor &bias, Tensor &running_mean,
    Tensor &running_var, double eps, double momentum, int64_t group_size) {
  TORCH_CHECK(running_mean.is_contiguous() && running_var.is_contiguous(),
              "sync_bn: running statistics must be contiguous");
  at::DeviceGuard guard(input_.device());
  const Tensor input = input_.contiguous();
  Tensor output = at::zeros_like(input);
  Tensor norm = at::zeros_like(input, float_options(input));
  Tensor std = at::zeros({input.size(1)}, float_options(input));
  SyncBNForwardOutputCUDAKernelLauncher(
      input, mean.contiguous(), var.contiguous(), running_mean, running_var,
      weight.contiguous(), bias.contiguous(), norm, std, output, eps, momentum,
      group_size);
  return {output, norm, std};
}

std::tuple<Tensor, Tensor> sync_bn_backward_param(const Tensor &grad_output_,
                                                  const Tensor &norm) {
  at::DeviceGuard guard(grad_output_.device());
  const Tensor grad_output = grad_output_.contiguous();
  Tensor grad_weight = at::zeros({grad_output.size(1)}, float_options(norm));
  Tensor grad_bias = at::zeros({grad_output.size(1)}, float_options(norm));
  SyncBNBackwardParamCUDAKernelLauncher(grad_output, norm.contiguous(),
                                        grad_weight, grad_bias);
  return {grad_weight, grad_bias};
}

Tensor sync_bn_backward_data(const Tensor &grad_output_, const Tensor &weight,
                             const Tensor &grad_weight, const Tensor &grad_bias,
                             const Tensor &norm, const Tensor &std) {
  at::DeviceGuard guard(grad_output_.device());
  const Tensor grad_output = grad_output_.contiguous();
  Tensor grad_input = at::zeros_like(grad_output);
  SyncBNBackwardDataCUDAKernelLauncher(
      grad_output, weight.contiguous(), grad_weight.contiguous(),
      grad_bias.contiguous(), norm.contiguous(), std.contiguous(), grad_input);
  return grad_input;
}

TORCH_LIBRARY_IMPL(mmcv, CUDA, m) {
  m.impl("sync_bn_forward_mean", TORCH_FN(sync_bn_forward_mean));
  m.impl("sync_bn_forward_var", TORCH_FN(sync_bn_forward_var));
  m.impl("sync_bn_forward_output", TORCH_FN(sync_bn_forward_output));
  m.impl("sync_bn_backward_param", TORCH_FN(sync_bn_backward_param));
  m.impl("sync_bn_backward_data", TORCH_FN(sync_bn_backward_data));
}
#endif

TORCH_LIBRARY_FRAGMENT(mmcv, m) {
  m.def("sync_bn_forward_mean(Tensor input) -> Tensor");
  m.def("sync_bn_forward_var(Tensor input, Tensor mean) -> Tensor");
  m.def(
      "sync_bn_forward_output(Tensor input, Tensor mean, Tensor var, "
      "Tensor weight, Tensor bias, Tensor(a!) running_mean, "
      "Tensor(b!) running_var, float eps, float momentum, int group_size) "
      "-> (Tensor, Tensor, Tensor)");
  m.def(
      "sync_bn_backward_param(Tensor grad_output, Tensor norm) "
      "-> (Tensor, Tensor)");
  m.def(
      "sync_bn_backward_data(Tensor grad_output, Tensor weight, "
      "Tensor grad_weight, Tensor grad_bias, Tensor norm, Tensor std) "
      "-> Tensor");
}
