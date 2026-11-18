# Licenses for special operations

Upstream MMCV ships five operations under the NVIDIA Source Code License
(non-commercial use only) instead of Apache 2.0: `upfirdn2d`,
`fused_leaky_relu` (`FusedBiasLeakyReLU`), `bias_act`, `filtered_lrelu` and
`conv2d_gradfix`.

This fork (branch `nv`) removed all five, so everything left is under Apache
2.0. They are still in the tag `pre-slim`.
