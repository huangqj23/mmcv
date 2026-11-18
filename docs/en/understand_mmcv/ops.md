## ops

We implement common ops used in detection, segmentation, etc.

This fork (branch `nv`) supports NVIDIA GPUs and CPU only. Ops that no OpenMMLab detection / segmentation / classification code uses, and the five ops under NVIDIA's non-commercial license, were removed; they are still in the tag `pre-slim`. CrissCrossAttention, CornerPool, MergeCells and PointSample are pure PyTorch and work without the compiled extension; MultiScaleDeformableAttn falls back to its PyTorch version on CPU, without the extension, and during ONNX export.

| Device                       | CPU | CUDA |
| ---------------------------- | --- | ---- |
| BBoxOverlaps                 | √   | √    |
| CARAFE                       |     | √    |
| CornerPool                   | √   | √    |
| CrissCrossAttention          | √   | √    |
| Deformable Convolution v1/v2 | √   | √    |
| Deformable RoIPool           |     | √    |
| MaskedConv                   |     | √    |
| MergeCells                   | √   | √    |
| ModulatedDeformConv2d        | √   | √    |
| MultiScaleDeformableAttn     | √   | √    |
| NMS                          | √   | √    |
| NMSMatch                     | √   |      |
| PointSample / SimpleRoIAlign | √   | √    |
| PSAMask                      | √   | √    |
| RoIAlign                     | √   | √    |
| RoIPool                      |     | √    |
| SAConv2d                     | √   | √    |
| SigmoidFocalLoss             |     | √    |
| SoftmaxFocalLoss             |     | √    |
| SoftNMS                      | √   |      |
| Synchronized BatchNorm       |     | √    |
