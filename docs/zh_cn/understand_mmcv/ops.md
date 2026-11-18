## 算子

MMCV 提供了检测、分割等任务中常用的算子

本 fork（`nv` 分支）只支持 NVIDIA GPU 和 CPU。OpenMMLab 的检测、分割、分类代码都用不到的算子，以及 5 个 NVIDIA 非商用许可的算子已经删除，tag `pre-slim` 里还有。CrissCrossAttention、CornerPool、MergeCells、PointSample 是纯 PyTorch 实现，没有编译库也能用；MultiScaleDeformableAttn 在 CPU 上、没有编译库时、以及导出 ONNX 时走它的 PyTorch 版。

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
