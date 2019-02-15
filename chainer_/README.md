# Image classification models on Chainer

[![PyPI](https://img.shields.io/pypi/v/chainercv2.svg)](https://pypi.python.org/pypi/chainercv2)
[![Downloads](https://pepy.tech/badge/chainercv2)](https://pepy.tech/project/chainercv2)

This is a collection of image classification models. Many of them are pretrained on ImageNet-1K and CIFAR-10/100
datasets and loaded automatically during use. All pretrained models require the same ordinary normalization.
Scripts for training/evaluating/converting models are in the [`imgclsmob`](https://github.com/osmr/imgclsmob) repo.

## List of implemented models

- AlexNet (['One weird trick for parallelizing convolutional neural networks'](https://arxiv.org/abs/1404.5997))
- ZFNet (['Visualizing and Understanding Convolutional Networks'](https://arxiv.org/abs/1311.2901))
- VGG/BN-VGG (['Very Deep Convolutional Networks for Large-Scale Image Recognition'](https://arxiv.org/abs/1409.1556))
- BN-Inception (['Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift'](https://arxiv.org/abs/1502.03167))
- ResNet (['Deep Residual Learning for Image Recognition'](https://arxiv.org/abs/1512.03385))
- PreResNet (['Identity Mappings in Deep Residual Networks'](https://arxiv.org/abs/1603.05027))
- ResNeXt (['Aggregated Residual Transformations for Deep Neural Networks'](http://arxiv.org/abs/1611.05431))
- SENet/SE-ResNet/SE-PreResNet/SE-ResNeXt (['Squeeze-and-Excitation Networks'](https://arxiv.org/abs/1709.01507))
- AirNet/AirNeXt (['Attention Inspiring Receptive-Fields Network for Learning Invariant Representations'](https://ieeexplore.ieee.org/document/8510896))
- BAM-ResNet (['BAM: Bottleneck Attention Module'](https://arxiv.org/abs/1807.06514))
- CBAM-ResNet (['CBAM: Convolutional Block Attention Module'](https://arxiv.org/abs/1807.06521))
- ResAttNet (['Residual Attention Network for Image Classification'](https://arxiv.org/abs/1704.06904))
- PyramidNet (['Deep Pyramidal Residual Networks'](https://arxiv.org/abs/1610.02915))
- DiracNetV2 (['DiracNets: Training Very Deep Neural Networks Without Skip-Connections'](https://arxiv.org/abs/1706.00388))
- ShaResNet (['ShaResNet: reducing residual network parameter number by sharing weights'](https://arxiv.org/abs/1702.08782))
- DenseNet (['Densely Connected Convolutional Networks'](https://arxiv.org/abs/1608.06993))
- CondenseNet (['CondenseNet: An Efficient DenseNet using Learned Group Convolutions'](https://arxiv.org/abs/1711.09224))
- SparseNet (['Sparsely Aggregated Convolutional Networks'](https://arxiv.org/abs/1801.05895))
- PeleeNet (['Pelee: A Real-Time Object Detection System on Mobile Devices'](https://arxiv.org/abs/1804.06882))
- WRN (['Wide Residual Networks'](https://arxiv.org/abs/1605.07146))
- DRN-C/DRN-D (['Dilated Residual Networks'](https://arxiv.org/abs/1705.09914))
- DPN (['Dual Path Networks'](https://arxiv.org/abs/1707.01629))
- DarkNet Ref/Tiny/19 (['Darknet: Open source neural networks in c'](https://github.com/pjreddie/darknet))
- DarkNet-53 (['YOLOv3: An Incremental Improvement'](https://arxiv.org/abs/1804.02767))
- ChannelNet (['ChannelNets: Compact and Efficient Convolutional Neural Networks via Channel-Wise Convolutions'](https://arxiv.org/abs/1809.01330))
- DLA (['Deep Layer Aggregation'](https://arxiv.org/abs/1707.06484))
- FishNet (['FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction'](http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction.pdf))
- SqueezeNet/SqueezeResNet (['SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size'](https://arxiv.org/abs/1602.07360))
- SqueezeNext (['SqueezeNext: Hardware-Aware Neural Network Design'](https://arxiv.org/abs/1803.10615))
- ShuffleNet (['ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices'](https://arxiv.org/abs/1707.01083))
- ShuffleNetV2 (['ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design'](https://arxiv.org/abs/1807.11164))
- MENet (['Merging and Evolution: Improving Convolutional Neural Networks for Mobile Applications'](https://arxiv.org/abs/1803.09127))
- MobileNet (['MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications'](https://arxiv.org/abs/1704.04861))
- FD-MobileNet (['FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy'](https://arxiv.org/abs/1802.03750))
- MobileNetV2 (['MobileNetV2: Inverted Residuals and Linear Bottlenecks'](https://arxiv.org/abs/1801.04381))
- IGCV3 (['IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks'](https://arxiv.org/abs/1806.00178))
- MnasNet (['MnasNet: Platform-Aware Neural Architecture Search for Mobile'](https://arxiv.org/abs/1807.11626))
- DARTS (['DARTS: Differentiable Architecture Search'](https://arxiv.org/abs/1806.09055))
- Xception (['Xception: Deep Learning with Depthwise Separable Convolutions'](https://arxiv.org/abs/1610.02357))
- InceptionV3 (['Rethinking the Inception Architecture for Computer Vision'](https://arxiv.org/abs/1512.00567))
- InceptionV4/InceptionResNetV2 (['Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning'](https://arxiv.org/abs/1602.07261))
- PolyNet (['PolyNet: A Pursuit of Structural Diversity in Very Deep Networks'](https://arxiv.org/abs/1611.05725))
- NASNet (['Learning Transferable Architectures for Scalable Image Recognition'](https://arxiv.org/abs/1707.07012))
- PNASNet (['Progressive Neural Architecture Search'](https://arxiv.org/abs/1712.00559))
- NIN (['Network In Network'](https://arxiv.org/abs/1312.4400))
- ResDrop-ResNet (['Deep Networks with Stochastic Depth'](https://arxiv.org/abs/1603.09382))
- Shake-Shake-ResNet (['Shake-Shake regularization'](https://arxiv.org/abs/1705.07485))
- ShakeDrop-ResNet (['ShakeDrop Regularization for Deep Residual Learning'](https://arxiv.org/abs/1802.02375))

## Installation

To use the models in your project, simply install the `chainercv2` package:
```
pip install chainercv2
```
To enable/disable different hardware supports, check out Chainer installation [instructions](https://chainer.org).

## Usage

Example of using a pretrained ResNet-18 model:
```
from chainercv2.model_provider import get_model as chcv2_get_model
import numpy as np

net = chcv2_get_model("resnet18", pretrained=True)
x = np.zeros((1, 3, 224, 224), np.float32)
y = net(x)
```

## Pretrained models

### Imagenet-1K

Some remarks:
- Top1/Top5 are the standard 1-crop Top-1/Top-5 errors (in percents) on the validation subset of the ImageNet-1K dataset.
- FLOPs/2 is the number of FLOPs divided by two to be similar to the number of MACs.
- Remark `Converted from GL model` means that the model was trained on `MXNet/Gluon` and then converted to Chainer.

| Model | Top1 | Top5 | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | ---: | --- |
| AlexNet | 44.08 | 21.32 | 61,100,840 | 714.83M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.108/alexnet-2132-cea565f1.npz.log)) |
| VGG-11 | 31.89 | 11.79 | 132,863,336 | 7,615.87M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg11-1179-3cc057e6.npz.log)) |
| VGG-13 | 31.06 | 11.16 | 133,047,848 | 11,317.65M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg13-1116-e835ca5a.npz.log)) |
| VGG-16 | 26.75 | 8.70 | 138,357,544 | 15,480.10M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg16-0870-8741ff5c.npz.log)) |
| VGG-19 | 25.86 | 8.23 | 143,667,240 | 19,642.55M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg19-0823-18980884.npz.log)) |
| BN-VGG-11b | 30.37 | 10.60 | 132,868,840 | 7,630.72M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg11b-1060-8964402b.npz.log)) |
| BN-VGG-13b | 29.45 | 10.19 | 133,053,736 | 11,342.14M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg13b-1019-0121b0a4.npz.log)) |
| BN-VGG-16b | 26.89 | 8.63 | 138,365,992 | 15,507.20M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg16b-0863-cbaa2105.npz.log)) |
| BN-VGG-19b | 25.65 | 8.16 | 143,678,248 | 19,672.26M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg19b-0816-dc5e37a5.npz.log)) |
| BN-Inception | 25.08 | 7.78 | 11,295,240 | 2,048.06M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.139/bninception-0778-99f685c2.npz.log)) |
| ResNet-10 | 37.12 | 15.49 | 5,418,792 | 894.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet10-1549-b31f1135.npz.log)) |
| ResNet-12 | 35.86 | 14.48 | 5,492,776 | 1,126.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.30/resnet12-1448-11acb729.npz.log)) |
| ResNet-14 | 32.84 | 12.42 | 5,788,200 | 1,357.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.40/resnet14-1242-4e65746b.npz.log)) |
| ResNet-16 | 30.66 | 11.07 | 6,968,872 | 1,589.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.41/resnet16-1107-b1d7fb7d.npz.log)) |
| ResNet-18 x0.25 | 49.08 | 24.48 | 831,096 | 137.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.47/resnet18_wd4-2448-58c4a007.npz.log)) |
| ResNet-18 x0.5 | 36.55 | 14.99 | 3,055,880 | 486.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.46/resnet18_wd2-1499-542ed773.npz.log)) |
| ResNet-18 x0.75 | 33.27 | 12.56 | 6,675,352 | 1,047.53M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet18_w3d4-1256-ce2011df.npz.log)) |
| ResNet-18 | 28.08 | 9.59 | 11,689,512 | 1,820.41M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.153/resnet18-0959-d80fbe60.npz.log)) |
| ResNet-34 | 25.35 | 7.95 | 21,797,672 | 3,672.68M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet34-0795-0b392267.npz.log)) |
| ResNet-50 | 22.61 | 6.41 | 25,557,032 | 3,877.95M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.147/resnet50-0641-ca0cd7a1.npz.log)) |
| ResNet-50b | 22.34 | 6.18 | 25,557,032 | 4,110.48M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.146/resnet50b-0618-42fffef9.npz.log)) |
| ResNet-101 | 21.65 | 6.01 | 44,549,160 | 7,597.95M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet101-0601-d8cddbea.npz.log)) |
| ResNet-101b | 20.79 | 5.40 | 44,549,160 | 7,830.48M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.145/resnet101b-0540-af300066.npz.log)) |
| ResNet-152 | 20.74 | 5.35 | 60,192,808 | 11,321.85M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.144/resnet152-0535-64c1daa7.npz.log)) |
| ResNet-152b | 20.29 | 5.27 | 60,192,808 | 11,554.38M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.143/resnet152b-0527-6efec251.npz.log)) |
| PreResNet-18 | 28.17 | 9.54 | 11,687,848 | 1,820.56M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.140/preresnet18-0954-21e4811a.npz.log)) |
| PreResNet-34 | 25.89 | 8.12 | 21,796,008 | 3,672.83M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet34-0812-829f5a23.npz.log)) |
| PreResNet-50 | 23.36 | 6.69 | 25,549,480 | 3,875.44M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet50-0669-40bd5e93.npz.log)) |
| PreResNet-50b | 23.08 | 6.67 | 25,549,480 | 4,107.97M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet50b-0667-b7d221ef.npz.log)) |
| PreResNet-101 | 21.45 | 5.75 | 44,541,608 | 7,595.44M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet101-0575-f6f6789a.npz.log)) |
| PreResNet-101b | 21.61 | 5.87 | 44,541,608 | 7,827.97M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet101b-0587-4211c5ab.npz.log)) |
| PreResNet-152 | 20.73 | 5.30 | 60,185,256 | 11,319.34M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet152-0530-021d99dc.npz.log)) |
| PreResNet-152b | 20.88 | 5.66 | 60,185,256 | 11,551.87M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet152b-0566-fdd337e7.npz.log)) |
| PreResNet-200b | 21.03 | 5.60 | 64,666,280 | 15,068.63M | From [tornadomeet/ResNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.45/preresnet200b-0560-f79bd952.npz.log)) |
| ResNeXt-101 (32x4d) | 21.11 | 5.69 | 44,177,704 | 8,003.45M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.26/resnext101_32x4d-0569-c6d1c30d.npz.log)) |
| ResNeXt-101 (64x4d) | 20.57 | 5.43 | 83,455,272 | 15,500.27M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.26/resnext101_64x4d-0543-dd8b7d96.npz.log)) |
| SE-ResNet-50 | 22.53 | 6.41 | 28,088,024 | 3,880.49M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.24/seresnet50-0641-f3d68cfc.npz.log)) |
| SE-ResNet-101 | 21.90 | 5.88 | 49,326,872 | 7,602.76M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.24/seresnet101-0588-e45a9f8f.npz.log)) |
| SE-ResNet-152 | 21.46 | 5.77 | 66,821,848 | 11,328.52M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.24/seresnet152-0577-a089ba52.npz.log)) |
| SE-ResNeXt-50 (32x4d) | 21.04 | 5.58 | 27,559,896 | 4,258.40M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.27/seresnext50_32x4d-0558-5c435c1b.npz.log)) |
| SE-ResNeXt-101 (32x4d) | 19.99 | 5.01 | 48,955,416 | 8,008.26M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.27/seresnext101_32x4d-0501-98ea6fc4.npz.log)) |
| SENet-154 | 18.79 | 4.63 | 115,088,984 | 20,745.78M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.28/senet154-0463-381d2494.npz.log)) |
| AirNet50-1x64d (r=2) | 22.46 | 6.20 | 27,425,864 | 4,772.11M | From [soeaver/AirNet-PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.120/airnet50_1x64d_r2-0620-b6a9359d.npz.log)) |
| AirNet50-1x64d (r=16) | 22.89 | 6.50 | 25,714,952 | 4,399.97M | From [soeaver/AirNet-PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.120/airnet50_1x64d_r16-0650-95da530f.npz.log)) |
| AirNeXt50-32x4d (r=2) | 21.50 | 5.73 | 27,604,296 | 5,339.58M | From [soeaver/AirNet-PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.120/airnext50_32x4d_r2-0573-160860f7.npz.log)) |
| BAM-ResNet-50 | 23.71 | 6.97 | 25,915,099 | 4,196.09M | From [Jongchan/attention-module] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.124/bam_resnet50-0697-a8c65533.npz.log)) |
| CBAM-ResNet-50 | 22.99 | 6.40 | 28,089,624 | 4,116.97M | From [Jongchan/attention-module] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.125/cbam_resnet50-0640-b2314d97.npz.log)) |
| PyramidNet-101 (a=360) | 22.66 | 6.49 | 42,455,070 | 8,743.54M | From [dyhan0920/Pyramid...PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.104/pyramidnet101_a360-0649-b68c786b.npz.log)) |
| DiracNetV2-18 | 30.60 | 11.13 | 11,511,784 | 1,796.62M | From [szagoruyko/diracnets] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.111/diracnet18v2-1113-b85b43d1.npz.log)) |
| DiracNetV2-34 | 27.90 | 9.48 | 21,616,232 | 3,646.93M | From [szagoruyko/diracnets] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.111/diracnet34v2-0948-0245163a.npz.log)) |
| DenseNet-121 | 25.04 | 7.79 | 7,978,856 | 2,872.13M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.29/densenet121-0779-06d5ebbf.npz.log)) |
| DenseNet-161 | 22.36 | 6.20 | 28,681,000 | 7,793.16M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.29/densenet161-0620-6d05f3b9.npz.log)) |
| DenseNet-169 | 23.85 | 6.86 | 14,149,480 | 3,403.89M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.29/densenet169-0686-1978656b.npz.log)) |
| DenseNet-201 | 22.64 | 6.29 | 20,013,928 | 4,347.15M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.29/densenet201-0629-77702939.npz.log)) |
| CondenseNet-74 (C=G=4) | 26.81 | 8.61 | 4,773,944 | 546.06M | From [ShichenLiu/CondenseNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.36/condensenet74_c4_g4-0861-ef6077ec.npz.log)) |
| CondenseNet-74 (C=G=8) | 29.74 | 10.43 | 2,935,416 | 291.52M | From [ShichenLiu/CondenseNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.36/condensenet74_c8_g8-1043-277fbfb8.npz.log)) |
| PeleeNet | 31.61 | 11.27 | 2,802,248 | 514.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.141/peleenet-1127-ef057fc9.npz.log)) |
| WRN-50-2 | 22.06 | 6.13 | 68,849,128 | 11,405.42M | From [szagoruyko/functional-zoo] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.113/wrn50_2-0613-d0cd9171.npz.log)) |
| DRN-C-26 | 25.68 | 7.88 | 21,126,584 | 16,993.90M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnc26-0788-762c34c1.npz.log)) |
| DRN-C-42 | 23.72 | 6.93 | 31,234,744 | 25,093.75M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnc42-0693-ec938cc4.npz.log)) |
| DRN-C-58 | 22.35 | 6.29 | 40,542,008 | 32,489.94M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnc58-0629-063ef199.npz.log)) |
| DRN-D-22 | 26.65 | 8.50 | 16,393,752 | 13,051.33M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd22-0850-b25d4757.npz.log)) |
| DRN-D-38 | 24.53 | 7.36 | 26,501,912 | 21,151.19M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd38-0736-153481d6.npz.log)) |
| DRN-D-54 | 22.08 | 6.23 | 35,809,176 | 28,547.38M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd54-0623-31e8eeb8.npz.log)) |
| DRN-D-105 | 21.32 | 5.84 | 54,801,304 | 43,442.43M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd105-0584-c0d7657b.npz.log)) |
| DPN-68 | 23.61 | 7.01 | 12,611,602 | 2,351.84M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.34/dpn68-0701-ad8cd4ec.npz.log)) |
| DPN-98 | 20.80 | 5.53 | 61,570,728 | 11,716.51M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.34/dpn98-0553-9cd57335.npz.log)) |
| DPN-131 | 20.04 | 5.23 | 79,254,504 | 16,076.15M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.34/dpn131-0523-e3721599.npz.log)) |
| DarkNet Tiny | 40.33 | 17.46 | 1,042,104 | 500.85M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.69/darknet_tiny-1746-b04fa463.npz.log)) |
| DarkNet Ref | 38.09 | 16.71 | 7,319,416 | 367.59M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.64/darknet_ref-1671-b2d5721f.npz.log)) |
| DarkNet-53 | 21.41 | 5.56 | 41,609,928 | 7,133.86M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.150/darknet53-0556-42c57951.npz.log)) |
| DLA-34 | 26.14 | 8.23 | 15,742,104 | 3,071.37M | From [ucbdrive/dla] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.202/dla34-0823-45504b09.npz.log)) |
| DLA-46-C | 36.78 | 14.71 | 1,301,400 | 585.45M | From [ucbdrive/dla] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.202/dla46c-1471-487ae254.npz.log)) |
| DLA-X-46-C | 35.59 | 13.96 | 1,068,440 | 546.72M | From [ucbdrive/dla] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.202/dla46xc-1396-a40cc675.npz.log)) |
| DLA-60 | 23.78 | 7.11 | 22,036,632 | 4,255.49M | From [ucbdrive/dla] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.202/dla60-0711-92693875.npz.log)) |
| DLA-X-60 | 22.46 | 6.20 | 17,352,344 | 3,543.68M | From [ucbdrive/dla] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.202/dla60x-0620-444f31ea.npz.log)) |
| DLA-X-60-C | 33.41 | 12.38 | 1,319,832 | 596.06M | From [ucbdrive/dla] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.202/dla60xc-1238-5c662c84.npz.log)) |
| DLA-102 | 22.87 | 6.42 | 33,268,888 | 7,190.95M | From [ucbdrive/dla] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.202/dla102-0642-c4ee6dcb.npz.log)) |
| DLA-X-102 | 21.93 | 5.99 | 26,309,272 | 5,884.94M | From [ucbdrive/dla] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.202/dla102x-0599-7f83bc04.npz.log)) |
| DLA-X2-102 | 21.11 | 5.54 | 41,282,200 | 9,340.61M | From [ucbdrive/dla] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.202/dla102x2-0554-6a27a094.npz.log)) |
| DLA-169 | 21.99 | 5.90 | 53,389,720 | 11,593.20M | From [ucbdrive/dla] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.202/dla169-0590-96b692a8.npz.log)) |
| FishNet-150 | 22.86 | 6.39 | 24,959,400 | 6,435.05M | From [kevin-ssy/FishNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.168/fishnet150-0639-114d15a6.npz.log)) |
| SqueezeNet v1.0 | 38.76 | 17.38 | 1,248,424 | 823.67M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.128/squeezenet_v1_0-1738-4c55a6a5.npz.log)) |
| SqueezeNet v1.1 | 39.13 | 17.40 | 1,235,496 | 352.02M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.88/squeezenet_v1_1-1740-b236c204.npz.log)) |
| SqueezeResNet v1.0 | 39.36 | 17.66 | 1,248,424 | 823.67M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.178/squeezeresnet_v1_0-1766-6dc69dc2.npz.log)) |
| SqueezeResNet v1.1 | 39.85 | 17.87 | 1,235,496 | 352.02M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.70/squeezeresnet_v1_1-1787-f40e6051.npz.log)) |
| 1.0-SqNxt-23 | 42.62 | 19.03 | 724,056 | 287.28M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.171/sqnxt23_w1-1903-ef3d725b.npz.log)) |
| 1.0-SqNxt-23v5 | 40.96 | 17.86 | 921,816 | 285.82M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.172/sqnxt23v5_w1-1786-8b24c6e3.npz.log)) |
| 1.5-SqNxt-23 | 34.71 | 13.44 | 1,511,824 | 552.39M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.210/sqnxt23_w3d2-1344-a5c3b21e.npz.log)) |
| 1.5-SqNxt-23v5 | 33.79 | 12.92 | 1,953,616 | 550.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.212/sqnxt23v5_w3d2-1292-c997e279.npz.log)) |
| ShuffleNet x0.25 (g=1) | 62.04 | 36.81 | 209,746 | 12.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.134/shufflenet_g1_wd4-3681-15d3e787.npz.log)) |
| ShuffleNet x0.25 (g=3) | 61.30 | 36.16 | 305,902 | 13.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.135/shufflenet_g3_wd4-3616-064f7f7f.npz.log)) |
| ShuffleNet x0.5 (g=1) | 46.24 | 22.35 | 534,484 | 41.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.174/shufflenet_g1_wd2-2235-5d83cc28.npz.log)) |
| ShuffleNet x0.5 (g=3) | 43.83 | 20.61 | 718,324 | 41.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.167/shufflenet_g3_wd2-2061-557e4397.npz.log)) |
| ShuffleNetV2 x0.5 | 43.45 | 20.73 | 1,366,792 | 43.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.90/shufflenetv2_wd2-2073-c5e5a23c.npz.log)) |
| ShuffleNetV2 x1.0 | 33.39 | 12.98 | 2,278,604 | 149.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.133/shufflenetv2_w1-1298-3830a2da.npz.log)) |
| ShuffleNetV2 x1.5 | 33.96 | 13.37 | 4,406,098 | 320.77M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.65/shufflenetv2_w3d2-1337-66c1d6ed.npz.log)) |
| ShuffleNetV2 x2.0 | 33.21 | 13.03 | 7,601,686 | 595.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.84/shufflenetv2_w2-1303-349e42b5.npz.log)) |
| ShuffleNetV2b x0.5 | 39.78 | 17.87 | 1,366,792 | 43.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.157/shufflenetv2b_wd2-1787-08a12021.npz.log)) |
| ShuffleNetV2b x1.0 | 30.36 | 11.00 | 2,279,760 | 150.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.161/shufflenetv2b_w1-1100-21562fb2.npz.log)) |
| ShuffleNetV2b x1.5 | 26.92 | 8.78 | 4,410,194 | 323.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.203/shufflenetv2b_w3d2-0878-7a5c7ed4.npz.log)) |
| 108-MENet-8x1 (g=3) | 43.67 | 20.42 | 654,516 | 42.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.89/menet108_8x1_g3-2042-9e3ff283.npz.log)) |
| 128-MENet-8x1 (g=4) | 42.07 | 19.19 | 750,796 | 45.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.103/menet128_8x1_g4-1919-f6fd56fa.npz.log)) |
| 160-MENet-8x1 (g=8) | 43.54 | 20.42 | 850,120 | 45.63M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.154/menet160_8x1_g8-2042-250fd765.npz.log)) |
| 228-MENet-12x1 (g=3) | 33.86 | 13.01 | 1,806,568 | 152.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.131/menet228_12x1_g3-1301-39c25ca3.npz.log)) |
| 256-MENet-12x1 (g=4) | 32.30 | 12.18 | 1,888,240 | 150.65M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.152/menet256_12x1_g4-1218-57160b09.npz.log)) |
| 348-MENet-12x1 (g=3) | 27.86 | 9.36 | 3,368,128 | 312.00M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.173/menet348_12x1_g3-0936-ee7e056d.npz.log)) |
| 352-MENet-12x1 (g=8) | 31.28 | 11.72 | 2,272,872 | 157.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.198/menet352_12x1_g8-1172-c256ae25.npz.log)) |
| 456-MENet-24x1 (g=3) | 29.55 | 10.39 | 5,304,784 | 567.90M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.33/menet456_24x1_g3-1039-f68c36a2.npz.log)) |
| MobileNet x0.25 | 45.85 | 22.16 | 470,072 | 44.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.62/mobilenet_wd4-2216-09c50ab8.npz.log)) |
| MobileNet x0.5 | 33.89 | 13.37 | 1,331,592 | 155.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.156/mobilenet_wd2-1337-48d12ee3.npz.log)) |
| MobileNet x0.75 | 29.86 | 10.53 | 2,585,560 | 333.99M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.130/mobilenet_w3d4-1053-d7ec3192.npz.log)) |
| MobileNet x1.0 | 26.47 | 8.66 | 4,231,976 | 579.80M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.155/mobilenet_w1-0866-b888f817.npz.log)) |
| FD-MobileNet x0.25 | 55.43 | 30.63 | 383,160 | 12.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.177/fdmobilenet_wd4-3063-55407f3a.npz.log)) |
| FD-MobileNet x0.5 | 42.68 | 19.76 | 993,928 | 41.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.83/fdmobilenet_wd2-1976-6299d442.npz.log)) |
| FD-MobileNet x0.75 | 37.94 | 15.99 | 1,833,304 | 86.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.159/fdmobilenet_w3d4-1599-cdfc2e04.npz.log)) |
| FD-MobileNet x1.0 | 33.90 | 13.16 | 2,901,288 | 147.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.162/fdmobilenet_w1-1316-0ed6f00c.npz.log)) |
| MobileNetV2 x0.25 | 48.10 | 24.11 | 1,516,392 | 34.24M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.137/mobilenetv2_wd4-2411-9fc398d3.npz.log)) |
| MobileNetV2 x0.5 | 35.56 | 14.44 | 1,964,736 | 100.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.170/mobilenetv2_wd2-1444-ca0906e1.npz.log)) |
| MobileNetV2 x0.75 | 31.28 | 11.48 | 2,627,592 | 198.50M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.31/mobilenetv2_w3d4-1148-a6f852ea.npz.log)) |
| MobileNetV2 x1.0 | 26.80 | 8.66 | 3,504,960 | 329.36M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.213/mobilenetv2_w1-0866-efc3331e.npz.log)) |
| IGCV3 x0.25 | 53.36 | 28.28 | 1,534,020 | 41.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.142/igcv3_wd4-2828-25942192.npz.log)) |
| IGCV3 x0.5 | 39.36 | 17.04 | 1,985,528 | 111.12M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.132/igcv3_wd2-1704-86246558.npz.log)) |
| IGCV3 x0.75 | 30.67 | 10.99 | 2,638,084 | 210.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.207/igcv3_w3d4-1099-b0dbc54a.npz.log)) |
| IGCV3 x1.0 | 28.20 | 9.55 | 3,491,688 | 340.79M | From [homles11/IGCV3] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.126/igcv3_w1-0955-1c00ac33.npz.log)) |
| MnasNet | 31.27 | 11.44 | 4,308,816 | 317.67M | From [zeusees/Mnasnet...Model] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.117/mnasnet-1144-688e523d.npz.log)) |
| DARTS | 27.29 | 8.97 | 4,718,752 | 539.86M | From [quark0/darts] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.118/darts-0897-8986fe64.npz.log)) |
| Xception | 21.04 | 5.47 | 22,855,952 | 8,403.63M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.115/xception-0547-7a5be958.npz.log)) |
| InceptionV3 | 21.11 | 5.61 | 23,834,568 | 5,743.06M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.92/inceptionv3-0561-4ddea4df.npz.log)) |
| InceptionV4 | 20.62 | 5.26 | 42,679,816 | 12,304.93M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.105/inceptionv4-0526-02e53701.npz.log)) |
| InceptionResNetV2 | 19.93 | 4.92 | 55,843,464 | 13,188.64M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.107/inceptionresnetv2-0492-3d3de82b.npz.log)) |
| PolyNet | 19.08 | 4.50 | 95,366,600 | 34,821.34M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.96/polynet-0450-6dc7028b.npz.log)) |
| NASNet-A 4@1056 | 25.36 | 7.96 | 5,289,978 | 584.90M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.97/nasnet_4a1056-0796-f09950c0.npz.log)) |
| NASNet-A 6@4032 | 18.17 | 4.22 | 88,753,150 | 23,976.44M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.101/nasnet_6a4032-0422-d49d4663.npz.log)) |
| PNASNet-5-Large | 17.90 | 4.26 | 86,057,668 | 25,140.77M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.114/pnasnet5large-0426-3c2755dc.npz.log)) |

### CIFAR-10

| Model | Error, % | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | --- |
| NIN | 7.43 | 966,986 | 222.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.175/nin_cifar10-0743-045abfde.npz.log)) |
| ResNet-20 | 5.97 | 272,474 | 41.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.163/resnet20_cifar10-0597-15145d2e.npz.log)) |
| ResNet-56 | 4.52 | 855,770 | 127.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.163/resnet56_cifar10-0452-eb7923aa.npz.log)) |
| ResNet-110 | 3.69 | 1,730,714 | 255.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.163/resnet110_cifar10-0369-27d76fce.npz.log)) |
| ResNet-164(BN) | 3.68 | 1,704,154 | 255.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.179/resnet164bn_cifar10-0368-d8659366.npz.log)) |
| ResNet-1001 | 3.28 | 10,328,602 | 1,536.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.201/resnet1001_cifar10-0328-0e27556c.npz.log)) |
| ResNet-1202 | 3.53 | 1,9424,026 | 2,857.17M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.214/resnet1202_cifar10-0353-d82bb435.npz.log)) |
| PreResNet-20 | 6.51 | 272,282 | 41.27M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.164/preresnet20_cifar10-0651-5cf94722.npz.log)) |
| PreResNet-56 | 4.49 | 855,578 | 127.03M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.164/preresnet56_cifar10-0449-73ea193a.npz.log)) |
| PreResNet-110 | 3.86 | 1,730,522 | 255.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.164/preresnet110_cifar10-0386-544ed0f0.npz.log)) |
| PreResNet-164(BN) | 3.64 | 1,703,258 | 255.08M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.196/preresnet164bn_cifar10-0364-c0ff2438.npz.log)) |
| PreResNet-1001 | 2.65 | 10,327,706 | 1,536.18M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.209/preresnet1001_cifar10-0265-1f3028bd.npz.log)) |
| ResNeXt-29 (32x4d) | 3.15 | 4,775,754 | 780.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.169/resnext29_32x4d_cifar10-0315-442eca6c.npz.log)) |
| ResNeXt-29 (16x64d) | 2.41 | 68,155,210 | 10,709.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.176/resnext29_16x64d_cifar10-0241-e80d3cb5.npz.log)) |
| PyramidNet-110 (a=48) | 3.72 | 1,772,706 | 408.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.184/pyramidnet110_a48_cifar10-0372-965fce37.npz.log)) |
| PyramidNet-110 (a=84) | 2.98 | 3,904,446 | 778.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.185/pyramidnet110_a84_cifar10-0298-7b38a0f6.npz.log)) |
| PyramidNet-110 (a=270) | 2.51 | 28,485,477 | 4,730.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.194/pyramidnet110_a270_cifar10-0251-b3456ddd.npz.log)) |
| DenseNet-40 (k=12) | 5.61 | 599,050 | 210.80M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.193/densenet40_k12_cifar10-0561-a37df881.npz.log)) |
| DenseNet-100 (k=12) | 3.66 | 4,068,490 | 1,353.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.205/densenet100_k12_cifar10-0366-85031735.npz.log)) |
| DenseNet-BC-100 (k=12) | 4.16 | 769,162 | 298.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.189/densenet100_k12_bc_cifar10-0416-160a0641.npz.log)) |
| WRN-16-10 | 2.93 | 17,116,634 | 2,414.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.166/wrn16_10_cifar10-0293-4ac60015.npz.log)) |
| WRN-28-10 | 2.39 | 36,479,194 | 5,246.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.166/wrn28_10_cifar10-0239-f8a24941.npz.log)) |
| WRN-40-8 | 2.37 | 35,748,314 | 5,176.90M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.166/wrn40_8_cifar10-0237-3f56f24a.npz.log)) |
| Shake-Shake-ResNet-20-2x16d | 5.15 | 541,082 | 81.78M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.215/shakeshakeresnet20_2x16d_cifar10-0515-e2f524b5.npz.log)) |

### CIFAR-100

| Model | Error, % | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | --- |
| NIN | 28.39 | 984,356 | 224.08M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.183/nin_cifar100-2839-89104763.npz.log)) |
| ResNet-20 | 29.64 | 278,324 | 41.30M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.180/resnet20_cifar100-2964-6a85f07e.npz.log)) |
| ResNet-56 | 24.88 | 861,620 | 127.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.181/resnet56_cifar100-2488-2d641cde.npz.log)) |
| ResNet-110 | 22.80 | 1,736,564 | 255.71M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.190/resnet110_cifar100-2280-d2ec4ff1.npz.log)) |
| ResNet-164(BN) | 20.44 | 1,727,284 | 255.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.182/resnet164bn_cifar100-2044-190ab6b4.npz.log)) |
| PreResNet-20 | 30.22 | 278,132 | 41.28M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.187/preresnet20_cifar100-3022-e3fd9391.npz.log)) |
| PreResNet-56 | 25.05 | 861,428 | 127.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.188/preresnet56_cifar100-2505-f879fb4e.npz.log)) |
| PreResNet-110 | 22.67 | 1,736,372 | 255.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.191/preresnet110_cifar100-2267-4e010af0.npz.log)) |
| PreResNet-164(BN) | 20.18 | 1,726,388 | 255.10M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.192/preresnet164bn_cifar100-2018-5228dfbd.npz.log)) |
| ResNeXt-29 (32x4d) | 19.50 | 4,868,004 | 780.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.200/resnext29_32x4d_cifar100-1950-de139852.npz.log)) |
| PyramidNet-110 (a=48) | 20.95 | 1,778,556 | 408.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.186/pyramidnet110_a48_cifar100-2095-b74f12c8.npz.log)) |
| PyramidNet-110 (a=84) | 18.87 | 3,913,536 | 778.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.199/pyramidnet110_a84_cifar100-1887-842b3809.npz.log)) |
| DenseNet-40 (k=12) | 24.90 | 622,360 | 210.82M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.195/densenet40_k12_cifar100-2490-d06839db.npz.log)) |
| DenseNet-100 (k=12) | 19.64 | 4,129,600 | 1,353.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.206/densenet100_k12_cifar100-1964-f04f5920.npz.log)) |
| DenseNet-BC-100 (k=12) | 21.19 | 800,032 | 298.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.208/densenet100_k12_bc_cifar100-2119-a37ebc2a.npz.log)) |
| WRN-16-10 | 18.95 | 17,174,324 | 2,414.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.204/wrn16_10_cifar100-1895-d6e85278.npz.log)) |

[dmlc/gluon-cv]: https://github.com/dmlc/gluon-cv
[tornadomeet/ResNet]: https://github.com/tornadomeet/ResNet
[Cadene/pretrained...pytorch]: https://github.com/Cadene/pretrained-models.pytorch
[ShichenLiu/CondenseNet]: https://github.com/ShichenLiu/CondenseNet
[clavichord93/MENet]: https://github.com/clavichord93/MENet
[clavichord93/FD-MobileNet]: https://github.com/clavichord93/FD-MobileNet
[tensorpack/tensorpack]: https://github.com/tensorpack/tensorpack
[dyhan0920/Pyramid...PyTorch]: https://github.com/dyhan0920/PyramidNet-PyTorch
[zeusees/Mnasnet...Model]: https://github.com/zeusees/Mnasnet-Pretrained-Model
[szagoruyko/diracnets]: https://github.com/szagoruyko/diracnets
[szagoruyko/functional-zoo]: https://github.com/szagoruyko/functional-zoo
[fyu/drn]: https://github.com/fyu/drn
[quark0/darts]: https://github.com/quark0/darts
[homles11/IGCV3]: https://github.com/homles11/IGCV3
[soeaver/AirNet-PyTorch]: https://github.com/soeaver/AirNet-PyTorch
[Jongchan/attention-module]: https://github.com/Jongchan/attention-module
[kevin-ssy/FishNet]: https://github.com/kevin-ssy/FishNet
[ucbdrive/dla]: https://github.com/ucbdrive/dla