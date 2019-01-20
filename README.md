# Large-scale image classification networks for embedded systems

[![Build Status](https://travis-ci.org/osmr/imgclsmob.svg?branch=master)](https://travis-ci.org/osmr/imgclsmob)
[![GitHub License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-2.7%2C3.6-lightgrey.svg)](https://github.com/osmr/imgclsmob)

This repo is used to research large-scale image classification models for embedded systems. For this purpose,
the repo contains (re)implementations of various classification models and scripts for training/evaluating/converting.

The following frameworks are used:
- MXNet/Gluon ([info](https://mxnet.apache.org)),
- PyTorch ([info](https://pytorch.org)),
- Chainer ([info](https://chainer.org)),
- Keras with MXNet backend ([info](https://github.com/awslabs/keras-apache-mxnet)),
- TensorFlow ([info](https://www.tensorflow.org)).

For each supported framework, there is a PIP-package containing pure models without auxiliary scripts. List of packages:
- [gluoncv2](https://pypi.org/project/gluoncv2) for Gluon,
- [pytorchcv](https://pypi.org/project/pytorchcv) for PyTorch,
- [chainercv2](https://pypi.org/project/chainercv2) for Chainer,
- [kerascv](https://pypi.org/project/kerascv) for Keras-MXNet,
- [tensorflowcv](https://pypi.org/project/tensorflowcv) for TensorFlow.

Currently, models are mostly implemented on Gluon and then ported to other frameworks. Some models are pretrained on
ImageNet-1K and CIFAR-10 datasets. All pretrained weights are loaded automatically during use. See examples of such
automatic loading of weights in the corresponding sections of the documentation dedicated to a particular package:
- [Gluon models](gluon/README.md),
- [PyTorch models](pytorch/README.md),
- [Chainer models](chainer_/README.md),
- [Keras models](keras_/README.md),
- [TensorFlow models](tensorflow_/README.md).

## Installation

To use training/evaluating scripts as well as all models, you need to clone the repository and install dependencies:
```
git clone git@github.com:osmr/imgclsmob.git
pip install -r requirements.txt
```

## Table of implemented models

| Name | Gluon | PyTorch | Chainer | Keras-MXNet | TensorFlow | Paper | Repo | Year |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AlexNet | - | - | - | - | - | ['One weird trick for parallelizing convolutional neural networks'](https://arxiv.org/abs/1404.5997) | - | - |
| ZFNet | - | - | - | - | - | ['Visualizing and Understanding Convolutional Networks'](https://arxiv.org/abs/1311.2901) | - | - |
| VGG/BN-VGG | - | - | - | - | - | ['Very Deep Convolutional Networks for Large-Scale Image Recognition'](https://arxiv.org/abs/1409.1556) | - | - |
| BN-Inception | - | - | - | - | - | ['Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift'](https://arxiv.org/abs/1502.03167) | - | - |
| ResNet | - | - | - | - | - | ['Deep Residual Learning for Image Recognition'](https://arxiv.org/abs/1512.03385) | - | - |
| PreResNet | - | - | - | - | - | ['Identity Mappings in Deep Residual Networks'](https://arxiv.org/abs/1603.05027) | - | - |
| ResNeXt | - | - | - | - | - | ['Aggregated Residual Transformations for Deep Neural Networks'](http://arxiv.org/abs/1611.05431) | - | - |
| SENet/SE-ResNet/SE-PreResNet/SE-ResNeXt | - | - | - | - | - | ['Squeeze-and-Excitation Networks'](https://arxiv.org/abs/1709.01507) | - | - |
| IBN-ResNet/IBN-ResNeXt/IBN-DenseNet | - | - | - | - | - | ['Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net'](https://arxiv.org/abs/1807.09441) | - | - |
| AirNet/AirNeXt | - | - | - | - | - | ['Attention Inspiring Receptive-Fields Network for Learning Invariant Representations'](https://ieeexplore.ieee.org/document/8510896) | - | - |
| BAM-ResNet | - | - | - | - | - | ['BAM: Bottleneck Attention Module'](https://arxiv.org/abs/1807.06514) | - | - |
| CBAM-ResNet | - | - | - | - | - | ['CBAM: Convolutional Block Attention Module'](https://arxiv.org/abs/1807.06521) | - | - |
| ResAttNet | - | - | - | - | - | ['Residual Attention Network for Image Classification'](https://arxiv.org/abs/1704.06904) | - | - |
| PyramidNet | - | - | - | - | - | ['Deep Pyramidal Residual Networks'](https://arxiv.org/abs/1610.02915) | - | - |
| DiracNetV2 | - | - | - | - | - | ['DiracNets: Training Very Deep Neural Networks Without Skip-Connections'](https://arxiv.org/abs/1706.00388) | - | - |
| DenseNet | - | - | - | - | - | ['Densely Connected Convolutional Networks'](https://arxiv.org/abs/1608.06993) | - | - |
| CondenseNet | - | - | - | - | - | ['CondenseNet: An Efficient DenseNet using Learned Group Convolutions'](https://arxiv.org/abs/1711.09224) | - | - |
| SparseNet | - | - | - | - | - | ['Sparsely Aggregated Convolutional Networks'](https://arxiv.org/abs/1801.05895) | - | - |
| PeleeNet | - | - | - | - | - | ['Pelee: A Real-Time Object Detection System on Mobile Devices'](https://arxiv.org/abs/1804.06882) | - | - |
| WRN | - | - | - | - | - | ['Wide Residual Networks'](https://arxiv.org/abs/1605.07146) | - | - |
| DRN-C/DRN-D | - | - | - | - | - | ['Dilated Residual Networks'](https://arxiv.org/abs/1705.09914)) | - | - |
| DPN | - | - | - | - | - | ['Dual Path Networks'](https://arxiv.org/abs/1707.01629) | - | - |
| DarkNet Ref/Tiny/19 | - | - | - | - | - | ['Darknet: Open source neural networks in c'](https://github.com/pjreddie/darknet) | - | - |
| DarkNet-53 | - | - | - | - | - | ['YOLOv3: An Incremental Improvement'](https://arxiv.org/abs/1804.02767) | - | - |
| ChannelNet | - | - | - | - | - | ['ChannelNets: Compact and Efficient Convolutional Neural Networks via Channel-Wise Convolutions'](https://arxiv.org/abs/1809.01330) | - | - |
| MSDNet | - | - | - | - | - | ['Multi-Scale Dense Networks for Resource Efficient Image Classification'](https://arxiv.org/abs/1703.09844) | - | - |
| FishNet | - | - | - | - | - | ['FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction'](http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction.pdf) | - | - |
| SqueezeNet/SqueezeResNet | - | - | - | - | - | ['SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size'](https://arxiv.org/abs/1602.07360) | - | - |
| SqueezeNext | - | - | - | - | - | ['SqueezeNext: Hardware-Aware Neural Network Design'](https://arxiv.org/abs/1803.10615) | - | - |
| ShuffleNet | - | - | - | - | - | ['ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices'](https://arxiv.org/abs/1707.01083) | - | - |
| ShuffleNetV2 | - | - | - | - | - | ['ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design'](https://arxiv.org/abs/1807.11164) | - | - |
| MENet | - | - | - | - | - | ['Merging and Evolution: Improving Convolutional Neural Networks for Mobile Applications'](https://arxiv.org/abs/1803.09127) | - | - |
| MobileNet | - | - | - | - | - | ['MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications'](https://arxiv.org/abs/1704.04861) | - | - |
| FD-MobileNet | - | - | - | - | - | ['FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy'](https://arxiv.org/abs/1802.03750) | - | - |
| MobileNetV2 | - | - | - | - | - | ['MobileNetV2: Inverted Residuals and Linear Bottlenecks'](https://arxiv.org/abs/1801.04381) | - | - |
| IGCV3 | - | - | - | - | - | ['IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks'](https://arxiv.org/abs/1806.00178) | - | - |
| MnasNet | - | - | - | - | - | ['MnasNet: Platform-Aware Neural Architecture Search for Mobile'](https://arxiv.org/abs/1807.11626) | - | - |
| DARTS | - | - | - | - | - | ['DARTS: Differentiable Architecture Search'](https://arxiv.org/abs/1806.09055) | - | - |
| Xception | - | - | - | - | - | ['Xception: Deep Learning with Depthwise Separable Convolutions'](https://arxiv.org/abs/1610.02357) | - | - |
| InceptionV3 | - | - | - | - | - | ['Rethinking the Inception Architecture for Computer Vision'](https://arxiv.org/abs/1512.00567) | - | - |
| InceptionV4/InceptionResNetV2 | - | - | - | - | - | ['Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning'](https://arxiv.org/abs/1602.07261) | - | - |
| PolyNet | - | - | - | - | - | ['PolyNet: A Pursuit of Structural Diversity in Very Deep Networks'](https://arxiv.org/abs/1611.05725) | - | - |
| NASNet | - | - | - | - | - | ['Learning Transferable Architectures for Scalable Image Recognition'](https://arxiv.org/abs/1707.07012) | - | - |
| PNASNet | - | - | - | - | - | ['Progressive Neural Architecture Search'](https://arxiv.org/abs/1712.00559) | - | - |
