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

## Table of implemented models (...in the process of filling...)

| Model | Gluon | PyTorch | Chainer | Keras | TensorFlow | Paper | Repo | Year |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AlexNet | A+ | A+ | A+ | A+ | A+ | [link](https://arxiv.org/abs/1404.5997) | - | - |
| ZFNet | A | A | A | - | - | [link](https://arxiv.org/abs/1311.2901) | - | - |
| VGG | A+ | A+ | A+ | A+ | A+ | [link](https://arxiv.org/abs/1409.1556) | - | - |
| BN-VGG | A+ | A+ | A+ | A+ | A+ | [link](https://arxiv.org/abs/1409.1556) | - | - |
| BN-Inception | A+ | A+ | A+ | - | - | [link](https://arxiv.org/abs/1502.03167) | - | - |
| ResNet | A+B+ | A+B+ | A+B+ | A+ | A+ | [link](https://arxiv.org/abs/1512.03385) | - | - |
| PreResNet | A+B+ | A+B+ | A+B+ | A+ | A+ | [link](https://arxiv.org/abs/1603.05027) | - | - |
| ResNeXt | A+B+ | A+B+ | A+B+ | A+ | A+ | [link](http://arxiv.org/abs/1611.05431) | - | - |
| SENet | A+ | A+ | A+ | A+ | A+ | [link](https://arxiv.org/abs/1709.01507) | - | - |
| SE-ResNet | A | - | - | - | - | [link](https://arxiv.org/abs/1709.01507) | - | - |
| SE-PreResNet | A | - | - | - | - | [link](https://arxiv.org/abs/1709.01507) | - | - |
| SE-ResNeXt | A | - | - | - | - | [link](https://arxiv.org/abs/1709.01507) | - | - |
| IBN-ResNet | A | - | - | - | - | [link](https://arxiv.org/abs/1807.09441) | - | - |
| IBN-ResNeXt | A | - | - | - | - | [link](https://arxiv.org/abs/1807.09441) | - | - |
| IBN-DenseNet | A | - | - | - | - | [link](https://arxiv.org/abs/1807.09441) | - | - |
| AirNet | A | - | - | - | - | [link](https://ieeexplore.ieee.org/document/8510896) | - | - |
| AirNeXt | A | - | - | - | - | [link](https://ieeexplore.ieee.org/document/8510896) | - | - |
| BAM-ResNet | A | - | - | - | - | [link](https://arxiv.org/abs/1807.06514) | - | - |
| CBAM-ResNet | A | - | - | - | - | [link](https://arxiv.org/abs/1807.06521) | - | - |
| ResAttNet | A | - | - | - | - | [link](https://arxiv.org/abs/1704.06904) | - | - |
| PyramidNet | A | - | - | - | - | [link](https://arxiv.org/abs/1610.02915) | - | - |
| DiracNetV2 | A | - | - | - | - | [link](https://arxiv.org/abs/1706.00388) | - | - |
| DenseNet | A | - | - | - | - | [link](https://arxiv.org/abs/1608.06993) | - | - |
| CondenseNet | A | - | - | - | - | [link](https://arxiv.org/abs/1711.09224) | - | - |
| SparseNet | A | - | - | - | - | [link](https://arxiv.org/abs/1801.05895) | - | - |
| PeleeNet | A | - | - | - | - | [link](https://arxiv.org/abs/1804.06882) | - | - |
| WRN | A | - | - | - | - | [link](https://arxiv.org/abs/1605.07146) | - | - |
| DRN-C | A | - | - | - | - | [link](https://arxiv.org/abs/1705.09914) | - | - |
| DRN-D | A | - | - | - | - | [link](https://arxiv.org/abs/1705.09914) | - | - |
| DPN | A | - | - | - | - | [link](https://arxiv.org/abs/1707.01629) | - | - |
| DarkNet Ref | A | - | - | - | - | [link](https://github.com/pjreddie/darknet) | - | - |
| DarkNet Tiny | A | - | - | - | - | [link](https://github.com/pjreddie/darknet) | - | - |
| DarkNet-19 | A | - | - | - | - | [link](https://github.com/pjreddie/darknet) | - | - |
| DarkNet-53 | A | - | - | - | - | [link](https://arxiv.org/abs/1804.02767) | - | - |
| ChannelNet | A | - | - | - | - | [link](https://arxiv.org/abs/1809.01330) | - | - |
| MSDNet | A | - | - | - | - | [link](https://arxiv.org/abs/1703.09844) | - | - |
| FishNet | A | - | - | - | - | [link](http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction.pdf) | - | - |
| SqueezeNet | A | - | - | - | - | [link](https://arxiv.org/abs/1602.07360) | - | - |
| SqueezeResNet | A | - | - | - | - | [link](https://arxiv.org/abs/1602.07360) | - | - |
| SqueezeNext | A | - | - | - | - | [link](https://arxiv.org/abs/1803.10615) | - | - |
| ShuffleNet | A | - | - | - | - | [link](https://arxiv.org/abs/1707.01083) | - | - |
| ShuffleNetV2 | A | - | - | - | - | [link](https://arxiv.org/abs/1807.11164) | - | - |
| MENet | A | - | - | - | - | [link](https://arxiv.org/abs/1803.09127) | - | - |
| MobileNet | A | - | - | - | - | [link](https://arxiv.org/abs/1704.04861) | - | - |
| FD-MobileNet | A | - | - | - | - | [link](https://arxiv.org/abs/1802.03750) | - | - |
| MobileNetV2 | A | - | - | - | - | [link](https://arxiv.org/abs/1801.04381) | - | - |
| IGCV3 | A | - | - | - | - | [link](https://arxiv.org/abs/1806.00178) | - | - |
| MnasNet | A | - | - | - | - | [link](https://arxiv.org/abs/1807.11626) | - | - |
| DARTS | A | - | - | - | - | [link](https://arxiv.org/abs/1806.09055) | - | - |
| Xception | A | - | - | - | - | [link](https://arxiv.org/abs/1610.02357) | - | - |
| InceptionV3 | A | - | - | - | - | [link](https://arxiv.org/abs/1512.00567) | - | - |
| InceptionV4 | A | - | - | - | - | [link](https://arxiv.org/abs/1602.07261) | - | - |
| InceptionResNetV2 | A | - | - | - | - | [link](https://arxiv.org/abs/1602.07261) | - | - |
| PolyNet | A | - | - | - | - | [link](https://arxiv.org/abs/1611.05725) | - | - |
| NASNet | A | - | - | - | - | [link](https://arxiv.org/abs/1707.07012) | - | - |
| PNASNet | A | - | - | - | - | [link](https://arxiv.org/abs/1712.00559) | - | - |
