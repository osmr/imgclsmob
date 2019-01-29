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
ImageNet-1K and CIFAR-10/100 datasets. All pretrained weights are loaded automatically during use. See examples of such
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

Some remarks:
- `Repo` is an author repository, if it exists.
- `A`, `B`, and `C` means the implementation of a model for ImageNet-1K, CIFAR-10, and CIFAR-100, respectively.
- `A+`, `B+`, and `C+` means having a pre-trained model for ImageNet-1K, CIFAR-10, and CIFAR-100, respectively.

| Model | [Gluon](gluon/README.md) | [PyTorch](pytorch/README.md) | [Chainer](chainer_/README.md) | [Keras](keras_/README.md) | [TensorFlow](tensorflow_/README.md) | Paper | Repo | Year |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AlexNet | A+ | A+ | A+ | A+ | A+ | [link](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) | [link](https://code.google.com/archive/p/cuda-convnet2) | 2012 |
| ZFNet | A | A | A | - | - | [link](https://arxiv.org/abs/1311.2901) | - | 2013 |
| NIN | B+C+ | B+C+ | B+C+ | - | - | [link](https://arxiv.org/abs/1312.4400) | [link](https://gist.github.com/mavenlin/e56253735ef32c3c296d) | 2013 |
| VGG | A+ | A+ | A+ | A+ | A+ | [link](https://arxiv.org/abs/1409.1556) | - | 2014 |
| BN-VGG | A+ | A+ | A+ | A+ | A+ | [link](https://arxiv.org/abs/1409.1556) | - | 2015 |
| BN-Inception | A+ | A+ | A+ | - | - | [link](https://arxiv.org/abs/1502.03167) | - | 2015 |
| ResNet | A+B+C+ | A+B+C+ | A+B+C+ | A+ | A+ | [link](https://arxiv.org/abs/1512.03385) | [link](https://github.com/KaimingHe/deep-residual-networks) | 2015 |
| PreResNet | A+B+C+ | A+B+C+ | A+B+C+ | A+ | A+ | [link](https://arxiv.org/abs/1603.05027) | [link](https://github.com/facebook/fb.resnet.torch) | 2016 |
| ResNeXt | A+B+ | A+B+ | A+B+ | A+ | A+ | [link](http://arxiv.org/abs/1611.05431) | [link](https://github.com/facebookresearch/ResNeXt) | 2016 |
| SENet | A+ | A+ | A+ | A+ | A+ | [link](https://arxiv.org/abs/1709.01507) | [link](https://github.com/hujie-frank/SENet) | 2017 |
| SE-ResNet | A+ | A+ | A+ | A+ | A+ | [link](https://arxiv.org/abs/1709.01507) | [link](https://github.com/hujie-frank/SENet) | 2017 |
| SE-PreResNet | A | A | A | A | A | [link](https://arxiv.org/abs/1709.01507) | [link](https://github.com/hujie-frank/SENet) | 2017 |
| SE-ResNeXt | A+ | A+ | A+ | A+ | A+ | [link](https://arxiv.org/abs/1709.01507) | [link](https://github.com/hujie-frank/SENet) | 2017 |
| IBN-ResNet | A+ | A+ | - | - | - | [link](https://arxiv.org/abs/1807.09441) | [link](https://github.com/XingangPan/IBN-Net) | 2018 |
| IBN-ResNeXt | A+ | A+ | - | - | - | [link](https://arxiv.org/abs/1807.09441) | [link](https://github.com/XingangPan/IBN-Net) | 2018 |
| IBN-DenseNet | A+ | A+ | - | - | - | [link](https://arxiv.org/abs/1807.09441) | [link](https://github.com/XingangPan/IBN-Net) | 2018 |
| AirNet | A+ | A+ | A+ | - | - | [link](https://ieeexplore.ieee.org/document/8510896) | [link](https://github.com/soeaver/AirNet-PyTorch) | 2018 |
| AirNeXt | A+ | A+ | A+ | - | - | [link](https://ieeexplore.ieee.org/document/8510896) | [link](https://github.com/soeaver/AirNet-PyTorch) | 2018 |
| BAM-ResNet | A+ | A+ | A+ | - | - | [link](https://arxiv.org/abs/1807.06514) | [link](https://github.com/Jongchan/attention-module) | 2018 |
| CBAM-ResNet | A+ | A+ | A+ | - | - | [link](https://arxiv.org/abs/1807.06521) | [link](https://github.com/Jongchan/attention-module) | 2018 |
| ResAttNet | A | A | A | - | - | [link](https://arxiv.org/abs/1704.06904) | [link](https://github.com/fwang91/residual-attention-network) | 2017 |
| PyramidNet | A+B+C+ | A+B+C+ | A+B+C+ | - | - | [link](https://arxiv.org/abs/1610.02915) | [link](https://github.com/jhkim89/PyramidNet) | 2016 |
| DiracNetV2 | A+ | A+ | A+ | - | - | [link](https://arxiv.org/abs/1706.00388) | [link](https://github.com/szagoruyko/diracnets) | 2017 |
| CRU-Net | A+ | - | - | - | - | [link](https://www.ijcai.org/proceedings/2018/88) | [link](https://github.com/cypw/CRU-Net) | 2018 |
| DenseNet | A+B+C+ | A+B+C+ | A+B+C+ | A+ | A+ | [link](https://arxiv.org/abs/1608.06993) | [link](https://github.com/liuzhuang13/DenseNet) | 2016 |
| CondenseNet | A+ | A+ | A+ | - | - | [link](https://arxiv.org/abs/1711.09224) | [link](https://github.com/ShichenLiu/CondenseNet) | 2017 |
| SparseNet | A | A | A | - | - | [link](https://arxiv.org/abs/1801.05895) | [link](https://github.com/Lyken17/SparseNet) | 2018 |
| PeleeNet | A+ | A+ | A+ | - | - | [link](https://arxiv.org/abs/1804.06882) | [link](https://github.com/Robert-JunWang/Pelee) | 2018 |
| WRN | A+B+ | A+B+ | A+B+ | - | - | [link](https://arxiv.org/abs/1605.07146) | [link](https://github.com/szagoruyko/wide-residual-networks) | 2016 |
| DRN-C | A+ | A+ | A+ | - | - | [link](https://arxiv.org/abs/1705.09914) | [link](https://github.com/fyu/drn) | 2017 |
| DRN-D | A+ | A+ | A+ | - | - | [link](https://arxiv.org/abs/1705.09914) | [link](https://github.com/fyu/drn) | 2017 |
| DPN | A+ | A+ | A+ | - | - | [link](https://arxiv.org/abs/1707.01629) | [link](https://github.com/cypw/DPNs) | 2017 |
| DarkNet Ref | A+ | A+ | A+ | A+ | A+ | [link](https://github.com/pjreddie/darknet) | [link](https://github.com/pjreddie/darknet) | - |
| DarkNet Tiny | A+ | A+ | A+ | A+ | A+ | [link](https://github.com/pjreddie/darknet) | [link](https://github.com/pjreddie/darknet) | - |
| DarkNet-19 | A | A | A | A | A | [link](https://github.com/pjreddie/darknet) | [link](https://github.com/pjreddie/darknet) | - |
| DarkNet-53 | A+ | A+ | A+ | A+ | A+ | [link](https://arxiv.org/abs/1804.02767) | [link](https://github.com/pjreddie/darknet) | 2018 |
| ChannelNet | A | A | A | - | A | [link](https://arxiv.org/abs/1809.01330) | [link](https://github.com/HongyangGao/ChannelNets) | 2018 |
| MSDNet | A | AB | - | - | - | [link](https://arxiv.org/abs/1703.09844) | [link](https://github.com/gaohuang/MSDNet) | 2017 |
| FishNet | A+ | A+ | A+ | - | - | [link](http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction.pdf) | [link](https://github.com/kevin-ssy/FishNet) | 2018 |
| SqueezeNet | A+ | A+ | A+ | A+ | A+ | [link](https://arxiv.org/abs/1602.07360) | [link](https://github.com/DeepScale/SqueezeNet) | 2016 |
| SqueezeResNet | A+ | A+ | A+ | A+ | A+ | [link](https://arxiv.org/abs/1602.07360) | - | 2016 |
| SqueezeNext | A+ | A+ | A+ | A+ | A+ | [link](https://arxiv.org/abs/1803.10615) | [link](https://github.com/amirgholami/SqueezeNext) | 2018 |
| ShuffleNet | A+ | A+ | A+ | A+ | A+ | [link](https://arxiv.org/abs/1707.01083) | - | 2017 |
| ShuffleNetV2 | A+ | A+ | A+ | A+ | A+ | [link](https://arxiv.org/abs/1807.11164) | - | 2018 |
| MENet | A+ | A+ | A+ | A+ | A+ | [link](https://arxiv.org/abs/1803.09127) | [link](https://github.com/clavichord93/MENet) | 2018 |
| MobileNet | A+ | A+ | A+ | A+ | A+ | [link](https://arxiv.org/abs/1704.04861) | [link](https://github.com/tensorflow/models) | 2017 |
| FD-MobileNet | A+ | A+ | A+ | A+ | A+ | [link](https://arxiv.org/abs/1802.03750) | [link](https://github.com/clavichord93/FD-MobileNet) | 2018 |
| MobileNetV2 | A+ | A+ | A+ | A+ | A+ | [link](https://arxiv.org/abs/1801.04381) | [link](https://github.com/tensorflow/models) | 2018 |
| IGCV3 | A+ | A+ | A+ | A+ | A+ | [link](https://arxiv.org/abs/1806.00178) | [link](https://github.com/homles11/IGCV3) | 2018 |
| MnasNet | A+ | A+ | A+ | A+ | A+ | [link](https://arxiv.org/abs/1807.11626) | - | 2018 |
| DARTS | A+ | A+ | A+ | - | - | [link](https://arxiv.org/abs/1806.09055) | [link](https://github.com/quark0/darts) | 2018 |
| Xception | A+ | A+ | A+ | - | - | [link](https://arxiv.org/abs/1610.02357) | [link](https://github.com/fchollet/deep-learning-models) | 2016 |
| InceptionV3 | A+ | A+ | A+ | - | - | [link](https://arxiv.org/abs/1512.00567) | [link](https://github.com/tensorflow/models) | 2015 |
| InceptionV4 | A+ | A+ | A+ | - | - | [link](https://arxiv.org/abs/1602.07261) | [link](https://github.com/tensorflow/models) | 2016 |
| InceptionResNetV2 | A+ | A+ | A+ | - | - | [link](https://arxiv.org/abs/1602.07261) | [link](https://github.com/tensorflow/models) | 2016 |
| PolyNet | A+ | A+ | A+ | - | - | [link](https://arxiv.org/abs/1611.05725) | [link](https://github.com/open-mmlab/polynet) | 2016 |
| NASNet-Large | A+ | A+ | A+ | - | - | [link](https://arxiv.org/abs/1707.07012) | [link](https://github.com/tensorflow/models) | 2017 |
| NASNet-Mobile | A+ | A+ | A+ | - | - | [link](https://arxiv.org/abs/1707.07012) | [link](https://github.com/tensorflow/models) | 2017 |
| PNASNet-Large | A+ | A+ | A+ | - | - | [link](https://arxiv.org/abs/1712.00559) | [link](https://github.com/tensorflow/models) | 2017 |
