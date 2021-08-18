# Deep learning networks

[![Build Status](https://travis-ci.org/osmr/imgclsmob.svg?branch=master)](https://travis-ci.org/osmr/imgclsmob)
[![GitHub License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-2.7%2C3.6%2C3.7-lightgrey.svg)](https://github.com/osmr/imgclsmob)

This repo is used to research convolutional networks primarily for computer vision tasks. For this purpose, the repo
contains (re)implementations of various classification, segmentation, detection, and pose estimation models and scripts
for training/evaluating/converting.

The following frameworks are used:
- MXNet/Gluon ([info](https://mxnet.apache.org)),
- PyTorch ([info](https://pytorch.org)),
- Chainer ([info](https://chainer.org)),
- Keras ([info](https://keras.io)),
- TensorFlow 1.x/2.x ([info](https://www.tensorflow.org)).

For each supported framework, there is a PIP-package containing pure models without auxiliary scripts. List of packages:
- [gluoncv2](https://pypi.org/project/gluoncv2) for Gluon,
- [pytorchcv](https://pypi.org/project/pytorchcv) for PyTorch,
- [chainercv2](https://pypi.org/project/chainercv2) for Chainer,
- [kerascv](https://pypi.org/project/kerascv) for Keras,
- [tensorflowcv](https://pypi.org/project/tensorflowcv) for TensorFlow 1.x,
- [tf2cv](https://pypi.org/project/tf2cv) for TensorFlow 2.x.

Currently, models are mostly implemented on Gluon and then ported to other frameworks. Some models are pretrained on
[ImageNet-1K](http://www.image-net.org), [CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html),
[SVHN](http://ufldl.stanford.edu/housenumbers), [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html),
[Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012), [ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K),
[Cityscapes](https://www.cityscapes-dataset.com), and [COCO](http://cocodataset.org) datasets. All pretrained weights
are loaded automatically during use. See examples of such automatic loading of weights in the corresponding sections of
the documentation dedicated to a particular package:
- [Gluon models](gluon/README.md),
- [PyTorch models](pytorch/README.md),
- [Chainer models](chainer_/README.md),
- [Keras models](keras_/README.md),
- [TensorFlow 1.x models](tensorflow_/README.md),
- [TensorFlow 2.x models](tensorflow2/README.md).

## Installation

To use training/evaluating scripts as well as all models, you need to clone the repository and install dependencies:
```
git clone git@github.com:osmr/imgclsmob.git
pip install -r requirements.txt
```

## Table of implemented classification models

Some remarks:
- `Repo` is an author repository, if it exists.
- `a`, `b`, `c`, `d`, and `e` means the implementation of a model for ImageNet-1K, CIFAR-10, CIFAR-100, SVHN, and CUB-200-2011, respectively.
- `A`, `B`, `C`, `D`, and `E` means having a pre-trained model for corresponding datasets.

| Model | [Gluon](gluon/README.md) | [PyTorch](pytorch/README.md) | [Chainer](chainer_/README.md) | [Keras](keras_/README.md) | [TF](tensorflow_/README.md) | [TF2](tensorflow2/README.md) | Paper | Repo | Year |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AlexNet | A | A | A | A | A | A | [link](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) | [link](https://code.google.com/archive/p/cuda-convnet2) | 2012 |
| ZFNet | A | A | A | A | A | A | [link](https://arxiv.org/abs/1311.2901) | - | 2013 |
| VGG | A | A | A | A | A | A | [link](https://arxiv.org/abs/1409.1556) | - | 2014 |
| BN-VGG | A | A | A | A | A | A | [link](https://arxiv.org/abs/1409.1556) | - | 2015 |
| BN-Inception | A | A | A | - | - | A | [link](https://arxiv.org/abs/1502.03167) | - | 2015 |
| ResNet | ABCDE | ABCDE | ABCDE | A | A | ABCDE | [link](https://arxiv.org/abs/1512.03385) | [link](https://github.com/KaimingHe/deep-residual-networks) | 2015 |
| PreResNet | ABCD | ABCD | ABCD | A | A | ABCD | [link](https://arxiv.org/abs/1603.05027) | [link](https://github.com/facebook/fb.resnet.torch) | 2016 |
| ResNeXt | ABCD | ABCD | ABCD | A | A | ABCD | [link](http://arxiv.org/abs/1611.05431) | [link](https://github.com/facebookresearch/ResNeXt) | 2016 |
| SENet | A | A | A | A | A | A | [link](https://arxiv.org/abs/1709.01507) | [link](https://github.com/hujie-frank/SENet) | 2017 |
| SE-ResNet | ABCDE | ABCDE | ABCDE | A | A | ABCDE | [link](https://arxiv.org/abs/1709.01507) | [link](https://github.com/hujie-frank/SENet) | 2017 |
| SE-PreResNet | ABCD | ABCD | ABCD | A | A | ABCD | [link](https://arxiv.org/abs/1709.01507) | [link](https://github.com/hujie-frank/SENet) | 2017 |
| SE-ResNeXt | A | A | A | A | A | A | [link](https://arxiv.org/abs/1709.01507) | [link](https://github.com/hujie-frank/SENet) | 2017 |
| ResNeSt(A) | A | A | A | - | - | A | [link](https://arxiv.org/abs/2004.08955) | [link](https://github.com/zhanghang1989/ResNeSt) | 2020 |
| IBN-ResNet | A | A | - | - | - | A | [link](https://arxiv.org/abs/1807.09441) | [link](https://github.com/XingangPan/IBN-Net) | 2018 |
| IBN-ResNeXt | A | A | - | - | - | A | [link](https://arxiv.org/abs/1807.09441) | [link](https://github.com/XingangPan/IBN-Net) | 2018 |
| IBN-DenseNet | A | A | - | - | - | A | [link](https://arxiv.org/abs/1807.09441) | [link](https://github.com/XingangPan/IBN-Net) | 2018 |
| AirNet | A | A | A | - | - | A | [link](https://ieeexplore.ieee.org/document/8510896) | [link](https://github.com/soeaver/AirNet-PyTorch) | 2018 |
| AirNeXt | A | A | A | - | - | A | [link](https://ieeexplore.ieee.org/document/8510896) | [link](https://github.com/soeaver/AirNet-PyTorch) | 2018 |
| BAM-ResNet | A | A | A | - | - | A | [link](https://arxiv.org/abs/1807.06514) | [link](https://github.com/Jongchan/attention-module) | 2018 |
| CBAM-ResNet | A | A | A | - | - | A | [link](https://arxiv.org/abs/1807.06521) | [link](https://github.com/Jongchan/attention-module) | 2018 |
| ResAttNet | a | a | a | - | - | - | [link](https://arxiv.org/abs/1704.06904) | [link](https://github.com/fwang91/residual-attention-network) | 2017 |
| SKNet | a | a | a | - | - | - | [link](https://arxiv.org/abs/1903.06586) | [link](https://github.com/implus/SKNet) | 2019 |
| SCNet | A | A | A | - | - | A | [link](http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf) | [link](https://github.com/MCG-NKU/SCNet) | 2020 |
| RegNet | A | A | A | - | - | A | [link](https://arxiv.org/abs/2003.13678) | [link](https://github.com/facebookresearch/pycls) | 2020 |
| DIA-ResNet | aBCD | aBCD | aBCD | - | - | - | [link](https://arxiv.org/abs/1905.10671) | [link](https://github.com/gbup-group/DIANet) | 2019 |
| DIA-PreResNet | aBCD | aBCD | aBCD | - | - | - | [link](https://arxiv.org/abs/1905.10671) | [link](https://github.com/gbup-group/DIANet) | 2019 |
| PyramidNet | ABCD | ABCD | ABCD | - | - | ABCD | [link](https://arxiv.org/abs/1610.02915) | [link](https://github.com/jhkim89/PyramidNet) | 2016 |
| DiracNetV2 | A | A | A | - | - | A | [link](https://arxiv.org/abs/1706.00388) | [link](https://github.com/szagoruyko/diracnets) | 2017 |
| ShaResNet | a | a | a | - | - | - | [link](https://arxiv.org/abs/1702.08782) | [link](https://github.com/aboulch/sharesnet) | 2017 |
| CRU-Net | A | - | - | - | - | - | [link](https://www.ijcai.org/proceedings/2018/88) | [link](https://github.com/cypw/CRU-Net) | 2018 |
| DenseNet | ABCD | ABCD | ABCD | A | A | ABCD | [link](https://arxiv.org/abs/1608.06993) | [link](https://github.com/liuzhuang13/DenseNet) | 2016 |
| CondenseNet | A | A | A | - | - | - | [link](https://arxiv.org/abs/1711.09224) | [link](https://github.com/ShichenLiu/CondenseNet) | 2017 |
| SparseNet | a | a | a | - | - | - | [link](https://arxiv.org/abs/1801.05895) | [link](https://github.com/Lyken17/SparseNet) | 2018 |
| PeleeNet | A | A | A | - | - | A | [link](https://arxiv.org/abs/1804.06882) | [link](https://github.com/Robert-JunWang/Pelee) | 2018 |
| Oct-ResNet | abcd | a | a | - | - | - | [link](https://arxiv.org/abs/1904.05049) | - | 2019 |
| Res2Net | a | - | - | - | - | - | [link](https://arxiv.org/abs/1904.01169) | - | 2019 |
| WRN | ABCD | ABCD | ABCD | - | - | a | [link](https://arxiv.org/abs/1605.07146) | [link](https://github.com/szagoruyko/wide-residual-networks) | 2016 |
| WRN-1bit | BCD | BCD | BCD | - | - | - | [link](https://arxiv.org/abs/1802.08530) | [link](https://github.com/McDonnell-Lab/1-bit-per-weight) | 2018 |
| DRN-C | A | A | A | - | - | A | [link](https://arxiv.org/abs/1705.09914) | [link](https://github.com/fyu/drn) | 2017 |
| DRN-D | A | A | A | - | - | A | [link](https://arxiv.org/abs/1705.09914) | [link](https://github.com/fyu/drn) | 2017 |
| DPN | A | A | A | - | - | A | [link](https://arxiv.org/abs/1707.01629) | [link](https://github.com/cypw/DPNs) | 2017 |
| DarkNet Ref | A | A | A | A | A | A | [link](https://github.com/pjreddie/darknet) | [link](https://github.com/pjreddie/darknet) | - |
| DarkNet Tiny | A | A | A | A | A | A | [link](https://github.com/pjreddie/darknet) | [link](https://github.com/pjreddie/darknet) | - |
| DarkNet-19 | a | a | a | a | a | a | [link](https://github.com/pjreddie/darknet) | [link](https://github.com/pjreddie/darknet) | - |
| DarkNet-53 | A | A | A | A | A | A | [link](https://arxiv.org/abs/1804.02767) | [link](https://github.com/pjreddie/darknet) | 2018 |
| ChannelNet | a | a | a | - | a | - | [link](https://arxiv.org/abs/1809.01330) | [link](https://github.com/HongyangGao/ChannelNets) | 2018 |
| iSQRT-COV-ResNet | a | a | - | - | - | - | [link](https://arxiv.org/abs/1712.01034) | [link](https://github.com/jiangtaoxie/fast-MPN-COV) | 2017 |
| RevNet | - | a | - | - | - | - | [link](https://arxiv.org/abs/1707.04585) | [link](https://github.com/renmengye/revnet-public) | 2017 |
| i-RevNet | A | A | A | - | - | - | [link](https://arxiv.org/abs/1802.07088) | [link](https://github.com/jhjacobsen/pytorch-i-revnet) | 2018 |
| BagNet | A | A | A | - | - | A | [link](https://openreview.net/pdf?id=SkfMWhAqYQ) | [link](https://github.com/wielandbrendel/bag-of-local-features-models) | 2019 |
| DLA | A | A | A | - | - | A | [link](https://arxiv.org/abs/1707.06484) | [link](https://github.com/ucbdrive/dla) | 2017 |
| MSDNet | a | ab | - | - | - | - | [link](https://arxiv.org/abs/1703.09844) | [link](https://github.com/gaohuang/MSDNet) | 2017 |
| FishNet | A | A | A | - | - | - | [link](http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction.pdf) | [link](https://github.com/kevin-ssy/FishNet) | 2018 |
| ESPNetv2 | A | A | A | - | - | - | [link](https://arxiv.org/abs/1811.11431) | [link](https://github.com/sacmehta/ESPNetv2) | 2018 |
| DiCENet | A | A | A | - | - | A | [link](https://arxiv.org/abs/1906.03516) | [link](https://github.com/sacmehta/EdgeNets) | 2019 |
| HRNet | A | A | A | - | - | A | [link](https://arxiv.org/abs/1908.07919) | [link](https://github.com/HRNet/HRNet-Image-Classification) | 2019 |
| VoVNet | A | A | A | - | - | A | [link](https://arxiv.org/abs/1904.09730) | [link](https://github.com/stigma0617/VoVNet.pytorch) | 2019 |
| SelecSLS | A | A | A | - | - | A | [link](https://arxiv.org/abs/1907.00837) | [link](https://github.com/mehtadushy/SelecSLS-Pytorch) | 2019 |
| HarDNet | A | A | A | - | - | A | [link](https://arxiv.org/abs/1909.00948) | [link](https://github.com/PingoLH/Pytorch-HarDNet) | 2019 |
| X-DenseNet | aBCD | aBCD | aBCD | - | - | - | [link](https://arxiv.org/abs/1711.08757) | [link](https://github.com/DrImpossible/Deep-Expander-Networks) | 2017 |
| SqueezeNet | A | A | A | A | A | A | [link](https://arxiv.org/abs/1602.07360) | [link](https://github.com/DeepScale/SqueezeNet) | 2016 |
| SqueezeResNet | A | A | A | A | A | A | [link](https://arxiv.org/abs/1602.07360) | - | 2016 |
| SqueezeNext | A | A | A | A | A | A | [link](https://arxiv.org/abs/1803.10615) | [link](https://github.com/amirgholami/SqueezeNext) | 2018 |
| ShuffleNet | A | A | A | A | A | A | [link](https://arxiv.org/abs/1707.01083) | - | 2017 |
| ShuffleNetV2 | A | A | A | A | A | A | [link](https://arxiv.org/abs/1807.11164) | - | 2018 |
| MENet | A | A | A | A | A | A | [link](https://arxiv.org/abs/1803.09127) | [link](https://github.com/clavichord93/MENet) | 2018 |
| MobileNet | AE | AE | AE | A | A | AE | [link](https://arxiv.org/abs/1704.04861) | [link](https://github.com/tensorflow/models) | 2017 |
| FD-MobileNet | A | A | A | A | A | A | [link](https://arxiv.org/abs/1802.03750) | [link](https://github.com/clavichord93/FD-MobileNet) | 2018 |
| MobileNetV2 | A | A | A | A | A | A | [link](https://arxiv.org/abs/1801.04381) | [link](https://github.com/tensorflow/models) | 2018 |
| MobileNetV3 | A | A | A | A | - | A | [link](https://arxiv.org/abs/1905.02244) | [link](https://github.com/tensorflow/models) | 2019 |
| IGCV3 | A | A | A | A | A | A | [link](https://arxiv.org/abs/1806.00178) | [link](https://github.com/homles11/IGCV3) | 2018 |
| GhostNet | a | a | a | - | - | a | [link](https://arxiv.org/abs/1911.11907) | [link](https://github.com/iamhankai/ghostnet) | 2019 |
| MnasNet | A | A | A | A | A | A | [link](https://arxiv.org/abs/1807.11626) | - | 2018 |
| DARTS | A | A | A | - | - | - | [link](https://arxiv.org/abs/1806.09055) | [link](https://github.com/quark0/darts) | 2018 |
| ProxylessNAS | AE | AE | AE | - | - | AE | [link](https://arxiv.org/abs/1812.00332) | [link](https://github.com/mit-han-lab/ProxylessNAS) | 2018 |
| FBNet-C | A | A | A | - | - | A | [link](https://arxiv.org/abs/1812.03443) | - | 2018 |
| Xception | A | A | A | - | - | A | [link](https://arxiv.org/abs/1610.02357) | [link](https://github.com/fchollet/deep-learning-models) | 2016 |
| InceptionV3 | A | A | A | - | - | A | [link](https://arxiv.org/abs/1512.00567) | [link](https://github.com/tensorflow/models) | 2015 |
| InceptionV4 | A | A | A | - | - | A | [link](https://arxiv.org/abs/1602.07261) | [link](https://github.com/tensorflow/models) | 2016 |
| InceptionResNetV1 | A | A | A | - | - | A | [link](https://arxiv.org/abs/1602.07261) | [link](https://github.com/tensorflow/models) | 2016 |
| InceptionResNetV2 | A | A | A | - | - | A | [link](https://arxiv.org/abs/1602.07261) | [link](https://github.com/tensorflow/models) | 2016 |
| PolyNet | A | A | A | - | - | A | [link](https://arxiv.org/abs/1611.05725) | [link](https://github.com/open-mmlab/polynet) | 2016 |
| NASNet-Large | A | A | A | - | - | A | [link](https://arxiv.org/abs/1707.07012) | [link](https://github.com/tensorflow/models) | 2017 |
| NASNet-Mobile | A | A | A | - | - | A | [link](https://arxiv.org/abs/1707.07012) | [link](https://github.com/tensorflow/models) | 2017 |
| PNASNet-Large | A | A | A | - | - | A | [link](https://arxiv.org/abs/1712.00559) | [link](https://github.com/tensorflow/models) | 2017 |
| SPNASNet | A | A | A | - | - | A | [link](https://arxiv.org/abs/1904.02877) | [link](https://github.com/dstamoulis/single-path-nas) | 2019 |
| EfficientNet | A | A | A | A | - | A | [link](https://arxiv.org/abs/1905.11946) | [link](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) | 2019 |
| MixNet | A | A | A | - | - | A | [link](https://arxiv.org/abs/1907.09595) | [link](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet) | 2019 |
| NIN | BCD | BCD | BCD | - | - | - | [link](https://arxiv.org/abs/1312.4400) | [link](https://gist.github.com/mavenlin/e56253735ef32c3c296d) | 2013 |
| RoR-3 | BCD | BCD | BCD | - | - | - | [link](https://arxiv.org/abs/1608.02908) | - | 2016 |
| RiR | BCD | BCD | BCD | - | - | - | [link](https://arxiv.org/abs/1603.08029) | - | 2016 |
| ResDrop-ResNet | bcd | bcd | bcd | - | - | - | [link](https://arxiv.org/abs/1603.09382) | [link](https://github.com/yueatsprograms/Stochastic_Depth) | 2016 |
| Shake-Shake-ResNet | BCD | BCD | BCD | - | - | - | [link](https://arxiv.org/abs/1705.07485) | [link](https://github.com/xgastaldi/shake-shake) | 2017 |
| ShakeDrop-ResNet | bcd | bcd | bcd | - | - | - | [link](https://arxiv.org/abs/1802.02375) | - | 2018 |
| FractalNet | bc | bc | - | - | - | - | [link](https://arxiv.org/abs/1605.07648) | [link](https://github.com/gustavla/fractalnet) | 2016 |
| NTS-Net | E | E | E | - | - | - | [link](https://arxiv.org/abs/1809.00287) | [link](https://github.com/yangze0930/NTS-Net) | 2018 |

## Table of implemented segmentation models

Some remarks:
- `a/A` corresponds to Pascal VOC2012.
- `b/B` corresponds to ADE20K.
- `c/C` corresponds to Cityscapes.
- `d/D` corresponds to COCO.
- `e/E` corresponds to CelebAMask-HQ.

| Model | [Gluon](gluon/README.md) | [PyTorch](pytorch/README.md) | [Chainer](chainer_/README.md) | [Keras](keras_/README.md) | [TF](tensorflow_/README.md)  | [TF2](tensorflow_/README.md) | Paper | Repo | Year |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PSPNet | ABCD | ABCD | ABCD | - | - | ABCD | [link](https://arxiv.org/abs/1612.01105) | - | 2016 |
| DeepLabv3 | ABcD | ABcD | ABcD | - | - | ABcD | [link](https://arxiv.org/abs/1706.05587) | - | 2017 |
| FCN-8s(d) | ABcD | ABcD | ABcD | - | - | ABcD | [link](https://arxiv.org/abs/1411.4038) | - | 2014 |
| ICNet | C | C | C | - | - | C | [link](https://arxiv.org/abs/1704.08545) | [link](https://github.com/hszhao/ICNet) | 2017 |
| SINet | C | C | C | - | - | c | [link](https://arxiv.org/abs/1911.09099) | [link](https://github.com/clovaai/c3_sinet) | 2019 |
| BiSeNet | e | e | e | - | - | e | [link](https://arxiv.org/abs/1808.00897) | - | 2018 |
| DANet | C | C | C | - | - | C | [link](https://arxiv.org/abs/1809.02983) | [link](https://github.com/junfu1115/DANet) | 2018 |
| Fast-SCNN | C | C | C | - | - | C | [link](https://arxiv.org/abs/1902.04502) | - | 2019 |
| CGNet | c | c | c | - | - | c | [link](https://arxiv.org/abs/1811.08201) | [link](https://github.com/wutianyiRosun/CGNet) | 2018 |
| DABNet | c | c | c | - | - | c | [link](https://arxiv.org/abs/1907.11357) | [link](https://github.com/Reagan1311/DABNet) | 2019 |
| FPENet | c | c | c | - | - | c | [link](https://arxiv.org/abs/1909.08599) | - | 2019 |
| ContextNet | - | c | - | - | - | - | [link](https://arxiv.org/abs/1805.04554) | - | 2018 |
| LEDNet | c | c | c | - | - | c | [link](https://arxiv.org/abs/1905.02423) | - | 2019 |
| ESNet | - | c | - | - | - | - | [link](https://arxiv.org/abs/1906.09826) | - | 2019 |
| EDANet | - | c | - | - | - | - | [link](https://arxiv.org/abs/1809.06323) | [link](https://github.com/shaoyuanlo/EDANet) | 2018 |
| ENet | - | c | - | - | - | - | [link](https://arxiv.org/abs/1606.02147) | - | 2016 |
| ERFNet | - | c | - | - | - | - | [link](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf) | - | 2017 |
| LinkNet | - | c | - | - | - | - | [link](https://arxiv.org/abs/1707.03718) | - | 2017 |
| SegNet | - | c | - | - | - | - | [link](https://arxiv.org/abs/1511.00561) | - | 2015 |
| U-Net | - | c | - | - | - | - | [link](https://arxiv.org/abs/1505.04597) | - | 2015 |
| SQNet | - | c | - | - | - | - | [link](https://openreview.net/pdf?id=S1uHiFyyg) | - | 2016 |

## Table of implemented object detection models

Some remarks:
- `a/A` corresponds to COCO.

| Model | [Gluon](gluon/README.md) | [PyTorch](pytorch/README.md) | [Chainer](chainer_/README.md) | [Keras](keras_/README.md) | [TF](tensorflow_/README.md)  | [TF2](tensorflow2/README.md) | Paper | Repo | Year |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CenterNet | a | a | a | - | - | a | [link](https://arxiv.org/abs/1904.07850) | [link](https://github.com/xingyizhou/CenterNet) | 2019 |

## Table of implemented human pose estimation models

Some remarks:
- `a/A` corresponds to COCO.

| Model | [Gluon](gluon/README.md) | [PyTorch](pytorch/README.md) | [Chainer](chainer_/README.md) | [Keras](keras_/README.md) | [TF](tensorflow_/README.md)  | [TF2](tensorflow2/README.md) | Paper | Repo | Year |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AlphaPose | A | A | A | - | - | A | [link](https://arxiv.org/abs/1612.00137) | [link](https://github.com/MVIG-SJTU/AlphaPose) | 2016 |
| SimplePose | A | A | A | - | - | A | [link](https://arxiv.org/abs/1804.06208) | [link](https://github.com/microsoft/human-pose-estimation.pytorch) | 2018 |
| SimplePose(Mobile) | A | A | A | - | - | A | [link](https://arxiv.org/abs/1804.06208) | - | 2018 |
| Lightweight OpenPose | A | A | A | - | - | A | [link](https://arxiv.org/abs/1811.12004) | [link](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch) | 2018 |
| IBPPose | A | A | A | - | - | A | [link](https://arxiv.org/abs/1911.10529) | [link](https://github.com/jialee93/Improved-Body-Parts) | 2019 |

## Table of implemented automatic speech recognition models

Some remarks:
- `a/A` corresponds to LibriSpeech.
- `b/B` corresponds to Mozilla Common Voice.

| Model | [Gluon](gluon/README.md) | [PyTorch](pytorch/README.md) | [Chainer](chainer_/README.md) | [Keras](keras_/README.md) | [TF](tensorflow_/README.md)  | [TF2](tensorflow2/README.md) | Paper | Repo | Year |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Jasper DR | AB | AB | ab | - | - | ab | [link](https://arxiv.org/abs/1904.03288) | [link](https://github.com/NVIDIA/NeMo) | 2019 |
| QuartzNet | AB | AB | ab | - | - | ab | [link](https://arxiv.org/abs/1910.10261) | [link](https://github.com/NVIDIA/NeMo) | 2019 |
