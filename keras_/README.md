# Large-scale image classification models on Keras

[![PyPI](https://img.shields.io/pypi/v/kerascv.svg)](https://pypi.python.org/pypi/kerascv)
[![Downloads](https://pepy.tech/badge/kerascv)](https://pepy.tech/project/kerascv)

This is a collection of large-scale image classification models. Many of them are pretrained on
[ImageNet-1K](http://www.image-net.org) dataset and loaded automatically during use. All pretrained models require the
same ordinary normalization. Scripts for training/evaluating/converting models are in the
[`imgclsmob`](https://github.com/osmr/imgclsmob) repo.

## List of implemented models

- AlexNet (['One weird trick for parallelizing convolutional neural networks'](https://arxiv.org/abs/1404.5997))
- VGG/BN-VGG (['Very Deep Convolutional Networks for Large-Scale Image Recognition'](https://arxiv.org/abs/1409.1556))
- ResNet (['Deep Residual Learning for Image Recognition'](https://arxiv.org/abs/1512.03385))
- PreResNet (['Identity Mappings in Deep Residual Networks'](https://arxiv.org/abs/1603.05027))
- ResNeXt (['Aggregated Residual Transformations for Deep Neural Networks'](http://arxiv.org/abs/1611.05431))
- SENet/SE-ResNet/SE-PreResNet/SE-ResNeXt (['Squeeze-and-Excitation Networks'](https://arxiv.org/abs/1709.01507))
- DenseNet (['Densely Connected Convolutional Networks'](https://arxiv.org/abs/1608.06993))
- DarkNet Ref/Tiny/19 (['Darknet: Open source neural networks in c'](https://github.com/pjreddie/darknet))
- DarkNet-53 (['YOLOv3: An Incremental Improvement'](https://arxiv.org/abs/1804.02767))
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
- EfficientNet (['EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks'](https://arxiv.org/abs/1905.11946))

## Installation

To use the models in your project, simply install the `kerascv` package with desired backend. For example for MXNet backend:
```
pip install mxnet>=1.2.1 keras-mxnet kerascv
```
Or if you prefer TensorFlow backend:
```
pip install tensorflow kerascv
```
To enable/disable different hardware supports, check out installation instruction for the corresponding backend.

After installation check that the `backend` field is set to the correct value in the file `~/.keras/keras.json`. It is
also preferable to set the value of the `image_data_format` field to `channels_first` in the case of using the MXNet backend.  

## Usage

Example of using a pretrained ResNet-18 model (for `channels_first` data format):
```
from kerascv.model_provider import get_model as kecv_get_model
import numpy as np

net = kecv_get_model("resnet18", pretrained=True)
x = np.zeros((1, 3, 224, 224), np.float32)
y = net.predict(x)
```

## Pretrained models (ImageNet-1K)

Some remarks:
- All quality values are estimated with MXNet backend.
- Top1/Top5 are the standard 1-crop Top-1/Top-5 errors (in percents) on the validation subset of the ImageNet-1K dataset.
- FLOPs/2 is the number of FLOPs divided by two to be similar to the number of MACs.
- Remark `Converted from GL model` means that the model was trained on `MXNet/Gluon` and then converted to Keras.

| Model | Top1 | Top5 | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | ---: | --- |
| AlexNet | 40.47 | 17.88 | 62,378,344 | 1,132.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.394/alexnet-1788-b00ce627.h5.log)) |
| AlexNet-b | 41.08 | 18.53 | 61,100,840 | 714.83M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.384/alexnetb-1853-045e80b5.h5.log)) |
| ZFNet | 39.56 | 17.15 | 62,357,608 | 1,170.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.395/zfnet-1715-3226638b.h5.log)) |
| ZFNet-b | 36.30 | 14.83 | 107,627,624 | 2,479.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.400/zfnetb-1483-6ff6768e.h5.log)) |
| VGG-11 | 29.59 | 10.16 | 132,863,336 | 7,615.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.381/vgg11-1016-c6bc31d0.h5.log)) |
| VGG-13 | 28.37 | 9.50 | 133,047,848 | 11,317.65M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.388/vgg13-0950-f0e5bed7.h5.log)) |
| VGG-16 | 26.61 | 8.32 | 138,357,544 | 15,480.10M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.401/vgg16-0832-baf4278d.h5.log)) |
| VGG-19 | 25.58 | 7.67 | 143,667,240 | 19,642.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.420/vgg19-0767-315c0bc8.h5.log)) |
| BN-VGG-11 | 28.55 | 9.34 | 132,866,088 | 7,630.21M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.339/bn_vgg11-0934-96a967ba.h5.log)) |
| BN-VGG-13 | 27.68 | 8.87 | 133,050,792 | 11,341.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.353/bn_vgg13-0887-d4a3da40.h5.log)) |
| BN-VGG-16 | 25.50 | 7.57 | 138,361,768 | 15,506.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.359/bn_vgg16-0757-2960ba13.h5.log)) |
| BN-VGG-19 | 23.91 | 6.89 | 143,672,744 | 19,671.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.360/bn_vgg19-0689-aaee8cb7.h5.log)) |
| BN-VGG-11b | 29.24 | 9.75 | 132,868,840 | 7,630.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.407/bn_vgg11b-0975-8a35fd72.h5.log)) |
| BN-VGG-13b | 29.48 | 10.16 | 133,053,736 | 11,342.14M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.123/bn_vgg13b-1016-b26cafd3.h5.log)) |
| BN-VGG-16b | 26.88 | 8.65 | 138,365,992 | 15,507.20M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.123/bn_vgg16b-0865-2272fdd1.h5.log)) |
| BN-VGG-19b | 25.65 | 8.14 | 143,678,248 | 19,672.26M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.123/bn_vgg19b-0814-852e2ca2.h5.log)) |
| ResNet-10 | 34.59 | 13.85 | 5,418,792 | 894.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.248/resnet10-1385-0a7d3ca6.h5.log)) |
| ResNet-12 | 33.43 | 13.03 | 5,492,776 | 1,126.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.253/resnet12-1303-3ba378de.h5.log)) |
| ResNet-14 | 32.18 | 12.20 | 5,788,200 | 1,357.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.256/resnet14-1220-b7cfec59.h5.log)) |
| ResNet-BC-14b | 30.25 | 11.16 | 10,064,936 | 1,479.12M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.309/resnetbc14b-1116-defe7c19.h5.log)) |
| ResNet-16 | 30.23 | 10.88 | 6,968,872 | 1,589.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.259/resnet16-1088-cc0968d3.h5.log)) |
| ResNet-18 x0.25 | 39.30 | 17.41 | 3,937,400 | 270.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.262/resnet18_wd4-1741-6d84323b.h5.log)) |
| ResNet-18 x0.5 | 33.40 | 12.83 | 5,804,296 | 608.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.263/resnet18_wd2-1283-8e70ce72.h5.log)) |
| ResNet-18 x0.75 | 29.98 | 10.66 | 8,476,056 | 1,129.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.266/resnet18_w3d4-1066-afa3a239.h5.log)) |
| ResNet-18 | 28.08 | 9.52 | 11,689,512 | 1,820.41M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.153/resnet18-0952-0817d058.h5.log)) |
| ResNet-26 | 26.12 | 8.37 | 17,960,232 | 2,746.79M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.305/resnet26-0837-b3c764c0.h5.log)) |
| ResNet-BC-26b | 24.85 | 7.59 | 15,995,176 | 2,356.67M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.313/resnetbc26b-0759-a1916fd0.h5.log)) |
| ResNet-34 | 24.53 | 7.44 | 21,797,672 | 3,672.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.291/resnet34-0744-d366daf8.h5.log)) |
| ResNet-BC-38b | 23.48 | 6.72 | 21,925,416 | 3,234.21M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.328/resnetbc38b-0672-703a7543.h5.log)) |
| ResNet-50 | 22.14 | 6.04 | 25,557,032 | 3,877.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.329/resnet50-0604-8e1e86d3.h5.log)) |
| ResNet-50b | 22.06 | 6.10 | 25,557,032 | 4,110.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.308/resnet50b-0610-8a54fb83.h5.log)) |
| ResNet-101 | 21.64 | 5.99 | 44,549,160 | 7,597.95M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.49/resnet101-0599-ab428947.h5.log)) |
| ResNet-101b | 20.25 | 5.11 | 44,549,160 | 7,830.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.357/resnet101b-0511-84e8ef69.h5.log)) |
| ResNet-152 | 20.74 | 5.35 | 60,192,808 | 11,321.85M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.144/resnet152-0535-43ecb2b0.h5.log)) |
| ResNet-152b | 19.63 | 4.79 | 60,192,808 | 11,554.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.378/resnet152b-0479-a0dd484c.h5.log)) |
| PreResNet-10 | 34.65 | 14.01 | 5,417,128 | 894.19M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.249/preresnet10-1401-2349a7c8.h5.log)) |
| PreResNet-12 | 33.56 | 13.22 | 5,491,112 | 1,126.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.257/preresnet12-1322-32f2f50c.h5.log)) |
| PreResNet-14 | 32.29 | 12.19 | 5,786,536 | 1,358.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.260/preresnet14-1219-b123205e.h5.log)) |
| PreResNet-BC-14b | 30.66 | 11.51 | 10,057,384 | 1,476.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.315/preresnetbc14b-1151-8989bc9f.h5.log)) |
| PreResNet-16 | 30.21 | 10.81 | 6,967,208 | 1,589.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.261/preresnet16-1081-ec02b799.h5.log)) |
| PreResNet-18 x0.25 | 39.63 | 17.78 | 3,935,960 | 270.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.272/preresnet18_wd4-1778-13ecb34c.h5.log)) |
| PreResNet-18 x0.5 | 33.67 | 13.19 | 5,802,440 | 608.73M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.273/preresnet18_wd2-1319-694dbc5b.h5.log)) |
| PreResNet-18 x0.75 | 29.95 | 10.68 | 8,473,784 | 1,129.51M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.274/preresnet18_w3d4-1068-13000951.h5.log)) |
| PreResNet-18 | 28.16 | 9.52 | 11,687,848 | 1,820.56M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.140/preresnet18-0952-b88bf767.h5.log)) |
| PreResNet-26 | 26.02 | 8.34 | 17,958,568 | 2,746.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.316/preresnet26-0834-be46d91c.h5.log)) |
| PreResNet-BC-26b | 25.20 | 7.86 | 15,987,624 | 2,354.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.325/preresnetbc26b-0786-f6ab507b.h5.log)) |
| PreResNet-34 | 24.55 | 7.51 | 21,796,008 | 3,672.83M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.300/preresnet34-0751-fcccbc33.h5.log)) |
| PreResNet-BC-38b | 22.65 | 6.33 | 21,917,864 | 3,231.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.348/preresnetbc38b-0633-b6793dec.h5.log)) |
| PreResNet-50 | 22.26 | 6.20 | 25,549,480 | 3,875.44M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.330/preresnet50-0620-91bd3a60.h5.log)) |
| PreResNet-50b | 22.35 | 6.32 | 25,549,480 | 4,107.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.307/preresnet50b-0632-d3f20f4e.h5.log)) |
| PreResNet-101 | 21.43 | 5.75 | 44,541,608 | 7,595.44M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.50/preresnet101-0575-5dff088d.h5.log)) |
| PreResNet-101b | 20.84 | 5.40 | 44,541,608 | 7,827.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.351/preresnet101b-0540-e70bed8e.h5.log)) |
| PreResNet-152 | 20.69 | 5.31 | 60,185,256 | 11,319.34M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.50/preresnet152-0531-a5ac128d.h5.log)) |
| PreResNet-152b | 19.89 | 5.00 | 60,185,256 | 11,551.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.386/preresnet152b-0500-360cd640.h5.log)) |
| PreResNet-200b | 21.09 | 5.64 | 64,666,280 | 15,068.63M | From [tornadomeet/ResNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.50/preresnet200b-0564-9172d4c0.h5.log)) |
| PreResNet-269b | 20.71 | 5.56 | 102,065,832 | 20,101.11M | From [soeaver/mxnet-model] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.239/preresnet269b-0556-bdd89388.h5.log)) |
| ResNeXt-14 (16x4d) | 31.65 | 12.24 | 7,127,336 | 1,045.77M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.370/resnext14_16x4d-1224-146ff5da.h5.log)) |
| ResNeXt-14 (32x2d) | 32.15 | 12.46 | 7,029,416 | 1,031.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.371/resnext14_32x2d-1246-3af87217.h5.log)) |
| ResNeXt-14 (32x4d) | 29.95 | 11.10 | 9,411,880 | 1,603.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.327/resnext14_32x4d-1110-86af26f7.h5.log)) |
| ResNeXt-26 (32x2d) | 26.34 | 8.50 | 9,924,136 | 1,461.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.373/resnext26_32x2d-0850-0e54facd.h5.log)) |
| ResNeXt-26 (32x4d) | 23.91 | 7.20 | 15,389,480 | 2,488.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.332/resnext26_32x4d-0720-a5e34838.h5.log)) |
| ResNeXt-50 (32x4d) | 20.64 | 5.46 | 25,028,904 | 4,255.86M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.417/resnext50_32x4d-0546-1c9906b0.h5.log)) |
| ResNeXt-101 (32x4d) | 19.62 | 4.92 | 44,177,704 | 8,003.45M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.417/resnext101_32x4d-0492-24e9dbdb.h5.log)) |
| ResNeXt-101 (64x4d) | 19.28 | 4.83 | 83,455,272 | 15,500.27M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.417/resnext101_64x4d-0483-a6b4bdef.h5.log)) |
| SE-ResNet-10 | 33.55 | 13.29 | 5,463,332 | 894.27M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.354/seresnet10-1329-f70cf6c7.h5.log)) |
| SE-ResNet-18 | 27.95 | 9.20 | 11,778,592 | 1,820.88M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.355/seresnet18-0920-bb27e273.h5.log)) |
| SE-ResNet-26 | 25.42 | 8.03 | 18,093,852 | 2,747.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.363/seresnet26-0803-d6897147.h5.log)) |
| SE-ResNet-BC-26b | 23.44 | 6.82 | 17,395,976 | 2,359.58M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.366/seresnetbc26b-0682-ba3e5170.h5.log)) |
| SE-ResNet-BC-38b | 21.44 | 5.75 | 24,026,616 | 3,238.58M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.374/seresnetbc38b-0575-53688136.h5.log)) |
| SE-ResNet-50 | 22.50 | 6.43 | 28,088,024 | 3,880.49M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.52/seresnet50-0643-fabfa406.h5.log)) |
| SE-ResNet-50b | 20.58 | 5.33 | 28,088,024 | 4,115.78M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.387/seresnet50b-0533-bc9d11ec.h5.log)) |
| SE-ResNet-101 | 21.92 | 5.88 | 49,326,872 | 7,602.76M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.52/seresnet101-0588-933d3415.h5.log)) |
| SE-ResNet-152 | 21.46 | 5.77 | 66,821,848 | 11,328.52M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.52/seresnet152-0577-d25ced7d.h5.log)) |
| SE-PreResNet-10 | 33.60 | 13.06 | 5,461,668 | 894.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.377/sepreresnet10-1306-6096e4d9.h5.log)) |
| SE-PreResNet-18 | 27.67 | 9.38 | 11,776,928 | 1,821.03M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.380/sepreresnet18-0938-d0bf29b9.h5.log)) |
| SE-PreResNet-BC-26b | 22.95 | 6.36 | 17,388,424 | 2,357.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.399/sepreresnetbc26b-0636-cc11e087.h5.log)) |
| SE-PreResNet-BC-38b | 21.42 | 5.63 | 24,019,064 | 3,236.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.409/sepreresnetbc38b-0563-f4b96ed7.h5.log)) |
| SE-ResNeXt-50 (32x4d) | 20.03 | 5.05 | 27,559,896 | 4,261.16M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.418/seresnext50_32x4d-0505-077f048f.h5.log)) |
| SE-ResNeXt-101 (32x4d) | 19.07 | 4.60 | 48,955,416 | 8,012.73M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.418/seresnext101_32x4d-0460-08ea8055.h5.log)) |
| SE-ResNeXt-101 (64x4d) | 18.98 | 4.66 | 88,232,984 | 15,509.54M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.418/seresnext101_64x4d-0466-28ff2d1f.h5.log)) |
| SENet-16 | 25.34 | 8.06 | 31,366,168 | 5,081.30M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.341/senet16-0806-8a634c50.h5.log)) |
| SENet-28 | 21.68 | 5.91 | 36,453,768 | 5,732.71M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.356/senet28-0591-33c65063.h5.log)) |
| SENet-154 | 18.83 | 4.65 | 115,088,984 | 20,745.78M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.54/senet154-0465-962aeede.h5.log)) |
| DenseNet-121 | 23.23 | 6.84 | 7,978,856 | 2,872.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.314/densenet121-0684-7c6d506a.h5.log)) |
| DenseNet-161 | 22.39 | 6.18 | 28,681,000 | 7,793.16M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.55/densenet161-0618-070fcb45.h5.log)) |
| DenseNet-169 | 22.09 | 6.05 | 14,149,480 | 3,403.89M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.406/densenet169-0605-7b3b7888.h5.log)) |
| DenseNet-201 | 22.69 | 6.35 | 20,013,928 | 4,347.15M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.55/densenet201-0635-cf3afbb2.h5.log)) |
| DarkNet Tiny | 40.31 | 17.46 | 1,042,104 | 500.85M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.69/darknet_tiny-1746-147e949b.h5.log)) |
| DarkNet Ref | 37.99 | 16.68 | 7,319,416 | 367.59M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.64/darknet_ref-1668-2ef080bb.h5.log)) |
| DarkNet-53 | 21.43 | 5.56 | 41,609,928 | 7,133.86M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.150/darknet53-0556-d6c6e7dc.h5.log)) |
| SqueezeNet v1.0 | 39.17 | 17.56 | 1,248,424 | 823.67M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.128/squeezenet_v1_0-1756-a4890923.h5.log)) |
| SqueezeNet v1.1 | 39.08 | 17.39 | 1,235,496 | 352.02M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.88/squeezenet_v1_1-1739-b9a8f9ea.h5.log)) |
| SqueezeResNet v1.0 | 39.40 | 17.80 | 1,248,424 | 823.67M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.178/squeezeresnet_v1_0-1780-fb9a54aa.h5.log)) |
| SqueezeResNet v1.1 | 39.82 | 17.84 | 1,235,496 | 352.02M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.70/squeezeresnet_v1_1-1784-43ee9cbb.h5.log)) |
| 1.0-SqNxt-23 | 42.28 | 18.62 | 724,056 | 287.28M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.171/sqnxt23_w1-1862-cab60636.h5.log)) |
| 1.0-SqNxt-23v5 | 40.38 | 17.57 | 921,816 | 285.82M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.172/sqnxt23v5_w1-1757-96b94e1d.h5.log)) |
| 1.5-SqNxt-23 | 34.59 | 13.30 | 1,511,824 | 552.39M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.210/sqnxt23_w3d2-1330-e52625a0.h5.log)) |
| 1.5-SqNxt-23v5 | 33.56 | 12.84 | 1,953,616 | 550.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.212/sqnxt23v5_w3d2-1284-fd150fcc.h5.log)) |
| 2.0-SqNxt-23 | 30.15 | 10.66 | 2,583,752 | 898.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.240/sqnxt23_w2-1066-a34e73b9.h5.log)) |
| 2.0-SqNxt-23v5 | 29.40 | 10.28 | 3,366,344 | 897.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.216/sqnxt23v5_w2-1028-13c5a598.h5.log)) |
| ShuffleNet x0.25 (g=1) | 62.00 | 36.76 | 209,746 | 12.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.134/shufflenet_g1_wd4-3676-cb39b773.h5.log)) |
| ShuffleNet x0.25 (g=3) | 61.32 | 36.15 | 305,902 | 13.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.135/shufflenet_g3_wd4-3615-21150468.h5.log)) |
| ShuffleNet x0.5 (g=1) | 46.21 | 22.38 | 534,484 | 41.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.174/shufflenet_g1_wd2-2238-76709a36.h5.log)) |
| ShuffleNet x0.5 (g=3) | 43.82 | 20.60 | 718,324 | 41.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.167/shufflenet_g3_wd2-2060-173a725c.h5.log)) |
| ShuffleNet x0.75 (g=1) | 39.24 | 16.75 | 975,214 | 86.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.218/shufflenet_g1_w3d4-1675-56aa4179.h5.log)) |
| ShuffleNet x0.75 (g=3) | 37.81 | 16.09 | 1,238,266 | 85.82M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.219/shufflenet_g3_w3d4-1609-34e28781.h5.log)) |
| ShuffleNet x1.0 (g=1) | 34.41 | 13.50 | 1,531,936 | 148.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.223/shufflenet_g1_w1-1350-f44c8a18.h5.log)) |
| ShuffleNet x1.0 (g=2) | 33.97 | 13.32 | 1,733,848 | 147.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.241/shufflenet_g2_w1-1332-8784a32b.h5.log)) |
| ShuffleNet x1.0 (g=3) | 33.96 | 13.29 | 1,865,728 | 145.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.244/shufflenet_g3_w1-1329-0e213e76.h5.log)) |
| ShuffleNet x1.0 (g=4) | 33.83 | 13.10 | 1,968,344 | 143.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.245/shufflenet_g4_w1-1310-ef2ff63e.h5.log)) |
| ShuffleNet x1.0 (g=8) | 33.64 | 13.20 | 2,434,768 | 150.76M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.250/shufflenet_g8_w1-1320-796314f1.h5.log)) |
| ShuffleNetV2 x0.5 | 40.76 | 18.40 | 1,366,792 | 43.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.90/shufflenetv2_wd2-1840-9b4b0964.h5.log)) |
| ShuffleNetV2 x1.0 | 31.02 | 11.33 | 2,278,604 | 149.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.133/shufflenetv2_w1-1133-bcba973e.h5.log)) |
| ShuffleNetV2 x1.5 | 27.32 | 9.27 | 4,406,098 | 320.77M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.288/shufflenetv2_w3d2-0927-17a26039.h5.log)) |
| ShuffleNetV2 x2.0 | 25.77 | 8.22 | 7,601,686 | 595.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.301/shufflenetv2_w2-0822-a0209f14.h5.log)) |
| ShuffleNetV2b x0.5 | 39.81 | 17.83 | 1,366,792 | 43.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.211/shufflenetv2b_wd2-1783-ca8409ae.h5.log)) |
| ShuffleNetV2b x1.0 | 30.38 | 11.01 | 2,279,760 | 150.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.211/shufflenetv2b_w1-1101-1caf1b22.h5.log)) |
| ShuffleNetV2b x1.5 | 26.89 | 8.80 | 4,410,194 | 323.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.211/shufflenetv2b_w3d2-0880-265c3c7c.h5.log)) |
| ShuffleNetV2b x2.0 | 25.18 | 8.10 | 7,611,290 | 603.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.242/shufflenetv2b_w2-0810-2149df38.h5.log)) |
| 108-MENet-8x1 (g=3) | 43.61 | 20.31 | 654,516 | 42.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.89/menet108_8x1_g3-2031-a4d43433.h5.log)) |
| 128-MENet-8x1 (g=4) | 42.08 | 19.14 | 750,796 | 45.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.103/menet128_8x1_g4-1914-5bb8f228.h5.log)) |
| 160-MENet-8x1 (g=8) | 43.47 | 20.28 | 850,120 | 45.63M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.154/menet160_8x1_g8-2028-09664de9.h5.log)) |
| 228-MENet-12x1 (g=3) | 33.85 | 12.88 | 1,806,568 | 152.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.131/menet228_12x1_g3-1288-c2eeac24.h5.log)) |
| 256-MENet-12x1 (g=4) | 32.22 | 12.17 | 1,888,240 | 150.65M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.152/menet256_12x1_g4-1217-b020cc33.h5.log)) |
| 348-MENet-12x1 (g=3) | 27.85 | 9.36 | 3,368,128 | 312.00M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.173/menet348_12x1_g3-0936-6795f007.h5.log)) |
| 352-MENet-12x1 (g=8) | 31.29 | 11.67 | 2,272,872 | 157.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.198/menet352_12x1_g8-1167-a9d9412d.h5.log)) |
| 456-MENet-24x1 (g=3) | 25.00 | 7.80 | 5,304,784 | 567.90M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.237/menet456_24x1_g3-0780-6645f594.h5.log)) |
| MobileNet x0.25 | 45.80 | 22.17 | 470,072 | 44.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.62/mobilenet_wd4-2217-fb7abda8.h5.log)) |
| MobileNet x0.5 | 33.94 | 13.30 | 1,331,592 | 155.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.156/mobilenet_wd2-1330-aa86f355.h5.log)) |
| MobileNet x0.75 | 29.85 | 10.51 | 2,585,560 | 333.99M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.130/mobilenet_w3d4-1051-d200ad45.h5.log)) |
| MobileNet x1.0 | 26.43 | 8.66 | 4,231,976 | 579.80M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.155/mobilenet_w1-0866-9661b555.h5.log)) |
| FD-MobileNet x0.25 | 55.42 | 30.52 | 383,160 | 12.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.177/fdmobilenet_wd4-3052-6c219205.h5.log)) |
| FD-MobileNet x0.5 | 42.61 | 19.69 | 993,928 | 41.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.83/fdmobilenet_wd2-1969-5678a212.h5.log)) |
| FD-MobileNet x0.75 | 37.90 | 16.01 | 1,833,304 | 86.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.159/fdmobilenet_w3d4-1601-2ea5eba9.h5.log)) |
| FD-MobileNet x1.0 | 33.80 | 13.12 | 2,901,288 | 147.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.162/fdmobilenet_w1-1312-e11d0dce.h5.log)) |
| MobileNetV2 x0.25 | 48.06 | 24.12 | 1,516,392 | 34.24M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.137/mobilenetv2_wd4-2412-62273372.h5.log)) |
| MobileNetV2 x0.5 | 35.63 | 14.43 | 1,964,736 | 100.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.170/mobilenetv2_wd2-1443-c7086bcc.h5.log)) |
| MobileNetV2 x0.75 | 29.76 | 10.44 | 2,627,592 | 198.50M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.230/mobilenetv2_w3d4-1044-29e9923c.h5.log)) |
| MobileNetV2 x1.0 | 26.76 | 8.64 | 3,504,960 | 329.36M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.213/mobilenetv2_w1-0864-5e487e82.h5.log)) |
| MobileNetV3 L/224/1.0 | 24.63 | 7.69 | 5,481,752 | 227.09M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.411/mobilenetv3_large_w1-0769-fc909b4c.h5.log)) |
| IGCV3 x0.25 | 53.41 | 28.29 | 1,534,020 | 41.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.142/igcv3_wd4-2829-00072caf.h5.log)) |
| IGCV3 x0.5 | 39.39 | 17.04 | 1,985,528 | 111.12M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.132/igcv3_wd2-1704-b8961ca3.h5.log)) |
| IGCV3 x0.75 | 30.71 | 10.97 | 2,638,084 | 210.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.207/igcv3_w3d4-1097-fb365b72.h5.log)) |
| IGCV3 x1.0 | 27.72 | 8.99 | 3,491,688 | 340.79M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.243/igcv3_w1-0899-968237cb.h5.log)) |
| MnasNet-B1 | 25.76 | 8.00 | 4,383,312 | 326.30M |  From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.419/mnasnet_b1-0800-9ce379b3.h5.log)) |
| MnasNet-A1 | 25.02 | 7.55 | 3,887,038 | 326.07M |  From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.419/mnasnet_a1-0755-8bf70a05.h5.log)) |
| EfficientNet-B0 | 24.50 | 7.22 | 5,288,548 | 414.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.364/efficientnet_b0-0722-2bea741f.h5.log)) |
| EfficientNet-B1 | 22.89 | 6.26 | 7,794,184 | 732.54M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.376/efficientnet_b1-0626-d7a4bf8b.h5.log)) |
| EfficientNet-B0b | 22.95 | 6.69 | 5,288,548 | 414.31M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.403/efficientnet_b0b-0669-436cc024.h5.log)) |
| EfficientNet-B1b | 20.97 | 5.64 | 7,794,184 | 732.54M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.403/efficientnet_b1b-0564-f2eb3cd8.h5.log)) |
| EfficientNet-B2b | 19.93 | 5.16 | 9,109,994 | 1,051.98M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.403/efficientnet_b2b-0516-9c08b839.h5.log)) |
| EfficientNet-B3b | 18.59 | 4.31 | 12,233,232 | 1,928.55M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.403/efficientnet_b3b-0431-d1545ea0.h5.log)) |
| EfficientNet-B4b | 17.24 | 3.76 | 19,341,616 | 4,607.46M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.403/efficientnet_b4b-0376-c7e29f57.h5.log)) |
| EfficientNet-B5b | 16.39 | 3.34 | 30,389,784 | 10,695.20M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.403/efficientnet_b5b-0334-4365cf12.h5.log)) |
| EfficientNet-B6b | 15.96 | 3.12 | 43,040,704 | 19,796.24M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.403/efficientnet_b6b-0312-7f3f3465.h5.log)) |
| EfficientNet-B7b | 15.70 | 3.11 | 66,347,960 | 39,010.98M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.403/efficientnet_b7b-0311-b4aac2ce.h5.log)) |

[dmlc/gluon-cv]: https://github.com/dmlc/gluon-cv
[tornadomeet/ResNet]: https://github.com/tornadomeet/ResNet
[Cadene/pretrained...pytorch]: https://github.com/Cadene/pretrained-models.pytorch
[clavichord93/MENet]: https://github.com/clavichord93/MENet
[zeusees/Mnasnet...Model]: https://github.com/zeusees/Mnasnet-Pretrained-Model
[soeaver/mxnet-model]: https://github.com/soeaver/mxnet-model
[rwightman/pyt...models]: https://github.com/rwightman/pytorch-image-models