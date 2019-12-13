# Large-scale image classification models on TensorFlow 2.x

[![PyPI](https://img.shields.io/pypi/v/tf2cv.svg)](https://pypi.python.org/pypi/tf2cv)
[![Downloads](https://pepy.tech/badge/tf2cv)](https://pepy.tech/project/tf2cv)

This is a collection of large-scale image classification models. Many of them are pretrained on
[ImageNet-1K](http://www.image-net.org) dataset and loaded automatically during use. All pretrained models require the
same ordinary normalization. Scripts for training/evaluating/converting models are in the
[`imgclsmob`](https://github.com/osmr/imgclsmob) repo.

## List of implemented models

- AlexNet (['One weird trick for parallelizing convolutional neural networks'](https://arxiv.org/abs/1404.5997))
- ZFNet (['Visualizing and Understanding Convolutional Networks'](https://arxiv.org/abs/1311.2901))
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

## Pretrained models (ImageNet-1K)

Some remarks:
- Top1/Top5 are the standard 1-crop Top-1/Top-5 errors (in percents) on the validation subset of the ImageNet-1K dataset.
- FLOPs/2 is the number of FLOPs divided by two to be similar to the number of MACs.
- Remark `Converted from GL model` means that the model was trained on `MXNet/Gluon` and then converted to TensorFlow.

| Model | Top1 | Top5 | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | ---: | --- |
| AlexNet | 40.50 | 17.89 | 62,378,344 | 1,132.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/alexnet-1789-ecc4bb4e.tf2.h5.log)) |
| AlexNet-b | 41.03 | 18.59 | 61,100,840 | 714.83M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/alexnetb-1859-9e390537.tf2.h5.log)) |
| ZFNet | 395.0 | 17.17 | 62,357,608 | 1,170.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/zfnet-1717-9500db30.tf2.h5.log)) |
| ZFNet-b | 36.28 | 14.80 | 107,627,624 | 2,479.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/zfnetb-1480-47533f6a.tf2.h5.log)) |
| VGG-11 | 29.59 | 10.17 | 132,863,336 | 7,615.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/vgg11-1017-c20556f4.tf2.h5.log)) |
| VGG-13 | 28.41 | 9.51 | 133,047,848 | 11,317.65M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/vgg13-0951-9fa609fc.tf2.h5.log)) |
| VGG-16 | 26.59 | 8.34 | 138,357,544 | 15,480.10M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/vgg16-0834-ce78831f.tf2.h5.log)) |
| VGG-19 | 25.57 | 7.68 | 143,667,240 | 19,642.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/vgg19-0768-ec5ac0ba.tf2.h5.log)) |
| BN-VGG-11 | 28.57 | 9.36 | 132,866,088 | 7,630.21M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/bn_vgg11-0936-ef31b866.tf2.h5.log)) |
| BN-VGG-13 | 27.67 | 8.87 | 133,050,792 | 11,341.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/bn_vgg13-0887-2cccc725.tf2.h5.log)) |
| BN-VGG-16 | 25.46 | 7.59 | 138,361,768 | 15,506.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/bn_vgg16-0759-1ca9dee8.tf2.h5.log)) |
| BN-VGG-19 | 23.89 | 6.88 | 143,672,744 | 19,671.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/bn_vgg19-0688-81d25be8.tf2.h5.log)) |
| BN-VGG-11b | 29.31 | 9.75 | 132,868,840 | 7,630.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/bn_vgg11b-0975-aeaccfdc.tf2.h5.log)) |
| BN-VGG-13b | 29.46 | 10.19 | 133,053,736 | 11,342.14M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/bn_vgg13b-1019-1102ffb7.tf2.h5.log)) |
| BN-VGG-16b | 26.89 | 8.62 | 138,365,992 | 15,507.20M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/bn_vgg16b-0862-137178f7.tf2.h5.log)) |
| BN-VGG-19b | 25.64 | 8.17 | 143,678,248 | 19,672.26M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/bn_vgg19b-0817-cd68a741.tf2.h5.log)) |
| ResNet-10 | 34.68 | 13.90 | 5,418,792 | 894.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet10-1390-9e787f63.tf2.h5.log)) |
| ResNet-12 | 33.43 | 13.01 | 5,492,776 | 1,126.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet12-1301-8bc41d1b.tf2.h5.log)) |
| ResNet-14 | 32.21 | 12.24 | 5,788,200 | 1,357.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet14-1224-7573d988.tf2.h5.log)) |
| ResNet-BC-14b | 30.21 | 11.15 | 10,064,936 | 1,479.12M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnetbc14b-1115-5f30b798.tf2.h5.log)) |
| ResNet-16 | 30.22 | 10.88 | 6,968,872 | 1,589.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet16-1088-14ce0d64.tf2.h5.log)) |
| ResNet-18 x0.25 | 39.30 | 17.45 | 3,937,400 | 270.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet18_wd4-1745-6e800416.tf2.h5.log)) |
| ResNet-18 x0.5 | 33.40 | 12.83 | 5,804,296 | 608.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet18_wd2-1283-85a7caff.tf2.h5.log)) |
| ResNet-18 x0.75 | 29.98 | 10.67 | 8,476,056 | 1,129.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet18_w3d4-1067-c1735b7d.tf2.h5.log)) |
| ResNet-18 | 28.10 | 9.56 | 11,689,512 | 1,820.41M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet18-0956-6645845a.tf2.h5.log)) |
| ResNet-26 | 26.15 | 8.37 | 17,960,232 | 2,746.79M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet26-0837-a8f20f71.tf2.h5.log)) |
| ResNet-BC-26b | 24.80 | 7.57 | 15,995,176 | 2,356.67M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnetbc26b-0757-d70a2cad.tf2.h5.log)) |
| ResNet-34 | 24.50 | 7.44 | 21,797,672 | 3,672.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet34-0744-7f7d70e7.tf2.h5.log)) |
| ResNet-BC-38b | 23.44 | 6.77 | 21,925,416 | 3,234.21M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnetbc38b-0677-75e405a7.tf2.h5.log)) |
| ResNet-50 | 22.09 | 6.04 | 25,557,032 | 3,877.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet50-0604-728800bf.tf2.h5.log)) |
| ResNet-50b | 22.09 | 6.14 | 25,557,032 | 4,110.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet50b-0614-b2a49da6.tf2.h5.log)) |
| ResNet-101 | 21.59 | 6.01 | 44,549,160 | 7,597.95M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet101-0601-b6befeb4.tf2.h5.log)) |
| ResNet-101b | 20.25 | 5.11 | 44,549,160 | 7,830.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet101b-0511-e3076227.tf2.h5.log)) |
| ResNet-152 | 20.72 | 5.34 | 60,192,808 | 11,321.85M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet152-0534-2d8e394a.tf2.h5.log)) |
| ResNet-152b | 19.60 | 4.80 | 60,192,808 | 11,554.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet152b-0480-b77f1e2c.tf2.h5.log)) |
| PreResNet-10 | 34.71 | 14.02 | 5,417,128 | 894.19M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet10-1402-541bf0e1.tf2.h5.log)) |
| PreResNet-12 | 33.63 | 13.20 | 5,491,112 | 1,126.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet12-1320-349c0df4.tf2.h5.log)) |
| PreResNet-14 | 32.29 | 12.24 | 5,786,536 | 1,358.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet14-1224-194b8762.tf2.h5.log)) |
| PreResNet-BC-14b | 30.73 | 11.52 | 10,057,384 | 1,476.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnetbc14b-1152-bc4e06ff.tf2.h5.log)) |
| PreResNet-16 | 30.17 | 10.80 | 6,967,208 | 1,589.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet16-1080-e00c40ee.tf2.h5.log)) |
| PreResNet-18 x0.25 | 39.61 | 17.80 | 3,935,960 | 270.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet18_wd4-1780-6ac7bc59.tf2.h5.log)) |
| PreResNet-18 x0.5 | 33.70 | 13.14 | 5,802,440 | 608.73M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet18_wd2-1314-0c0528c8.tf2.h5.log)) |
| PreResNet-18 x0.75 | 29.95 | 10.70 | 8,473,784 | 1,129.51M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet18_w3d4-1070-056b46c6.tf2.h5.log)) |
| PreResNet-18 | 28.20 | 9.55 | 11,687,848 | 1,820.56M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet18-0955-621ead92.tf2.h5.log)) |
| PreResNet-26 | 25.98 | 8.37 | 17,958,568 | 2,746.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet26-0837-1a92a732.tf2.h5.log)) |
| PreResNet-BC-26b | 25.22 | 7.88 | 15,987,624 | 2,354.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnetbc26b-0788-1f737cd6.tf2.h5.log)) |
| PreResNet-34 | 24.60 | 7.54 | 21,796,008 | 3,672.83M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet34-0754-3cc5ae14.tf2.h5.log)) |
| PreResNet-BC-38b | 22.70 | 6.36 | 21,917,864 | 3,231.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnetbc38b-0636-3396b49b.tf2.h5.log)) |
| PreResNet-50 | 22.22 | 6.25 | 25,549,480 | 3,875.44M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet50-0625-20860562.tf2.h5.log)) |
| PreResNet-50b | 22.37 | 6.34 | 25,549,480 | 4,107.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet50b-0634-711227b1.tf2.h5.log)) |
| PreResNet-101 | 21.47 | 5.73 | 44,541,608 | 7,595.44M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet101-0573-d45ea488.tf2.h5.log)) |
| PreResNet-101b | 20.86 | 5.39 | 44,541,608 | 7,827.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet101b-0539-54d23aff.tf2.h5.log)) |
| PreResNet-152 | 20.71 | 5.32 | 60,185,256 | 11,319.34M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet152-0532-0ad4b58f.tf2.h5.log)) |
| PreResNet-152b | 19.86 | 5.00 | 60,185,256 | 11,551.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet152b-0500-119062d9.tf2.h5.log)) |
| PreResNet-200b | 21.07 | 5.63 | 64,666,280 | 15,068.63M | From [tornadomeet/ResNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet200b-0563-2f9c761d.tf2.h5.log)) |
| PreResNet-269b | 20.75 | 5.57 | 102,065,832 | 20,101.11M | From [soeaver/mxnet-model] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet269b-0557-7003b3c4.tf2.h5.log)) |
| ResNeXt-14 (16x4d) | 31.69 | 12.22 | 7,127,336 | 1,045.77M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnext14_16x4d-1222-bff90c1d.tf2.h5.log)) |
| ResNeXt-14 (32x2d) | 32.14 | 12.47 | 7,029,416 | 1,031.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnext14_32x2d-1247-06aa6709.tf2.h5.log)) |
| ResNeXt-14 (32x4d) | 29.94 | 11.15 | 9,411,880 | 1,603.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnext14_32x4d-1115-3acdaec1.tf2.h5.log)) |
| ResNeXt-26 (32x2d) | 26.32 | 8.51 | 9,924,136 | 1,461.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnext26_32x2d-0851-827791cc.tf2.h5.log)) |
| ResNeXt-26 (32x4d) | 23.94 | 7.18 | 15,389,480 | 2,488.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnext26_32x4d-0718-4f05525e.tf2.h5.log)) |
| ResNeXt-50 (32x4d) | 20.62 | 5.47 | 25,028,904 | 4,255.86M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnext50_32x4d-0547-45234d14.tf2.h5.log)) |
| ResNeXt-101 (32x4d) | 19.65 | 4.94 | 44,177,704 | 8,003.45M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnext101_32x4d-0494-3990ddd1.tf2.h5.log)) |
| ResNeXt-101 (64x4d) | 19.31 | 4.84 | 83,455,272 | 15,500.27M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnext101_64x4d-0484-f8cf1580.tf2.h5.log)) |
| SE-ResNet-10 | 33.54 | 13.32 | 5,463,332 | 894.27M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnet10-1332-33a592e1.tf2.h5.log)) |
| SE-ResNet-18 | 27.97 | 9.21 | 11,778,592 | 1,820.88M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnet18-0921-46c847ab.tf2.h5.log)) |
| SE-ResNet-26 | 25.42 | 8.07 | 18,093,852 | 2,747.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnet26-0807-5178b3b1.tf2.h5.log)) |
| SE-ResNet-BC-26b | 23.39 | 6.84 | 17,395,976 | 2,359.58M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnetbc26b-0684-1460a381.tf2.h5.log)) |
| SE-ResNet-BC-38b | 21.43 | 5.75 | 24,026,616 | 3,238.58M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnetbc38b-0575-18fcfcc1.tf2.h5.log)) |
| SE-ResNet-50 | 22.52 | 6.42 | 28,088,024 | 3,880.49M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnet50-0642-21b366af.tf2.h5.log)) |
| SE-ResNet-50b | 20.58 | 5.33 | 28,088,024 | 4,115.78M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnet50b-0533-256002c3.tf2.h5.log)) |
| SE-ResNet-101 | 21.94 | 5.89 | 49,326,872 | 7,602.76M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnet101-0589-2a22ba87.tf2.h5.log)) |
| SE-ResNet-152 | 21.47 | 5.76 | 66,821,848 | 11,328.52M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnet152-0576-8023259a.tf2.h5.log)) |
| SE-PreResNet-10 | 33.62 | 13.09 | 5,461,668 | 894.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/sepreresnet10-1309-af20d06c.tf2.h5.log)) |
| SE-PreResNet-18 | 27.70 | 9.40 | 11,776,928 | 1,821.03M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/sepreresnet18-0940-fe403280.tf2.h5.log)) |
| SE-PreResNet-BC-26b | 22.95 | 6.40 | 17,388,424 | 2,357.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/sepreresnetbc26b-0640-a72bf876.tf2.h5.log)) |
| SE-PreResNet-BC-38b | 21.44 | 5.67 | 24,019,064 | 3,236.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/sepreresnetbc38b-0567-17d10c63.tf2.h5.log)) |
| SE-ResNeXt-50 (32x4d) | 19.98 | 5.09 | 27,559,896 | 4,261.16M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnext50_32x4d-0509-4244900a.tf2.h5.log)) |
| SE-ResNeXt-101 (32x4d) | 19.01 | 4.59 | 48,955,416 | 8,012.73M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnext101_32x4d-0459-13a9b2fd.tf2.h5.log)) |
| SE-ResNeXt-101 (64x4d) | 18.96 | 4.65 | 88,232,984 | 15,509.54M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnext101_64x4d-0465-ec0a3b13.tf2.h5.log)) |
| SENet-16 | 25.37 | 8.05 | 31,366,168 | 5,081.30M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/senet16-0805-f5f57656.tf2.h5.log)) |
| SENet-28 | 21.68 | 5.90 | 36,453,768 | 5,732.71M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/senet28-0590-667d5687.tf2.h5.log)) |
| SENet-154 | 18.78 | 4.66 | 115,088,984 | 20,745.78M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/senet154-0466-f1b79a9b.tf2.h5.log)) |
| DenseNet-121 | 23.23 | 6.84 | 7,978,856 | 2,872.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/densenet121-0684-e9196a9c.tf2.h5.log)) |
| DenseNet-161 | 22.37 | 6.18 | 28,681,000 | 7,793.16M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/densenet161-0618-e77cf292.tf2.h5.log)) |
| DenseNet-169 | 22.13 | 6.06 | 14,149,480 | 3,403.89M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/densenet169-0606-f708dc33.tf2.h5.log)) |
| DenseNet-201 | 22.67 | 6.37 | 20,013,928 | 4,347.15M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/densenet201-0637-f45e9450.tf2.h5.log)) |
| DarkNet Tiny | 40.34 | 17.45 | 1,042,104 | 500.85M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/darknet_tiny-1745-d30be41a.tf2.h5.log)) |
| DarkNet Ref | 38.10 | 16.71 | 7,319,416 | 367.59M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/darknet_ref-1671-b4991f6b.tf2.h5.log)) |
| DarkNet-53 | 21.41 | 5.58 | 41,609,928 | 7,133.86M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/darknet53-0558-4a63ab30.tf2.h5.log)) |
| SqueezeNet v1.0 | 39.23 | 17.60 | 1,248,424 | 823.67M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/squeezenet_v1_0-1760-d13ba732.tf2.h5.log)) |
| SqueezeNet v1.1 | 39.12 | 17.42 | 1,235,496 | 352.02M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/squeezenet_v1_1-1742-95b61448.tf2.h5.log)) |
| SqueezeResNet v1.0 | 39.38 | 17.83 | 1,248,424 | 823.67M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/squeezeresnet_v1_0-1783-db620d99.tf2.h5.log)) |
| SqueezeResNet v1.1 | 39.85 | 17.89 | 1,235,496 | 352.02M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/squeezeresnet_v1_1-1789-13d6bc6b.tf2.h5.log)) |
| 1.0-SqNxt-23 | 42.31 | 18.61 | 724,056 | 287.28M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/sqnxt23_w1-1861-379975eb.tf2.h5.log)) |
| 1.0-SqNxt-23v5 | 40.44 | 17.62 | 921,816 | 285.82M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/sqnxt23v5_w1-1762-153b4ce7.tf2.h5.log)) |
| 1.5-SqNxt-23 | 34.62 | 13.34 | 1,511,824 | 552.39M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/sqnxt23_w3d2-1334-a2ba956c.tf2.h5.log)) |
| 1.5-SqNxt-23v5 | 33.55 | 12.84 | 1,953,616 | 550.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/sqnxt23v5_w3d2-1284-72efaa71.tf2.h5.log)) |
| 2.0-SqNxt-23 | 30.12 | 10.69 | 2,583,752 | 898.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/sqnxt23_w2-1069-f43dee19.tf2.h5.log)) |
| 2.0-SqNxt-23v5 | 29.40 | 10.26 | 3,366,344 | 897.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/sqnxt23v5_w2-1026-da80c640.tf2.h5.log)) |
| ShuffleNet x0.25 (g=1) | 62.05 | 36.81 | 209,746 | 12.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenet_g1_wd4-3681-04a9e2d4.tf2.h5.log)) |
| ShuffleNet x0.25 (g=3) | 61.31 | 36.18 | 305,902 | 13.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenet_g3_wd4-3618-c9aad0f0.tf2.h5.log)) |
| ShuffleNet x0.5 (g=1) | 46.25 | 22.36 | 534,484 | 41.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenet_g1_wd2-2236-082db702.tf2.h5.log)) |
| ShuffleNet x0.5 (g=3) | 43.84 | 20.59 | 718,324 | 41.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenet_g3_wd2-2059-e3aefeeb.tf2.h5.log)) |
| ShuffleNet x0.75 (g=1) | 39.24 | 16.79 | 975,214 | 86.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenet_g1_w3d4-1679-a1cc5da3.tf2.h5.log)) |
| ShuffleNet x0.75 (g=3) | 37.80 | 16.11 | 1,238,266 | 85.82M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenet_g3_w3d4-1611-89546a05.tf2.h5.log)) |
| ShuffleNet x1.0 (g=1) | 34.48 | 13.48 | 1,531,936 | 148.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenet_g1_w1-1348-52ddb20f.tf2.h5.log)) |
| ShuffleNet x1.0 (g=2) | 33.95 | 13.33 | 1,733,848 | 147.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenet_g2_w1-1333-2a8ba692.tf2.h5.log)) |
| ShuffleNet x1.0 (g=3) | 33.93 | 13.32 | 1,865,728 | 145.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenet_g3_w1-1326-daaec8b8.tf2.h5.log)) |
| ShuffleNet x1.0 (g=4) | 33.88 | 13.13 | 1,968,344 | 143.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenet_g4_w1-1313-35dbd6b9.tf2.h5.log)) |
| ShuffleNet x1.0 (g=8) | 33.71 | 13.22 | 2,434,768 | 150.76M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenet_g8_w1-1322-449fb276.tf2.h5.log)) |
| ShuffleNetV2 x0.5 | 40.75 | 18.43 | 1,366,792 | 43.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenetv2_wd2-1843-d492d721.tf2.h5.log)) |
| ShuffleNetV2 x1.0 | 31.00 | 11.35 | 2,278,604 | 149.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenetv2_w1-1135-dae13ee9.tf2.h5.log)) |
| ShuffleNetV2 x1.5 | 27.41 | 9.23 | 4,406,098 | 320.77M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenetv2_w3d2-0923-ea615baa.tf2.h5.log)) |
| ShuffleNetV2 x2.0 | 25.83 | 8.21 | 7,601,686 | 595.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenetv2_w2-0821-6ccac868.tf2.h5.log)) |
| ShuffleNetV2b x0.5 | 39.80 | 17.84 | 1,366,792 | 43.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenetv2b_wd2-1784-d5644a6a.tf2.h5.log)) |
| ShuffleNetV2b x1.0 | 30.36 | 11.04 | 2,279,760 | 150.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenetv2b_w1-1104-b7db0ca0.tf2.h5.log)) |
| ShuffleNetV2b x1.5 | 26.90 | 8.77 | 4,410,194 | 323.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenetv2b_w3d2-0877-9efb13f7.tf2.h5.log)) |
| ShuffleNetV2b x2.0 | 25.24 | 8.08 | 7,611,290 | 603.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenetv2b_w2-0808-ba5c7ddc.tf2.h5.log)) |
| 108-MENet-8x1 (g=3) | 43.64 | 20.39 | 654,516 | 42.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/menet108_8x1_g3-2039-1a8cfc92.tf2.h5.log)) |
| 128-MENet-8x1 (g=4) | 42.04 | 19.18 | 750,796 | 45.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/menet128_8x1_g4-1918-7fb59f0a.tf2.h5.log)) |
| 160-MENet-8x1 (g=8) | 43.48 | 20.34 | 850,120 | 45.63M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/menet160_8x1_g8-2034-3cf9eb2a.tf2.h5.log)) |
| 228-MENet-12x1 (g=3) | 33.80 | 12.91 | 1,806,568 | 152.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/menet228_12x1_g3-1291-21bd19bf.tf2.h5.log)) |
| 256-MENet-12x1 (g=4) | 32.28 | 12.17 | 1,888,240 | 150.65M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/menet256_12x1_g4-1217-d9f2e10e.tf2.h5.log)) |
| 348-MENet-12x1 (g=3) | 27.81 | 9.37 | 3,368,128 | 312.00M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/menet348_12x1_g3-0937-cee7691c.tf2.h5.log)) |
| 352-MENet-12x1 (g=8) | 31.33 | 11.67 | 2,272,872 | 157.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/menet352_12x1_g8-1167-54a916bc.tf2.h5.log)) |
| 456-MENet-24x1 (g=3) | 25.02 | 7.79 | 5,304,784 | 567.90M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/menet456_24x1_g3-0779-2a70b14b.tf2.h5.log)) |
| MobileNet x0.25 | 45.84 | 22.13 | 470,072 | 44.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/mobilenet_wd4-2213-ad04596a.tf2.h5.log)) |
| MobileNet x0.5 | 33.86 | 13.33 | 1,331,592 | 155.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/mobilenet_wd2-1333-01395e1b.tf2.h5.log)) |
| MobileNet x0.75 | 29.88 | 10.51 | 2,585,560 | 333.99M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/mobilenet_w3d4-1051-7832561b.tf2.h5.log)) |
| MobileNet x1.0 | 26.45 | 8.66 | 4,231,976 | 579.80M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/mobilenet_w1-0866-6939232b.tf2.h5.log)) |
| FD-MobileNet x0.25 | 55.42 | 30.62 | 383,160 | 12.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/fdmobilenet_wd4-3062-36aa16df.tf2.h5.log)) |
| FD-MobileNet x0.5 | 42.66 | 19.77 | 993,928 | 41.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/fdmobilenet_wd2-1977-34541b84.tf2.h5.log)) |
| FD-MobileNet x0.75 | 37.97 | 15.97 | 1,833,304 | 86.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/fdmobilenet_w3d4-1597-0123c031.tf2.h5.log)) |
| FD-MobileNet x1.0 | 33.90 | 13.12 | 2,901,288 | 147.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/fdmobilenet_w1-1312-fa99fb8d.tf2.h5.log)) |
| MobileNetV2 x0.25 | 48.10 | 24.13 | 1,516,392 | 34.24M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/mobilenetv2_wd4-2413-c3705f55.tf2.h5.log)) |
| MobileNetV2 x0.5 | 35.62 | 14.46 | 1,964,736 | 100.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/mobilenetv2_wd2-1446-b0c9a98b.tf2.h5.log)) |
| MobileNetV2 x0.75 | 29.75 | 10.44 | 2,627,592 | 198.50M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/mobilenetv2_w3d4-1044-e122c73e.tf2.h5.log)) |
| MobileNetV2 x1.0 | 26.80 | 8.63 | 3,504,960 | 329.36M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/mobilenetv2_w1-0863-b32cede3.tf2.h5.log)) |
| IGCV3 x0.25 | 53.38 | 28.28 | 1,534,020 | 41.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/igcv3_wd4-2828-309359dc.tf2.h5.log)) |
| IGCV3 x0.5 | 39.36 | 17.01 | 1,985,528 | 111.12M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/igcv3_wd2-1701-b952333a.tf2.h5.log)) |
| IGCV3 x0.75 | 30.74 | 11.00 | 2,638,084 | 210.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/igcv3_w3d4-1100-00294c7b.tf2.h5.log)) |
| IGCV3 x1.0 | 27.70 | 8.99 | 3,491,688 | 340.79M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/igcv3_w1-0899-a0cb775d.tf2.h5.log)) |
| MnasNet-B1 | 25.72 | 8.02 | 4,383,312 | 326.30M |  From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/mnasnet_b1-0802-763d6849.tf2.h5.log)) |
| MnasNet-A1 | 25.02 | 7.56 | 3,887,038 | 326.07M |  From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/mnasnet_a1-0756-8e0f4948.tf2.h5.log)) |

[dmlc/gluon-cv]: https://github.com/dmlc/gluon-cv
[tornadomeet/ResNet]: https://github.com/tornadomeet/ResNet
[Cadene/pretrained...pytorch]: https://github.com/Cadene/pretrained-models.pytorch
[tensorpack/tensorpack]: https://github.com/tensorpack/tensorpack
[clavichord93/MENet]: https://github.com/clavichord93/MENet
[zeusees/Mnasnet...Model]: https://github.com/zeusees/Mnasnet-Pretrained-Model
[soeaver/mxnet-model]: https://github.com/soeaver/mxnet-model