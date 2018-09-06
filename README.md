# Large-scale image classification networks for embedded systems
This repository contains several classification models on MXNet/Gluon, PyTorch, and Chainer, with scripts for trainig/converting models.
All models are designed for using with ImageNet-1k dataset.

## Requirements
All models/scripts are on Python. Tested on Python 2.7 and 3.6.

### For Gluon way
If you only want to use models:
```
mxnet >= 1.2.1
```
If you want also training models:
```
gluoncv >= 0.2.0
```

### For PyTorch way
```
torch >= 0.4.1
torchvision >= 0.2.1
```

### For Chainer way
```
chainer >= 5.0.0b4
chainercv >= 0.10.0
```

## List of models

- ResNet (['Deep Residual Learning for Image Recognition'](https://arxiv.org/abs/1512.03385))
- PreResNet (['Identity Mappings in Deep Residual Networks'](https://arxiv.org/abs/1603.05027))
- ResNeXt (['Aggregated Residual Transformations for Deep Neural Networks'](http://arxiv.org/abs/1611.05431))
- SENet/SE-ResNet/SE-PreResNet/SE-ResNeXt (['Squeeze-and-Excitation Networks'](https://arxiv.org/abs/1709.01507))
- DenseNet (['Densely Connected Convolutional Networks'](https://arxiv.org/abs/1608.06993))
- CondenseNet (['CondenseNet: An Efficient DenseNet using Learned Group Convolutions'](https://arxiv.org/abs/1711.09224))
- DPN (['Dual Path Networks'](https://arxiv.org/abs/1707.01629))
- DarkNet (['Darknet: Open source neural networks in c'](https://github.com/pjreddie/darknet)) 
- SqueezeNet (['SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size'](https://arxiv.org/abs/1602.07360))
- SqueezeNext (['SqueezeNext: Hardware-Aware Neural Network Design'](https://arxiv.org/abs/1803.10615))
- ShuffleNet (['ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices'](https://arxiv.org/abs/1707.01083))
- ShuffleNetV2 (['ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design'](https://arxiv.org/abs/1807.11164))
- MENet (['Merging and Evolution: Improving Convolutional Neural Networks for Mobile Applications'](https://arxiv.org/abs/1803.09127))
- MobileNet (['MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications'](https://arxiv.org/abs/1704.04861))
- FD-MobileNet (['FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy'](https://arxiv.org/abs/1802.03750))
- MobileNetV2 (['MobileNetV2: Inverted Residuals and Linear Bottlenecks'](https://arxiv.org/abs/1801.04381))
- NASNet-A-Mobile (['Learning Transferable Architectures for Scalable Image Recognition'](https://arxiv.org/abs/1707.07012))

## Pretrained models

Some remarks:
- All pretrained models can be downloaded automatically during use (use the parameter `pretrained`).
- Top1/Top5 are the standard 1-crop Top-1/Top-5 errors (in percents) on the validation subset of the ImageNet1k dataset.
- ResNet/PreResNet with b-suffix is a version of the networks with the stride in the second convolution of the
bottleneck block. Respectively a network without b-suffix has the stride in the first convolution.
- ResNet/PreResNet models do not use biasses in convolutions at all.
- CondenseNet models are only so-called converted versions.
- All models have an input 224x224 with ordinary normalization.

### For Gluon

| Model | Top1 | Top5 | Params | FLOPs | Remarks |
| --- | ---: | ---: | ---: | ---: | --- |
| ResNet-10 | 37.09 | 15.55 | 5,418,792 | 892.62M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet10-1555-cfb0a76d.params.log)) |
| ResNet-12 | 35.86 | 14.46 | 5,492,776 | 1,124.23M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.30/resnet12-1446-9ce715b0.params.log)) |
| ResNet-14 | 36.17 | 14.52 | 5,788,200 | 1,355.64M | 1-stage training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet14-1452-70faeeaa.params.log)) |
| ResNet-16 | 33.57 | 12.50 | 6,968,872 | 1,586.95M | 1-stage training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet16-1250-fc901840.params.log)) |
| ResNet-18 x0.25 | 53.13 | 27.77 | 831,096 | 136.64M | 1-stage training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet18_wd4-2777-42c5a34c.params.log)) |
| ResNet-18 x0.5 | 38.94 | 16.46 | 3,055,880 | 485.22M | 1-stage training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet18_wd2-1646-99006438.params.log)) |
| ResNet-18 x0.75 | 33.25 | 12.54 | 6,675,352 | 1,045.75M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.18/resnet18_w3d4-1254-d6548612.params.log)) |
| ResNet-18 | 29.13 | 9.94 | 11,689,512 | 1,818.21M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet18-0994-ae25f2b2.params.log)) |
| ResNet-34 | 25.34 | 7.92 | 21,797,672 | 3,669.16M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet34-0792-5b875f49.params.log)) |
| ResNet-50 | 23.50 | 6.87 | 25,557,032 | 3,868.96M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet50-0687-79fae958.params.log)) |
| ResNet-50b | 22.92 | 6.44 | 25,557,032 | 4,100.70M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet50b-0644-27a36c02.params.log)) |
| ResNet-101 | 21.66 | 5.99 | 44,549,160 | 7,586.30M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet101-0599-a6d3a5f4.params.log)) |
| ResNet-101b | 21.18 | 5.60 | 44,549,160 | 7,818.04M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet101b-0560-6517274e.params.log)) |
| ResNet-152 | 21.01 | 5.61 | 60,192,808 | 11,304.85M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet152-0561-d05971c8.params.log)) |
| ResNet-152b | 20.54 | 5.37 | 60,192,808 | 11,536.58M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet152b-0537-4f5bd879.params.log)) |
| PreResNet-18 | 29.45 | 10.29 | 11,687,848 | 1,818.41M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet18-1029-26f46f0b.params.log)) |
| PreResNet-34 | 25.88 | 8.11 | 21,796,008 | 3,669.36M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet34-0811-f8fe98a2.params.log)) |
| PreResNet-50 | 23.39 | 6.68 | 25,549,480 | 3,869.16M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet50-0668-4940c94b.params.log)) |
| PreResNet-50b | 23.16 | 6.64 | 25,549,480 | 4,100.90M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet50b-0664-2fcfddb1.params.log)) |
| PreResNet-101 | 21.45 | 5.75 | 44,541,608 | 7,586.50M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet101-0575-e2887e53.params.log)) |
| PreResNet-101b | 21.73 | 5.88 | 44,541,608 | 7,818.24M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet101b-0588-1015145a.params.log)) |
| PreResNet-152 | 20.70 | 5.32 | 60,185,256 | 11,305.05M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.14/preresnet152-0532-31505f71.params.log)) |
| PreResNet-152b | 21.00 | 5.75 | 60,185,256 | 11,536.78M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet152b-0575-dc303191.params.log)) |
| ResNeXt-101 (32x4d) | 21.32 | 5.79 | 44,177,704 | 7,991.62M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.10/resnext101_32x4d-0579-9afbfdbc.params.log)) |
| ResNeXt-101 (64x4d) | 20.60 | 5.41 | 83,455,272 | 15,491.88M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.10/resnext101_64x4d-0541-0d4fd87b.params.log)) |
| SE-ResNet-50 | 22.51 | 6.44 | 28,088,024 | 3,877.01M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.11/seresnet50-0644-10954a84.params.log)) |
| SE-ResNet-101 | 21.92 | 5.89 | 49,326,872 | 7,600.01M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.11/seresnet101-0589-4c10238d.params.log)) |
| SE-ResNet-152 | 21.48 | 5.77 | 66,821,848 | 11,324.62M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.11/seresnet152-0577-de6f099d.params.log)) |
| SE-ResNeXt-50 (32x4d) | 21.06 | 5.58 | 27,559,896 | 4,253.33M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.12/seresnext50_32x4d-0558-a49f8fb0.params.log)) |
| SE-ResNeXt-101 (32x4d) | 19.99 | 5.00 | 48,955,416 | 8,005.33M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.12/seresnext101_32x4d-0500-cf161260.params.log)) |
| SENet-154 | 18.84 | 4.65 | 115,088,984 | 20,742.40M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.13/senet154-0465-dd244507.params.log)) |
| DenseNet-121 | 25.11 | 7.80 | 7,978,856 | 2,852.39M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.3/densenet121-0780-49b72d04.params.log)) |
| DenseNet-161 | 22.40 | 6.18 | 28,681,000 | 7,761.25M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.3/densenet161-0618-52e30516.params.log)) |
| DenseNet-169 | 23.89 | 6.89 | 14,149,480 | 3,381.48M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.3/densenet169-0689-281ec06b.params.log)) |
| DenseNet-201 | 22.71 | 6.36 | 20,013,928 | 4,318.75M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.3/densenet201-0636-65b5d389.params.log)) |
| CondenseNet-74 (C=G=4) | 26.82 | 8.64 | 4,773,944 | 533.64M | From [ShichenLiu/CondenseNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.4/condensenett74_c4_g4-0864-cde68fa2.params.log)) |
| CondenseNet-74 (C=G=8) | 29.76 | 10.49 | 2,935,416 | 278.55M | From [ShichenLiu/CondenseNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.4/condensenett74_c8_g8-1049-4cf4a08e.params.log)) |
| DPN-68 | 23.57 | 7.00 | 12,611,602 | 2,338.71M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.17/dpn68-0700-3114719d.params.log)) |
| DPN-98 | 20.23 | 5.28 | 61,570,728 | 11,702.80M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.17/dpn98-0528-fa5d6fca.params.log)) |
| DPN-131 | 20.03 | 5.22 | 79,254,504 | 16,056.22M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.17/dpn131-0522-35ac2f82.params.log)) |
| DarkNet Tiny | 43.36 | 19.46 | 1,042,104 | 496.34M | Trained ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.32/darknet_tiny-1946-c5cda790.params.log)) |
| SqueezeNet v1.0 | 40.97 | 18.96 | 1,248,424 | 828.30M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.19/squeezenet_v1_0-1896-b69a4607.params.log)) |
| SqueezeNet v1.1 | 41.37 | 19.20 | 1,235,496 | 354.88M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.15/squeezenet_v1_1-1920-6d01104e.params.log)) |
| 108-MENet-8x1 (g=3) | 46.11 | 22.37 | 654,516 | 40.64M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet108_8x1_g3-2237-d3bb5a4f.params.log)) |
| 128-MENet-8x1 (g=4) | 45.80 | 21.93 | 750,796 | 43.58M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet128_8x1_g4-2193-fe760f0d.params.log)) |
| 228-MENet-12x1 (g=3) | 35.03 | 13.99 | 1,806,568 | 148.93M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet228_12x1_g3-1399-8c28d22f.params.log)) |
| 256-MENet-12x1 (g=4) | 34.49 | 13.90 | 1,888,240 | 146.11M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet256_12x1_g4-1390-4502f223.params.log)) |
| 348-MENet-12x1 (g=3) | 31.17 | 11.41 | 3,368,128 | 306.31M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet348_12x1_g3-1141-ac69b246.params.log)) |
| 352-MENet-12x1 (g=8) | 34.70 | 13.75 | 2,272,872 | 151.03M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet352_12x1_g8-1375-85779b8a.params.log)) |
| 456-MENet-24x1 (g=3) | 29.57 | 10.43 | 5,304,784 | 560.72M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet456_24x1_g3-1043-6e777068.params.log)) |
| MobileNet x0.25 | 48.37 | 24.10 | 470,072 | 42.30M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.7/mobilenet_wd4-2410-db312a26.params.log)) |
| MobileNet x0.5 | 37.37 | 15.37 | 1,331,592 | 152.04M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.7/mobilenet_wd2-1537-5419ccc2.params.log)) |
| MobileNet x0.75 | 32.71 | 12.28 | 2,585,560 | 329.22M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.7/mobilenet_w3d4-1228-dc11727a.params.log)) |
| MobileNet x1.0 | 29.25 | 10.03 | 4,231,976 | 573.83M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.7/mobilenet_w1-1003-b4fb8f1b.params.log)) |
| FD-MobileNet x0.25 | 56.73 | 31.99 | 383,160 | 12.44M | From [clavichord93/FD-MobileNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.8/fdmobilenet_wd4-3199-351c0023.params.log)) |
| FD-MobileNet x0.5 | 44.66 | 21.08 | 993,928 | 40.93M | From [clavichord93/FD-MobileNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.8/fdmobilenet_wd2-2108-21376755.params.log)) |
| FD-MobileNet x1.0 | 35.95 | 14.72 | 2,901,288 | 146.08M | From [clavichord93/FD-MobileNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.8/fdmobilenet_w1-1472-a525b206.params.log)) |
| MobileNetV2 x0.25 | 48.89 | 25.24 | 1,516,392 | 32.22M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.9/mobilenetv2_wd4-2524-a2468611.params.log)) |
| MobileNetV2 x0.5 | 35.51 | 14.64 | 1,964,736 | 95.62M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.9/mobilenetv2_wd2-1464-02fe7ff2.params.log)) |
| MobileNetV2 x0.75 | 30.82 | 11.26 | 2,627,592 | 191.61M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.9/mobilenetv2_w3d4-1126-152672f5.params.log)) |
| MobileNetV2 x1.0 | 28.51 | 9.90 | 3,504,960 | 320.19M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.9/mobilenetv2_w1-0990-4e1a3878.params.log)) |
| NASNet-A-Mobile | 26.14 | 8.26 | 5,289,978 | 587.29M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.16/nasnet_a_mobile-0826-1871d8e7.params.log)) |

### For PyTorch

| Model | Top1 | Top5 | Params | FLOPs | Remarks |
| --- | ---: | ---: | ---: | ---: | --- |
| ResNet-10 | 37.46 | 15.85 | 5,418,792 | 892.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet10-1585-ef8a3ae3.pth.log)) |
| ResNet-12 | 36.18 | 14.80 | 5,492,776 | 1,124.23M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.30/resnet12-1480-c2263f73.pth.log)) |
| ResNet-14 | 36.50 | 14.84 | 5,788,200 | 1,355.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet14-1484-542e6bd4.pth.log)) |
| ResNet-16 | 33.73 | 12.87 | 6,968,872 | 1,586.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet16-1287-bdb0b7fa.pth.log)) |
| ResNet-18 x0.25 | 53.37 | 28.06 | 831,096 | 136.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet18_wd4-2806-d0cda855.pth.log)) |
| ResNet-18 x0.5 | 39.31 | 16.79 | 3,055,880 | 485.22M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet18_wd2-1679-12f81d73.pth.log)) |
| ResNet-18 x0.75 | 33.61 | 12.85 | 6,675,352 | 1,045.75M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.18/resnet18_w3d4-1285-94713e0e.pth.log)) |
| ResNet-18 | 29.52 | 10.21 | 11,689,512 | 1,818.21M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet18-1021-b0d7daea.pth.log)) |
| ResNet-34 | 25.66 | 8.18 | 21,797,672 | 3,669.16M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet34-0818-6f947d40.pth.log)) |
| ResNet-50 | 23.79 | 7.05 | 25,557,032 | 3,868.96M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet50-0705-f7a2027e.pth.log)) |
| ResNet-50b | 23.05 | 6.65 | 25,557,032 | 4,100.70M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet50b-0665-89691746.pth.log)) |
| ResNet-101 | 21.90 | 6.22 | 44,549,160 | 7,586.30M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet101-0622-ab0cf005.pth.log)) |
| ResNet-101b | 21.45 | 5.81 | 44,549,160 | 7,818.04M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet101b-0581-d983e682.pth.log)) |
| ResNet-152 | 21.26 | 5.82 | 60,192,808 | 11,304.85M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet152-0582-af1a3bd5.pth.log)) |
| ResNet-152b | 20.74 | 5.50 | 60,192,808 | 11,536.58M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet152b-0550-216604cf.pth.log)) |
| PreResNet-18 | 29.76 | 10.57 | 11,687,848 | 1,818.41M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet18-1057-119bd3de.pth.log)) |
| PreResNet-34 | 26.23 | 8.41 | 21,796,008 | 3,669.36M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet34-0841-b4dd761f.pth.log)) |
| PreResNet-50 | 23.70 | 6.85 | 25,549,480 | 3,869.16M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet50-0685-d81a7aca.pth.log)) |
| PreResNet-50b | 23.33 | 6.87 | 25,549,480 | 4,100.90M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet50b-0687-65be98fb.pth.log)) |
| PreResNet-101 | 21.74 | 5.91 | 44,541,608 | 7,586.50M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet101-0591-4bacff79.pth.log)) |
| PreResNet-101b | 21.95 | 6.03 | 44,541,608 | 7,818.24M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet101b-0603-b1e37a09.pth.log)) |
| PreResNet-152 | 20.94 | 5.55 | 60,185,256 | 11,305.05M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.14/preresnet152-0555-c842a030.pth.log)) |
| PreResNet-152b | 21.34 | 5.91 | 60,185,256 | 11,536.78M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet152b-0591-2c91ab2c.pth.log)) |
| ResNeXt-101 (32x4d) | 21.81 | 6.11 | 44,177,704 | 7,991.62M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.10/resnext101_32x4d-0611-cf962440.pth.log)) |
| ResNeXt-101 (64x4d) | 21.04 | 5.75 | 83,455,272 | 15,491.88M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.10/resnext101_64x4d-0575-651abd02.pth.log)) |
| SE-ResNet-50 | 22.47 | 6.40 | 28,088,024 | 3,877.01M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.11/seresnet50-0640-8820f2af.pth.log)) |
| SE-ResNet-101 | 21.88 | 5.89 | 49,326,872 | 7,600.01M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.11/seresnet101-0589-5e6e831b.pth.log)) |
| SE-ResNet-152 | 21.48 | 5.76 | 66,821,848 | 11,324.62M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.11/seresnet152-0576-814cf72e.pth.log)) |
| SE-ResNeXt-50 (32x4d) | 21.00 | 5.54 | 27,559,896 | 4,253.33M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.12/seresnext50_32x4d-0554-99e0e9aa.pth.log)) |
| SE-ResNeXt-101 (32x4d) | 19.96 | 5.05 | 48,955,416 | 8,005.33M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.12/seresnext101_32x4d-0505-0924f0a2.pth.log)) |
| SENet-154 | 18.62 | 4.61 | 115,088,984 | 20,742.40M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.13/senet154-0461-6512228c.pth.log)) |
| DenseNet-121 | 25.57 | 8.03 | 7,978,856 | 2,852.39M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.3/densenet121-0803-f994107a.pth.log)) |
| DenseNet-161 | 22.86 | 6.44 | 28,681,000 | 7,761.25M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.3/densenet161-0644-c0fb22c8.pth.log)) |
| DenseNet-169 | 24.40 | 7.19 | 14,149,480 | 3,381.48M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.3/densenet169-0719-27139105.pth.log)) |
| DenseNet-201 | 23.10 | 6.63 | 20,013,928 | 4,318.75M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.3/densenet201-0663-71ece4ad.pth.log)) |
| CondenseNet-74 (C=G=4) | 26.25 | 8.28 | 4,773,944 | 533.64M | From [ShichenLiu/CondenseNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.4/condensenett74_c4_g4-0828-5ba55049.pth.log)) |
| CondenseNet-74 (C=G=8) | 28.93 | 10.06 | 2,935,416 | 278.55M | From [ShichenLiu/CondenseNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.4/condensenett74_c8_g8-1006-3574d874.pth.log)) |
| DPN-68 | 24.17 | 7.27 | 12,611,602 | 2,338.71M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.17/dpn68-0727-43849233.pth.log)) |
| DPN-98 | 20.81 | 5.53 | 61,570,728 | 11,702.80M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.17/dpn98-0553-52c55969.pth.log)) |
| DPN-131 | 20.54 | 5.48 | 79,254,504 | 16,056.22M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.17/dpn131-0548-0c53e5b3.pth.log)) |
| DarkNet Tiny | 43.65 | 19.80 | 1,042,104 | 496.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.32/darknet_tiny-1980-0467ab13.pth.log)) |
| SqueezeNet v1.0 | 41.31 | 19.32 | 1,248,424 | 828.30M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.19/squeezenet_v1_0-1932-e4017303.pth.log)) |
| SqueezeNet v1.1 | 41.82 | 19.38 | 1,235,496 | 354.88M | Converted from TorchVision ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.5/squeezenet_v1_1-1938-8dcd1cc5.pth.log)) |
| 108-MENet-8x1 (g=3) | 43.92 | 20.76 | 654,516 | 40.64M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet108_8x1_g3-2076-7f47b37e.pth.log)) |
| 128-MENet-8x1 (g=4) | 43.95 | 20.62 | 750,796 | 43.58M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet128_8x1_g4-2062-dd4531fd.pth.log)) |
| 228-MENet-12x1 (g=3) | 33.57 | 13.28 | 1,806,568 | 148.93M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet228_12x1_g3-1328-27991387.pth.log)) |
| 256-MENet-12x1 (g=4) | 33.41 | 13.26 | 1,888,240 | 146.11M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet256_12x1_g4-1326-e5d35476.pth.log)) |
| 348-MENet-12x1 (g=3) | 30.10 | 10.92 | 3,368,128 | 306.31M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet348_12x1_g3-1092-66be1a18.pth.log)) |
| 352-MENet-12x1 (g=8) | 33.31 | 13.08 | 2,272,872 | 151.03M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet352_12x1_g8-1308-e91ec72c.pth.log)) |
| 456-MENet-24x1 (g=3) | 28.40 | 9.93 | 5,304,784 | 560.72M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet456_24x1_g3-0993-cb9fd376.pth.log)) |
| MobileNet x0.25 | 49.13 | 24.93 | 470,072 | 42.30M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.7/mobilenet_wd4-2493-c05b5fab.pth.log)) |
| MobileNet x0.5 | 38.12 | 15.99 | 1,331,592 | 152.04M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.7/mobilenet_wd2-1599-5883b38d.pth.log)) |
| MobileNet x0.75 | 33.54 | 12.85 | 2,585,560 | 329.22M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.7/mobilenet_w3d4-1285-b8022fae.pth.log)) |
| MobileNet x1.0 | 29.86 | 10.36 | 4,231,976 | 573.83M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.7/mobilenet_w1-1036-34f7a0cb.pth.log)) |
| FD-MobileNet x0.25 | 55.77 | 31.32 | 383,160 | 12.44M | From [clavichord93/FD-MobileNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.8/fdmobilenet_wd4-3132-0b242eff.pth.log)) |
| FD-MobileNet x0.5 | 43.85 | 20.72 | 993,928 | 40.93M | From [clavichord93/FD-MobileNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.8/fdmobilenet_wd2-2072-884550e9.pth.log)) |
| FD-MobileNet x1.0 | 34.70 | 14.05 | 2,901,288 | 146.08M | From [clavichord93/FD-MobileNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.8/fdmobilenet_w1-1405-a6538879.pth.log)) |
| MobileNetV2 x0.25 | 49.72 | 25.87 | 1,516,392 | 32.22M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.9/mobilenetv2_wd4-2587-189d4ea2.pth.log)) |
| MobileNetV2 x0.5 | 36.54 | 15.19 | 1,964,736 | 95.62M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.9/mobilenetv2_wd2-1519-d0937a23.pth.log)) |
| MobileNetV2 x0.75 | 31.89 | 11.76 | 2,627,592 | 191.61M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.9/mobilenetv2_w3d4-1176-1b966ff4.pth.log)) |
| MobileNetV2 x1.0 | 29.31 | 10.39 | 3,504,960 | 320.19M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.9/mobilenetv2_w1-1039-7532eb72.pth.log)) |
| NASNet-A-Mobile | 26.27 | 8.45 | 5,289,978 | 587.29M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.16/nasnet_a_mobile-0845-ccc5284e.pth.log)) |

### For Chainer

| Model | Top1 | Top5 | Params | FLOPs | Remarks |
| --- | ---: | ---: | ---: | ---: | --- |
| ResNet-10 | 37.12 | 15.49 | 5,418,792 | 892.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet10-1549-b31f1135.npz.log)) |
| ResNet-12 | 35.86 | 14.48 | 5,492,776 | 1,124.23M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.30/resnet12-1448-11acb729.npz.log)) |
| ResNet-14 | 36.20 | 14.54 | 5,788,200 | 1,355.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet14-1454-7c69aaa0.npz.log)) |
| ResNet-16 | 33.61 | 12.53 | 6,968,872 | 1,586.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet16-1253-6e751065.npz.log)) |
| ResNet-18 x0.25 | 53.17 | 27.70 | 831,096 | 136.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet18_wd4-2770-72465bfe.npz.log)) |
| ResNet-18 x0.5 | 38.95 | 16.46 | 3,055,880 | 485.22M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet18_wd2-1646-58261fc5.npz.log)) |
| ResNet-18 x0.75 | 33.27 | 12.56 | 6,675,352 | 1,045.75M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet18_w3d4-1256-ce2011df.npz.log)) |
| ResNet-18 | 29.08 | 9.97 | 11,689,512 | 1,818.21M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet18-0997-9862a84f.npz.log)) |
| ResNet-34 | 25.35 | 7.95 | 21,797,672 | 3,669.16M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet34-0795-0b392267.npz.log)) |
| ResNet-50 | 23.50 | 6.83 | 25,557,032 | 3,868.96M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet50-0683-9c795737.npz.log)) |
| ResNet-50b | 22.93 | 6.46 | 25,557,032 | 4,100.70M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet50b-0646-225a550e.npz.log)) |
| ResNet-101 | 21.65 | 6.01 | 44,549,160 | 7,586.30M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet101-0601-d8cddbea.npz.log)) |
| ResNet-101b | 21.16 | 5.59 | 44,549,160 | 7,818.04M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet101b-0559-b5c3b4b6.npz.log)) |
| ResNet-152 | 21.07 | 5.67 | 60,192,808 | 11,304.85M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet152-0567-62d194fc.npz.log)) |
| ResNet-152b | 20.44 | 5.39 | 60,192,808 | 11,536.58M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet152b-0539-2b175728.npz.log)) |
| PreResNet-18 | 29.43 | 10.34 | 11,687,848 | 1,818.41M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet18-1034-7d174fc2.npz.log)) |
| PreResNet-34 | 25.89 | 8.12 | 21,796,008 | 3,669.36M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet34-0812-829f5a23.npz.log)) |
| PreResNet-50 | 23.36 | 6.69 | 25,549,480 | 3,869.16M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet50-0669-40bd5e93.npz.log)) |
| PreResNet-50b | 23.08 | 6.67 | 25,549,480 | 4,100.90M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet50b-0667-b7d221ef.npz.log)) |
| PreResNet-101 | 21.45 | 5.75 | 44,541,608 | 7,586.50M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet101-0575-f6f6789a.npz.log)) |
| PreResNet-101b | 21.61 | 5.87 | 44,541,608 | 7,818.24M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet101b-0587-4211c5ab.npz.log)) |
| PreResNet-152 | 20.73 | 5.30 | 60,185,256 | 11,305.05M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet152-0530-021d99dc.npz.log)) |
| PreResNet-152b | 20.88 | 5.66 | 60,185,256 | 11,536.78M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet152b-0566-fdd337e7.npz.log)) |
| ResNeXt-101 (32x4d) | 21.11 | 5.69 | 44,177,704 | 7,991.62M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.26/resnext101_32x4d-0569-c6d1c30d.npz.log)) |
| ResNeXt-101 (64x4d) | 20.57 | 5.43 | 83,455,272 | 15,491.88M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.26/resnext101_64x4d-0543-dd8b7d96.npz.log)) |
| SE-ResNet-50 | 22.53 | 6.41 | 28,088,024 | 3,877.01M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.24/seresnet50-0641-f3d68cfc.npz.log)) |
| SE-ResNet-101 | 21.90 | 5.88 | 49,326,872 | 7,600.01M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.24/seresnet101-0588-e45a9f8f.npz.log)) |
| SE-ResNet-152 | 21.46 | 5.77 | 66,821,848 | 11,324.62M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.24/seresnet152-0577-a089ba52.npz.log)) |
| SE-ResNeXt-50 (32x4d) | 21.04 | 5.58 | 27,559,896 | 4,253.33M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.27/seresnext50_32x4d-0558-5c435c1b.npz.log)) |
| SE-ResNeXt-101 (32x4d) | 19.99 | 5.01 | 48,955,416 | 8,005.33M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.27/seresnext101_32x4d-0501-98ea6fc4.npz.log)) |
| SENet-154 | 18.79 | 4.63 | 115,088,984 | 20,742.40M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.28/senet154-0463-381d2494.npz.log)) |
| DenseNet-121 | 25.04 | 7.79 | 7,978,856 | 2,852.39M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.29/densenet121-0779-06d5ebbf.npz.log)) |
| DenseNet-161 | 22.36 | 6.20 | 28,681,000 | 7,761.25M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.29/densenet161-0620-6d05f3b9.npz.log)) |
| DenseNet-169 | 23.85 | 6.86 | 14,149,480 | 3,381.48M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.29/densenet169-0686-1978656b.npz.log)) |
| DenseNet-201 | 22.64 | 6.29 | 20,013,928 | 4,318.75M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.29/densenet201-0629-77702939.npz.log)) |
| DarkNet Tiny | 43.31 | 19.47 | 1,042,104 | 496.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.32/darknet_tiny-1947-0ba271d4.npz.log)) |
| SqueezeNet v1.0 | 41.01 | 18.96 | 1,248,424 | 828.30M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.20/squeezenet_v1_0-1896-6cbb35ce.npz.log)) |
| SqueezeNet v1.1 | 41.36 | 19.25 | 1,235,496 | 354.88M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.20/squeezenet_v1_1-1925-0ca73cf3.npz.log)) |
| MobileNet x0.25 | 48.60 | 24.28 | 470,072 | 42.30M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.21/mobilenet_wd4-2428-21ddc10d.npz.log)) |
| MobileNet x0.5 | 37.64 | 15.66 | 1,331,592 | 152.04M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.21/mobilenet_wd2-1566-d398ee98.npz.log)) |
| MobileNet x0.75 | 33.24 | 12.52 | 2,585,560 | 329.22M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.21/mobilenet_w3d4-1252-6675b58c.npz.log)) |
| MobileNet x1.0 | 29.71 | 10.31 | 4,231,976 | 573.83M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.21/mobilenet_w1-1031-3ecb405b.npz.log)) |
| FD-MobileNet x0.25 | 56.67 | 31.96 | 383,160 | 12.44M | From [clavichord93/FD-MobileNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.25/fdmobilenet_wd4-3196-463330f8.npz.log)) |
| FD-MobileNet x0.5 | 44.67 | 21.09 | 993,928 | 40.93M | From [clavichord93/FD-MobileNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.25/fdmobilenet_wd2-2109-cc9bd695.npz.log)) |
| FD-MobileNet x1.0 | 35.94 | 14.70 | 2,901,288 | 146.08M | From [clavichord93/FD-MobileNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.25/fdmobilenet_w1-1470-b40709cb.npz.log)) |
| MobileNetV2 x0.25 | 49.11 | 25.49 | 1,516,392 | 32.22M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.31/mobilenetv2_wd4-2549-b5ff8bfd.npz.log)) |
| MobileNetV2 x0.5 | 35.96 | 14.98 | 1,964,736 | 95.62M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.31/mobilenetv2_wd2-1498-4b767a98.npz.log)) |
| MobileNetV2 x0.75 | 31.28 | 11.48 | 2,627,592 | 191.61M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.31/mobilenetv2_w3d4-1148-a6f852ea.npz.log)) |
| MobileNetV2 x1.0 | 28.87 | 10.05 | 3,504,960 | 320.19M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.31/mobilenetv2_w1-1005-3b6d1764.npz.log)) |

[Cadene/pretrained...pytorch]: https://github.com/Cadene/pretrained-models.pytorch
[ShichenLiu/CondenseNet]: https://github.com/ShichenLiu/CondenseNet
[clavichord93/MENet]: https://github.com/clavichord93/MENet
[clavichord93/FD-MobileNet]: https://github.com/clavichord93/FD-MobileNet