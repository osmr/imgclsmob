# Large-scale image classification networks for embedded systems
This repository contains several classification models on MXNet/Gluon and PyTorch and scripts for trainig/converting models. All models are designed for using with ImageNet-1k dataset.

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

## List of models

- ResNet (['Deep Residual Learning for Image Recognition'](https://arxiv.org/abs/1512.03385))
- PreResNet (['Identity Mappings in Deep Residual Networks'](https://arxiv.org/abs/1603.05027))
- DenseNet (['Densely Connected Convolutional Networks'](https://arxiv.org/abs/1608.06993))
- DarkNet (['Darknet: Open source neural networks in c'](https://github.com/pjreddie/darknet)) 
- SqueezeNet (['SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size'](https://arxiv.org/abs/1602.07360))
- SqueezeNext (['SqueezeNext: Hardware-Aware Neural Network Design'](https://arxiv.org/abs/1803.10615))
- ShuffleNet (['ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices'](https://arxiv.org/abs/1707.01083))
- MENet (['Merging and Evolution: Improving Convolutional Neural Networks for Mobile Applications'](https://arxiv.org/abs/1803.09127))
- MobileNet (['MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications'](https://arxiv.org/abs/1704.04861))
- FD-MobileNet (['FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy'](https://arxiv.org/abs/1802.03750))
- MobileNetV2 (['MobileNetV2: Inverted Residuals and Linear Bottlenecks'](https://arxiv.org/abs/1801.04381))

## Pretrained models

Some remarks:
- All pretrained models can be downloaded automatically during use (use the parameter `pretrained`).
- Top1/Top5 are the standard 1-crop Top-1/Top-5 errors (in percents) on the validation subset of the ImageNet1k dataset.
- ResNet/PreResNet with b-suffix is a version of the networks with the stride in the second convolution of the
bottleneck block. Respectively a network without b-suffix has the stride in the first convolution.
- ResNet/PreResNet models do not use biasses in convolutions at all.

### For Gluon

| Model | Top1 | Top5 | Params | FLOPs | Remarks |
| --- | ---: | ---: | ---: | ---: | --- |
| ResNet-10 | 38.88 | 16.63 | 5,418,792 | 892.62M | 1-stage training ([log](https://github.com/osmr/tmp1/releases/download/v0.0.5/resnet10-1663-6dc653d3.params.log)) |
| ResNet-12 | 37.62 | 15.56 | 5,492,776 | 1,124.23M | 1-stage training ([log](https://github.com/osmr/tmp1/releases/download/v0.0.5/resnet12-1556-6395e8b1.params.log)) |
| ResNet-14 | 36.17 | 14.52 | 5,788,200 | 1,355.64M | 1-stage training ([log](https://github.com/osmr/tmp1/releases/download/v0.0.5/resnet14-1452-70faeeaa.params.log)) |
| ResNet-18 x0.25 | 53.13 | 27.77 | 831,096 | 136.64M | 1-stage training ([log](https://github.com/osmr/tmp1/releases/download/v0.0.5/resnet18_wd4-2777-42c5a34c.params.log)) |
| ResNet-18 x0.5 | 38.94 | 16.46 | 3,055,880 | 485.22M | 1-stage training ([log](https://github.com/osmr/tmp1/releases/download/v0.0.5/resnet18_wd2-1646-99006438.params.log)) |
| ResNet-18 | 29.06 | 10.08 | 11,689,512 | 1,818.21M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/resnet18-1008-4f9f7e8f.params.log)) |
| ResNet-34 | 25.34 | 7.92 | 21,797,672 | 3,669.16M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/resnet34-0792-5b875f49.params.log)) |
| ResNet-50 | 23.50 | 6.87 | 25,557,032 | 3,868.96M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/resnet50-0687-79fae958.params.log)) |
| ResNet-50b | 22.92 | 6.44 | 25,557,032 | 4,100.70M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/resnet50b-0644-27a36c02.params.log)) |
| ResNet-101 | 21.66 | 5.99 | 44,549,160 | 7,586.30M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/resnet101-0599-a6d3a5f4.params.log)) |
| ResNet-101b | 21.18 | 5.60 | 44,549,160 | 7,818.04M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/resnet101b-0560-6517274e.params.log)) |
| ResNet-152 | 21.01 | 5.61 | 60,192,808 | 11,304.85M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/resnet152-0561-d05971c8.params.log)) |
| ResNet-152b | 20.54 | 5.37 | 60,192,808 | 11,536.58M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/resnet152b-0537-4f5bd879.params.log)) |
| PreResNet-18 | 29.45 | 10.29 | 11,687,848 | 1,818.41M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/preresnet18-1029-26f46f0b.params.log)) |
| PreResNet-34 | 25.88 | 8.11 | 21,796,008 | 3,669.36M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/preresnet34-0811-f8fe98a2.params.log)) |
| PreResNet-50 | 23.39 | 6.68 | 25,549,480 | 3,869.16M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.4/preresnet50-0668-4940c94b.params.log)) |
| PreResNet-50b | 23.16 | 6.64 | 25,549,480 | 4,100.90M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/preresnet50b-0664-2fcfddb1.params.log)) |
| ~~PreResNet-101~~ | 39.91 | 17.46 | 44,541,608 | 7,586.50M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/preresnet101-1746-1015145a.params.log)) |
| PreResNet-101b | 21.73 | 5.88 | 44,541,608 | 7,818.24M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/preresnet101b-0588-1015145a.params.log)) |
| ~~PreResNet-152~~ | 35.88 | 14.51 | 60,185,256 | 11,305.05M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/preresnet152-1451-dc303191.params.log)) |
| PreResNet-152b | 21.00 | 5.75 | 60,185,256 | 11,536.78M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/preresnet152b-0575-dc303191.params.log)) |
| SqueezeNet v1.0 | 42.81 | 19.98 | 1,248,424 | 828.30M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.6/squeezenet_v1_0-1998-1b771149.params.log)) |
| SqueezeNet v1.1 | 43.06 | 20.23 | 1,235,496 | 354.88M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.6/squeezenet_v1_1-2023-ab455761.params.log)) |

### For PyTorch

| Model | Top1 | Top5 | Params | FLOPs | Remarks |
| --- | ---: | ---: | ---: | ---: | --- |
| ResNet-10 | 39.09 | 16.90 | 5,418,792 | 892.62M | Converted from GL model ([log](https://github.com/osmr/tmp1/releases/download/v0.0.5/resnet10-1690-37e57029.pth.log)) |
| ResNet-12 | 38.02 | 15.89 | 5,492,776 | 1,124.23M | Converted from GL model ([log](https://github.com/osmr/tmp1/releases/download/v0.0.5/resnet12-1589-9552d5a8.pth.log)) |
| ResNet-12 | 36.50 | 14.84 | 5,492,776 | 1,355.64M | Converted from GL model ([log](https://github.com/osmr/tmp1/releases/download/v0.0.5/resnet14-1484-542e6bd4.pth.log)) |
| ResNet-18 x0.25 | 53.37 | 28.06 | 831,096 | 136.64M | Converted from GL model ([log](https://github.com/osmr/tmp1/releases/download/v0.0.5/resnet18_wd4-2806-d0cda855.pth.log)) |
| ResNet-18 x0.5 | 39.31 | 16.79 | 3,055,880 | 485.22M | Converted from GL model ([log](https://github.com/osmr/tmp1/releases/download/v0.0.5/resnet18_wd2-1679-12f81d73.pth.log)) |
| ResNet-18 | 29.33 | 10.30 | 11,689,512 | 1,818.21M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.3/resnet18-1030-a516bab5.pth.log)) |
| ResNet-34 | 25.66 | 8.18 | 21,797,672 | 3,669.16M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.3/resnet34-0818-6f947d40.pth.log)) |
| ResNet-50 | 23.79 | 7.05 | 25,557,032 | 3,868.96M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.3/resnet50-0705-f7a2027e.pth.log)) |
| ResNet-50b | 23.05 | 6.65 | 25,557,032 | 4,100.70M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.3/resnet50b-0665-89691746.pth.log)) |
| ResNet-101 | 21.90 | 6.22 | 44,549,160 | 7,586.30M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.3/resnet101-0622-ab0cf005.pth.log)) |
| ResNet-101b | 21.45 | 5.81 | 44,549,160 | 7,818.04M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.3/resnet101b-0581-d983e682.pth.log)) |
| ResNet-152 | 21.26 | 5.82 | 60,192,808 | 11,304.85M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.3/resnet152-0582-af1a3bd5.pth.log)) |
| ResNet-152b | 20.74 | 5.50 | 60,192,808 | 11,536.58M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.3/resnet152b-0550-216604cf.pth.log)) |
| PreResNet-18 | 29.76 | 10.57 | 11,687,848 | 1,818.41M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.3/preresnet18-1057-119bd3de.pth.log)) |
| PreResNet-34 | 26.23 | 8.41 | 21,796,008 | 3,669.36M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.3/preresnet34-0841-b4dd761f.pth.log)) |
| PreResNet-50 | 23.70 | 6.85 | 25,549,480 | 3,869.16M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.4/preresnet50-0685-d81a7aca.pth.log)) |
| PreResNet-50b | 23.33 | 6.87 | 25,549,480 | 4,100.90M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.3/preresnet50b-0687-65be98fb.pth.log)) |
| ~~PreResNet-101~~ | 40.29 | 18.08 | 44,541,608 | 7,586.50M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.3/preresnet101-1808-0340579d.pth.log)) |
| PreResNet-101b | 21.95 | 6.03 | 44,541,608 | 7,818.24M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.3/preresnet101b-0603-b1e37a09.pth.log)) |
| ~~PreResNet-152~~ | 36.35 | 14.88 | 60,185,256 | 11,305.05M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.3/preresnet152-1488-0cecb4fc.pth.log)) |
| PreResNet-152b | 21.34 | 5.91 | 60,185,256 | 11,536.78M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.3/preresnet152b-0591-2c91ab2c.pth.log)) |
| SqueezeNet v1.0 | 41.91 | 19.58 | 1,248,424 | 828.30M | Converted from TorchVision ([log](https://github.com/osmr/tmp1/releases/download/v0.0.6/squeezenet_v1_0-1958-d6d59f9c.pth.log)) |
| SqueezeNet v1.1 | 41.82 | 19.38 | 1,235,496 | 354.88M | Converted from TorchVision ([log](https://github.com/osmr/tmp1/releases/download/v0.0.6/squeezenet_v1_1-1938-8dcd1cc5.pth.log)) |
