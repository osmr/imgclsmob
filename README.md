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

### For Gluon

| Model | Top1 | Top5 | Params | Remarks |
| --- | ---: | ---: | ---: | --- |
| ResNet-18 | 29.06 | 10.08 | 11,689,512 | Converted for Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/resnet18-1008-4f9f7e8f.params.log)) |
| ResNet-34 | 25.34 | 7.92 | 21,797,672 | Converted for Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/resnet34-0792-5b875f49.params.log)) |
| ResNet-50 | 23.50 | 6.87 | 25,557,032 | Converted for Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/resnet50-0687-79fae958.params.log)) |
| ResNet-50b | 22.92 | 6.44 | 25,557,032 | Converted for Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/resnet50b-0644-27a36c02.params.log)) |
| ResNet-101 | 21.66 | 5.99 | 44,549,160 | Converted for Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/resnet101-0599-a6d3a5f4.params.log)) |
| ResNet-101b | 21.18 | 5.60 | 44,549,160 | Converted for Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/resnet101b-0560-6517274e.params.log)) |
| ResNet-152 | 21.01 | 5.61 | 60,192,808 | Converted for Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/resnet152-0561-d05971c8.params.log)) |
| ResNet-152b | 20.54 | 5.37 | 60,192,808 | Converted for Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/resnet152b-0537-4f5bd879.params.log)) |
| PreResNet-18 | 29.45 | 10.29 | 11,687,848 | Converted for Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/preresnet18-1029-26f46f0b.params.log)) |
| PreResNet-34 | 25.88 | 8.11 | 21,796,008 | Converted for Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/preresnet34-0811-f8fe98a2.params.log)) |
| ~~PreResNet-50~~ | 40.97 | 18.29 | 25,549,480 | Converted for Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/preresnet50-1829-2fcfddb1.params.log)) |
| PreResNet-50b | 23.16 | 6.64 | 25,549,480 | Converted for Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/preresnet50b-0664-2fcfddb1.params.log)) |
| ~~PreResNet-101~~ | 39.91 | 17.46 | 44,541,608 | Converted for Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/preresnet101-1746-1015145a.params.log)) |
| PreResNet-101b | 21.73 | 5.88 | 44,541,608 | Converted for Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/preresnet101b-0588-1015145a.params.log)) |
| ~~PreResNet-152~~ | 35.88 | 14.51 | 60,185,256 | Converted for Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/preresnet152-1451-dc303191.params.log)) |
| PreResNet-152b | 21.00 | 5.75 | 60,185,256 | Converted for Gluon Model Zoo ([log](https://github.com/osmr/tmp1/releases/download/v0.0.2/preresnet152b-0575-dc303191.params.log)) |
