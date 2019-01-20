# Large-scale image classification models on Keras with MXNet backend

This is a collection of large-scale image classification models. Many of them are pretrained on ImageNet-1K dataset
and loaded automatically during use. All pretrained models require the same ordinary normalization.

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

## Installation

To use the models in your project, simply install the `kerascv` package with `mxnet`:
```
pip install kerascv mxnet>=1.2.1
```
To enable different hardware supports such as GPUs, check out [MXNet variants](https://pypi.org/project/mxnet/).
For example, you can install with CUDA-9.2 supported MXNet:
```
pip install kerascv mxnet-cu92>=1.2.1
```
After installation change the value of the `image_data_format` field to `channels_first` in the file
`~/.keras/keras.json`. Also check that the `backend` field is set to `mxnet`. 

## Usage

Example of using the pretrained ResNet-18 model:
```
from kerascv.model_provider import get_model as kecv_get_model
import numpy as np

net = kecv_get_model("resnet18", pretrained=True)
x = np.zeros((1, 3, 224, 224), np.float32)
y = net.predict(x)
```

## Pretrained models

Some remarks:
- Top1/Top5 are the standard 1-crop Top-1/Top-5 errors (in percents) on the validation subset of the ImageNet-1K dataset.
- FLOPs/2 is the number of FLOPs divided by two to be similar to the number of MACs.
- Remark `Converted from GL model` means that the model was trained on `MXNet/Gluon` and then converted to Keras.

| Model | Top1 | Top5 | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | ---: | --- |
| AlexNet | 44.10 | 21.26 | 61,100,840 | 714.83M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.121/alexnet-2126-56fb1c54.h5.log)) |
| VGG-11 | 31.90 | 11.75 | 132,863,336 | 7,615.87M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.122/vgg11-1175-daa3c646.h5.log)) |
| VGG-13 | 31.06 | 11.12 | 133,047,848 | 11,317.65M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.122/vgg13-1112-90b447ec.h5.log)) |
| VGG-16 | 26.78 | 8.69 | 138,357,544 | 15,507.20M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.122/vgg16-0869-13d19be6.h5.log)) |
| VGG-19 | 25.87 | 8.23 | 143,667,240 | 19,642.55M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.122/vgg19-0823-cab851b8.h5.log)) |
| BN-VGG-11b | 30.34 | 10.57 | 132,868,840 | 7,630.72M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.123/bn_vgg11b-1057-8b6a294a.h5.log)) |
| BN-VGG-13b | 29.48 | 10.16 | 133,053,736 | 11,342.14M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.123/bn_vgg13b-1016-b26cafd3.h5.log)) |
| BN-VGG-16b | 26.88 | 8.65 | 138,365,992 | 15,507.20M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.123/bn_vgg16b-0865-2272fdd1.h5.log)) |
| BN-VGG-19b | 25.65 | 8.14 | 143,678,248 | 19,672.26M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.123/bn_vgg19b-0814-852e2ca2.h5.log)) |
| ResNet-10 | 37.09 | 15.54 | 5,418,792 | 894.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.49/resnet10-1554-294a0786.h5.log)) |
| ResNet-12 | 35.86 | 14.45 | 5,492,776 | 1,126.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.49/resnet12-1445-285da75b.h5.log)) |
| ResNet-14 | 32.85 | 12.42 | 5,788,200 | 1,357.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.49/resnet14-1242-e2ffca6e.h5.log)) |
| ResNet-16 | 30.67 | 11.09 | 6,968,872 | 1,589.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.49/resnet16-1109-8f70f97e.h5.log)) |
| ResNet-18 x0.25 | 49.14 | 24.45 | 831,096 | 137.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.49/resnet18_wd4-2445-dd6ba54d.h5.log)) |
| ResNet-18 x0.5 | 36.54 | 14.96 | 3,055,880 | 486.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.49/resnet18_wd2-1496-9bc78e3b.h5.log)) |
| ResNet-18 x0.75 | 33.24 | 12.54 | 6,675,352 | 1,047.53M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.49/resnet18_w3d4-1254-f6374cc3.h5.log)) |
| ResNet-18 | 28.08 | 9.52 | 11,689,512 | 1,820.41M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.153/resnet18-0952-0817d058.h5.log)) |
| ResNet-34 | 25.32 | 7.92 | 21,797,672 | 3,672.68M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.49/resnet34-0792-3ea662f5.h5.log)) |
| ResNet-50 | 22.63 | 6.41 | 25,557,032 | 3,877.95M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.147/resnet50-0641-38a4c231.h5.log)) |
| ResNet-50b | 22.31 | 6.18 | 25,557,032 | 4,110.48M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.146/resnet50b-0618-6be0de5f.h5.log)) |
| ResNet-101 | 21.64 | 5.99 | 44,549,160 | 7,597.95M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.49/resnet101-0599-ab428947.h5.log)) |
| ResNet-101b | 20.78 | 5.39 | 44,549,160 | 7,830.48M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.145/resnet101b-0539-2d572d9b.h5.log)) |
| ResNet-152 | 20.74 | 5.35 | 60,192,808 | 11,321.85M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.144/resnet152-0535-43ecb2b0.h5.log)) |
| ResNet-152b | 20.30 | 5.25 | 60,192,808 | 11,554.38M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.143/resnet152b-0525-c34915fe.h5.log)) |
| PreResNet-18 | 28.16 | 9.52 | 11,687,848 | 1,820.56M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.140/preresnet18-0952-b88bf767.h5.log)) |
| PreResNet-34 | 25.86 | 8.11 | 21,796,008 | 3,672.83M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.50/preresnet34-0811-1663d695.h5.log)) |
| PreResNet-50 | 23.38 | 6.68 | 25,549,480 | 3,875.44M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.50/preresnet50-0668-90326d19.h5.log)) |
| PreResNet-50b | 23.14 | 6.63 | 25,549,480 | 4,107.97M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.50/preresnet50b-0663-c30588ee.h5.log)) |
| PreResNet-101 | 21.43 | 5.75 | 44,541,608 | 7,595.44M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.50/preresnet101-0575-5dff088d.h5.log)) |
| PreResNet-101b | 21.71 | 5.88 | 44,541,608 | 7,827.97M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.50/preresnet101b-0588-fad1f60c.h5.log)) |
| PreResNet-152 | 20.69 | 5.31 | 60,185,256 | 11,319.34M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.50/preresnet152-0531-a5ac128d.h5.log)) |
| PreResNet-152b | 20.99 | 5.76 | 60,185,256 | 11,551.87M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.50/preresnet152b-0576-ea9dda1e.h5.log)) |
| PreResNet-200b | 21.09 | 5.64 | 64,666,280 | 15,068.63M | From [tornadomeet/ResNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.50/preresnet200b-0564-9172d4c0.h5.log)) |
| ResNeXt-101 (32x4d) | 21.30 | 5.78 | 44,177,704 | 8,003.45M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.51/resnext101_32x4d-0578-7623f640.h5.log)) |
| ResNeXt-101 (64x4d) | 20.59 | 5.41 | 83,455,272 | 15,500.27M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.51/resnext101_64x4d-0541-7b58eaae.h5.log)) |
| SE-ResNet-50 | 22.50 | 6.43 | 28,088,024 | 3,880.49M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.52/seresnet50-0643-fabfa406.h5.log)) |
| SE-ResNet-101 | 21.92 | 5.88 | 49,326,872 | 7,602.76M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.52/seresnet101-0588-933d3415.h5.log)) |
| SE-ResNet-152 | 21.46 | 5.77 | 66,821,848 | 11,328.52M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.52/seresnet152-0577-d25ced7d.h5.log)) |
| SE-ResNeXt-50 (32x4d) | 21.05 | 5.57 | 27,559,896 | 4,258.40M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.53/seresnext50_32x4d-0557-997ef4dd.h5.log)) |
| SE-ResNeXt-101 (32x4d) | 19.98 | 4.99 | 48,955,416 | 8,008.26M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.53/seresnext101_32x4d-0499-59e4e584.h5.log)) |
| SENet-154 | 18.83 | 4.65 | 115,088,984 | 20,745.78M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.54/senet154-0465-962aeede.h5.log)) |
| DenseNet-121 | 25.09 | 7.80 | 7,978,856 | 2,872.13M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.55/densenet121-0780-52b0611c.h5.log)) |
| DenseNet-161 | 22.39 | 6.18 | 28,681,000 | 7,793.16M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.55/densenet161-0618-070fcb45.h5.log)) |
| DenseNet-169 | 23.88 | 6.89 | 14,149,480 | 3,403.89M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.55/densenet169-0689-ae41b4a6.h5.log)) |
| DenseNet-201 | 22.69 | 6.35 | 20,013,928 | 4,347.15M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.55/densenet201-0635-cf3afbb2.h5.log)) |
| DarkNet Tiny | 40.31 | 17.46 | 1,042,104 | 500.85M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.69/darknet_tiny-1746-147e949b.h5.log)) |
| DarkNet Ref | 37.99 | 16.68 | 7,319,416 | 367.59M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.64/darknet_ref-1668-2ef080bb.h5.log)) |
| DarkNet-53 | 21.43 | 5.56 | 41,609,928 | 7,133.86M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.150/darknet53-0556-d6c6e7dc.h5.log)) |
| SqueezeNet v1.0 | 39.17 | 17.56 | 1,248,424 | 823.67M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.128/squeezenet_v1_0-1756-a4890923.h5.log)) |
| SqueezeNet v1.1 | 39.08 | 17.39 | 1,235,496 | 352.02M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.88/squeezenet_v1_1-1739-b9a8f9ea.h5.log)) |
| SqueezeResNet v1.1 | 39.82 | 17.84 | 1,235,496 | 352.02M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.70/squeezeresnet_v1_1-1784-43ee9cbb.h5.log)) |
| 1.0-SqNxt-23 | 42.28 | 18.62 | 724,056 | 287.28M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.171/sqnxt23_w1-1862-cab60636.h5.log)) |
| 1.0-SqNxt-23v5 | 40.38 | 17.57 | 921,816 | 285.82M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.172/sqnxt23v5_w1-1757-96b94e1d.h5.log)) |
| ShuffleNet x0.25 (g=1) | 62.00 | 36.76 | 209,746 | 12.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.134/shufflenet_g1_wd4-3676-cb39b773.h5.log)) |
| ShuffleNet x0.25 (g=3) | 61.32 | 36.15 | 305,902 | 13.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.135/shufflenet_g3_wd4-3615-21150468.h5.log)) |
| ShuffleNet x0.5 (g=3) | 43.82 | 20.60 | 718,324 | 41.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.167/shufflenet_g3_wd2-2060-173a725c.h5.log)) |
| ShuffleNetV2 x0.5 | 40.76 | 18.40 | 1,366,792 | 43.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.90/shufflenetv2_wd2-1840-9b4b0964.h5.log)) |
| ShuffleNetV2 x1.0 | 31.02 | 11.33 | 2,278,604 | 149.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.133/shufflenetv2_w1-1133-bcba973e.h5.log)) |
| ShuffleNetV2 x1.5 | 32.46 | 12.47 | 4,406,098 | 320.77M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.65/shufflenetv2_w3d2-1247-f7f813b4.h5.log)) |
| ShuffleNetV2 x2.0 | 31.91 | 12.23 | 7,601,686 | 595.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.84/shufflenetv2_w2-1223-63291468.h5.log)) |
| 108-MENet-8x1 (g=3) | 43.61 | 20.31 | 654,516 | 42.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.89/menet108_8x1_g3-2031-a4d43433.h5.log)) |
| 128-MENet-8x1 (g=4) | 42.08 | 19.14 | 750,796 | 45.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.103/menet128_8x1_g4-1914-5bb8f228.h5.log)) |
| 160-MENet-8x1 (g=8) | 43.47 | 20.28 | 850,120 | 45.63M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.154/menet160_8x1_g8-2028-09664de9.h5.log)) |
| 228-MENet-12x1 (g=3) | 33.85 | 12.88 | 1,806,568 | 152.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.131/menet228_12x1_g3-1288-c2eeac24.h5.log)) |
| 256-MENet-12x1 (g=4) | 32.22 | 12.17 | 1,888,240 | 150.65M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.152/menet256_12x1_g4-1217-b020cc33.h5.log)) |
| 348-MENet-12x1 (g=3) | 31.17 | 11.42 | 3,368,128 | 312.00M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.58/menet348_12x1_g3-1142-0715c866.h5.log)) |
| 352-MENet-12x1 (g=8) | 34.69 | 13.75 | 2,272,872 | 157.35M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.58/menet352_12x1_g8-1375-9007c933.h5.log)) |
| 456-MENet-24x1 (g=3) | 29.55 | 10.44 | 5,304,784 | 567.90M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.58/menet456_24x1_g3-1044-c090af59.h5.log)) |
| MobileNet x0.25 | 45.80 | 22.17 | 470,072 | 44.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.62/mobilenet_wd4-2217-fb7abda8.h5.log)) |
| MobileNet x0.5 | 33.94 | 13.30 | 1,331,592 | 155.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.156/mobilenet_wd2-1330-aa86f355.h5.log)) |
| MobileNet x0.75 | 29.85 | 10.51 | 2,585,560 | 333.99M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.130/mobilenet_w3d4-1051-d200ad45.h5.log)) |
| MobileNet x1.0 | 26.43 | 8.66 | 4,231,976 | 579.80M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.155/mobilenet_w1-0866-9661b555.h5.log)) |
| FD-MobileNet x0.25 | 56.17 | 31.37 | 383,160 | 12.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.68/fdmobilenet_wd4-3137-153934e4.h5.log)) |
| FD-MobileNet x0.5 | 42.61 | 19.69 | 993,928 | 41.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.83/fdmobilenet_wd2-1969-5678a212.h5.log)) |
| FD-MobileNet x0.75 | 37.90 | 16.01 | 1,833,304 | 86.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.159/fdmobilenet_w3d4-1601-2ea5eba9.h5.log)) |
| FD-MobileNet x1.0 | 33.80 | 13.12 | 2,901,288 | 147.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.162/fdmobilenet_w1-1312-e11d0dce.h5.log)) |
| MobileNetV2 x0.25 | 48.06 | 24.12 | 1,516,392 | 34.24M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.137/mobilenetv2_wd4-2412-62273372.h5.log)) |
| MobileNetV2 x0.5 | 35.63 | 14.43 | 1,964,736 | 100.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.170/mobilenetv2_wd2-1443-c7086bcc.h5.log)) |
| MobileNetV2 x0.75 | 30.81 | 11.26 | 2,627,592 | 198.50M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.61/mobilenetv2_w3d4-1126-f2f664da.h5.log)) |
| MobileNetV2 x1.0 | 28.50 | 9.90 | 3,504,960 | 329.36M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.61/mobilenetv2_w1-0990-cbb8be96.h5.log)) |
| IGCV3 x0.25 | 53.41 | 28.29 | 1,534,020 | 41.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.142/igcv3_wd4-2829-00072caf.h5.log)) |
| IGCV3 x0.5 | 39.39 | 17.04 | 1,985,528 | 111.12M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.132/igcv3_wd2-1704-b8961ca3.h5.log)) |
| IGCV3 x1.0 | 28.21 | 9.55 | 3,491,688 | 340.79M | From [homles11/IGCV3] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.126/igcv3_w1-0955-e2bde79d.h5.log)) |
| MnasNet | 31.30 | 11.45 | 4,308,816 | 317.67M | From [zeusees/Mnasnet...Model] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.117/mnasnet-1145-11b6acf1.h5.log)) |

[dmlc/gluon-cv]: https://github.com/dmlc/gluon-cv
[Cadene/pretrained...pytorch]: https://github.com/Cadene/pretrained-models.pytorch
[clavichord93/MENet]: https://github.com/clavichord93/MENet
[homles11/IGCV3]: https://github.com/homles11/IGCV3
[zeusees/Mnasnet...Model]: https://github.com/zeusees/Mnasnet-Pretrained-Model
