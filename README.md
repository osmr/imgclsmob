# Large-scale image classification networks for embedded systems

[![Build Status](https://travis-ci.org/osmr/imgclsmob.svg?branch=master)](https://travis-ci.org/osmr/imgclsmob)
[![GitHub License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-2.7%2C3.6-lightgrey.svg)](https://github.com/osmr/imgclsmob)

This repository contains several classification models on MXNet/Gluon, PyTorch, Chainer, Keras, and TensorFlow, with
scripts for training/validating/converting models. All models are designed for using with ImageNet-1k dataset.

## List of models

- AlexNet (['One weird trick for parallelizing convolutional neural networks'](https://arxiv.org/abs/1404.5997))
- VGG/BN-VGG (['Very Deep Convolutional Networks for Large-Scale Image Recognition'](https://arxiv.org/abs/1409.1556))
- ResNet (['Deep Residual Learning for Image Recognition'](https://arxiv.org/abs/1512.03385))
- PreResNet (['Identity Mappings in Deep Residual Networks'](https://arxiv.org/abs/1603.05027))
- ResNeXt (['Aggregated Residual Transformations for Deep Neural Networks'](http://arxiv.org/abs/1611.05431))
- SENet/SE-ResNet/SE-PreResNet/SE-ResNeXt (['Squeeze-and-Excitation Networks'](https://arxiv.org/abs/1709.01507))
- PyramidNet (['Deep Pyramidal Residual Networks'](https://arxiv.org/abs/1610.02915))
- DiracNetV2 (['DiracNets: Training Very Deep Neural Networks Without Skip-Connections'](https://arxiv.org/abs/1706.00388))
- DenseNet (['Densely Connected Convolutional Networks'](https://arxiv.org/abs/1608.06993))
- CondenseNet (['CondenseNet: An Efficient DenseNet using Learned Group Convolutions'](https://arxiv.org/abs/1711.09224))
- WRN (['Wide Residual Networks'](https://arxiv.org/abs/1605.07146))
- DRN-C/DRN-D (['Dilated Residual Networks'](https://arxiv.org/abs/1705.09914))
- DPN (['Dual Path Networks'](https://arxiv.org/abs/1707.01629))
- DarkNet (['Darknet: Open source neural networks in c'](https://github.com/pjreddie/darknet)) 
- SqueezeNet/SqueezeResNet (['SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size'](https://arxiv.org/abs/1602.07360))
- SqueezeNext (['SqueezeNext: Hardware-Aware Neural Network Design'](https://arxiv.org/abs/1803.10615))
- ShuffleNet (['ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices'](https://arxiv.org/abs/1707.01083))
- ShuffleNetV2 (['ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design'](https://arxiv.org/abs/1807.11164))
- MENet (['Merging and Evolution: Improving Convolutional Neural Networks for Mobile Applications'](https://arxiv.org/abs/1803.09127))
- MobileNet (['MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications'](https://arxiv.org/abs/1704.04861))
- FD-MobileNet (['FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy'](https://arxiv.org/abs/1802.03750))
- MobileNetV2 (['MobileNetV2: Inverted Residuals and Linear Bottlenecks'](https://arxiv.org/abs/1801.04381))
- MnasNet (['MnasNet: Platform-Aware Neural Architecture Search for Mobile'](https://arxiv.org/abs/1807.11626))
- Xception (['Xception: Deep Learning with Depthwise Separable Convolutions'](https://arxiv.org/abs/1610.02357))
- InceptionV3 (['Rethinking the Inception Architecture for Computer Vision'](https://arxiv.org/abs/1512.00567))
- InceptionV4/InceptionResNetV2 (['Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning'](https://arxiv.org/abs/1602.07261))
- PolyNet (['PolyNet: A Pursuit of Structural Diversity in Very Deep Networks'](https://arxiv.org/abs/1611.05725))
- NASNet (['Learning Transferable Architectures for Scalable Image Recognition'](https://arxiv.org/abs/1707.07012))
- PNASNet (['Progressive Neural Architecture Search'](https://arxiv.org/abs/1712.00559))

## Installation

### For Gluon way

To use only Gluon models in your project, simply install the `gluoncv2` package with `mxnet`:
```
pip install gluoncv2 mxnet>=1.2.1
```
To enable different hardware supports such as GPUs, check out [MXNet variants](https://pypi.org/project/mxnet/).
For example, you can install with CUDA-9.2 supported MXNet:
```
pip install gluoncv2 mxnet-cu92>=1.2.1
```

### For PyTorch way

To use only PyTorch models in your project, simply install the `pytorchcv` package with `torch` (>=0.4.1 is recommended):
```
pip install pytorchcv torch>=0.4.0
```
To enable/disable different hardware supports such as GPUs, check out PyTorch installation [instructions](https://pytorch.org/).

### For Chainer way

To use only Chainer models in your project, simply install the `chainercv2` package:
```
pip install chainercv2
```

### For Keras way

To use only Keras models in your project, simply install the `kerascv` package with `mxnet`:
```
pip install kerascv mxnet>=1.2.1
```
To enable different hardware supports such as GPUs, check out [MXNet variants](https://pypi.org/project/mxnet/).
For example, you can install with CUDA-9.2 supported MXNet:
```
pip install kerascv mxnet-cu92>=1.2.1
```
After installation change the value of the field `image_data_format` to `channels_first` in the file `~/.keras/keras.json`. 

### For TensorFlow way

To use only TensorFlow models in your project, simply install the `tensorflowcv` package with `tensorflow-gpu`:
```
pip install tensorflowcv tensorflow-gpu>=1.11.0
```
To enable/disable different hardware supports, check out TensorFlow installation [instructions](https://www.tensorflow.org/).

Note that the models use NCHW data format. The current version of TensorFlow cannot work with them on CPU.

### For research

To use the repository for training/validation/converting models:
```
git clone git@github.com:osmr/imgclsmob.git
pip install -r requirements.txt
```

## Usage

### For Gluon way

Example of using the pretrained ResNet-18 model on Gluon:
```
from gluoncv2.model_provider import get_model as glcv2_get_model
import mxnet as mx

net = glcv2_get_model("resnet18", pretrained=True)
x = mx.nd.zeros((1, 3, 224, 224), ctx=mx.cpu())
y = net(x)
```

### For PyTorch way

Example of using the pretrained ResNet-18 model on PyTorch:
```
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from torch.autograd import Variable

net = ptcv_get_model("resnet18", pretrained=True)
x = Variable(torch.randn(1, 3, 224, 224))
y = net(x)
```

### For Chainer way

Example of using the pretrained ResNet-18 model on Chainer:
```
from chainercv2.model_provider import get_model as chcv2_get_model
import numpy as np

net = chcv2_get_model("resnet18", pretrained=True)
x = np.zeros((1, 3, 224, 224), np.float32)
y = net(x)
```

### For Keras way

Example of using the pretrained ResNet-18 model on Keras:
```
from kerascv.model_provider import get_model as kecv_get_model
import numpy as np

net = kecv_get_model("resnet18", pretrained=True)
x = np.zeros((1, 3, 224, 224), np.float32)
y = net.predict(x)
```

### For TensorFlow way

Example of using the pretrained ResNet-18 model on TensorFlow:
```
from tensorflowcv.model_provider import get_model as tfcv_get_model
from tensorflowcv.model_provider import init_variables_from_state_dict as tfcv_init_variables_from_state_dict
import tensorflow as tf
import numpy as np

net = tfcv_get_model("resnet18", pretrained=True)
x = tf.placeholder(dtype=tf.float32, shape=(None, 3, 224, 224), name='xx')
y_net = net(x)

with tf.Session() as sess:
    tfcv_init_variables_from_state_dict(sess=sess, state_dict=net.state_dict)
    x_value = np.zeros((1, 3, 224, 224), np.float32)
    y = sess.run(y_net, feed_dict={x: x_value})
```

## Pretrained models

Some remarks:
- All pretrained models can be downloaded automatically during use (use the parameter `pretrained`).
- Top1/Top5 are the standard 1-crop Top-1/Top-5 errors (in percents) on the validation subset of the ImageNet1k dataset.
- ResNet/PreResNet with b-suffix is a version of the networks with the stride in the second convolution of the
bottleneck block. Respectively a network without b-suffix has the stride in the first convolution.
- ResNet/PreResNet models do not use biases in convolutions at all.
- CondenseNet models are only so-called converted versions.
- ShuffleNetV2/ShuffleNetV2b/ShuffleNetV2c are different implementations of the same architecture.
- All models require ordinary normalization.

### For Gluon

| Model | Top1 | Top5 | Params | FLOPs | Remarks |
| --- | ---: | ---: | ---: | ---: | --- |
| AlexNet | 44.12 | 21.26 | 61,100,840 | 715.49M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.108/alexnet-2126-9cb87ebd.params.log)) |
| VGG-11 | 31.91 | 11.76 | 132,863,336 | 7,622.65M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg11-1176-95dd287d.params.log)) |
| VGG-13 | 31.06 | 11.12 | 133,047,848 | 11,326.85M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg13-1112-a0db3c6c.params.log)) |
| VGG-16 | 26.78 | 8.69 | 138,357,544 | 15,489.95M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg16-0869-57a2556f.params.log)) |
| VGG-19 | 25.88 | 8.23 | 143,667,240 | 19,653.05M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg19-0823-0e2a1e0a.params.log)) |
| BN-VGG-11b | 30.34 | 10.57 | 132,868,840 | 7,622.65M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg11b-1057-b2d8f382.params.log)) |
| BN-VGG-13b | 29.48 | 10.16 | 133,053,736 | 11,326.85M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg13b-1016-f384ff52.params.log)) |
| BN-VGG-16b | 26.89 | 8.65 | 138,365,992 | 15,489.95M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg16b-0865-b5e33db8.params.log)) |
| BN-VGG-19b | 25.66 | 8.15 | 143,678,248 | 19,653.05M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg19b-0815-3a0e43e6.params.log)) |
| ResNet-10 | 37.09 | 15.55 | 5,418,792 | 892.62M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet10-1555-cfb0a76d.params.log)) |
| ResNet-12 | 35.86 | 14.46 | 5,492,776 | 1,124.23M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.30/resnet12-1446-9ce715b0.params.log)) |
| ResNet-14 | 32.85 | 12.41 | 5,788,200 | 1,355.64M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.40/resnet14-1241-a8955ff3.params.log)) |
| ResNet-16 | 30.68 | 11.10 | 6,968,872 | 1,586.95M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.41/resnet16-1110-1be996d1.params.log)) |
| ResNet-18 x0.25 | 49.16 | 24.45 | 831,096 | 136.64M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.47/resnet18_wd4-2445-28d15cf4.params.log)) |
| ResNet-18 x0.5 | 36.54 | 14.96 | 3,055,880 | 485.22M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.46/resnet18_wd2-1496-d839c509.params.log)) |
| ResNet-18 x0.75 | 33.25 | 12.54 | 6,675,352 | 1,045.75M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.18/resnet18_w3d4-1254-d6548612.params.log)) |
| ResNet-18 | 29.13 | 9.94 | 11,689,512 | 1,818.21M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet18-0994-ae25f2b2.params.log)) |
| ResNet-34 | 25.34 | 7.92 | 21,797,672 | 3,669.16M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet34-0792-5b875f49.params.log)) |
| ResNet-50 | 23.50 | 6.87 | 25,557,032 | 3,868.96M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet50-0687-79fae958.params.log)) |
| ResNet-50b | 22.92 | 6.44 | 25,557,032 | 4,100.70M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet50b-0644-27a36c02.params.log)) |
| ResNet-101 | 21.66 | 5.99 | 44,549,160 | 7,586.30M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet101-0599-a6d3a5f4.params.log)) |
| ResNet-101b | 21.18 | 5.60 | 44,549,160 | 7,818.04M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet101b-0560-6517274e.params.log)) |
| ResNet-152 | 21.01 | 5.61 | 60,192,808 | 11,304.85M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet152-0561-d05971c8.params.log)) |
| ResNet-152b | 20.54 | 5.37 | 60,192,808 | 11,536.58M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet152b-0537-4f5bd879.params.log)) |
| PreResNet-18 | 28.72 | 9.88 | 11,687,848 | 1,818.41M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.39/preresnet18-0988-5defff0e.params.log)) |
| PreResNet-34 | 25.88 | 8.11 | 21,796,008 | 3,669.36M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet34-0811-f8fe98a2.params.log)) |
| PreResNet-50 | 23.39 | 6.68 | 25,549,480 | 3,869.16M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet50-0668-4940c94b.params.log)) |
| PreResNet-50b | 23.16 | 6.64 | 25,549,480 | 4,100.90M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet50b-0664-2fcfddb1.params.log)) |
| PreResNet-101 | 21.45 | 5.75 | 44,541,608 | 7,586.50M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet101-0575-e2887e53.params.log)) |
| PreResNet-101b | 21.73 | 5.88 | 44,541,608 | 7,818.24M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet101b-0588-1015145a.params.log)) |
| PreResNet-152 | 20.70 | 5.32 | 60,185,256 | 11,305.05M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.14/preresnet152-0532-31505f71.params.log)) |
| PreResNet-152b | 21.00 | 5.75 | 60,185,256 | 11,536.78M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet152b-0575-dc303191.params.log)) |
| PreResNet-200b | 21.10 | 5.64 | 64,666,280 | 15,040.27M | From [tornadomeet/ResNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.45/preresnet200b-0564-38f849a6.params.log)) |
| ResNeXt-101 (32x4d) | 21.32 | 5.79 | 44,177,704 | 7,991.62M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.10/resnext101_32x4d-0579-9afbfdbc.params.log)) |
| ResNeXt-101 (64x4d) | 20.60 | 5.41 | 83,455,272 | 15,491.88M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.10/resnext101_64x4d-0541-0d4fd87b.params.log)) |
| SE-ResNet-50 | 22.51 | 6.44 | 28,088,024 | 3,877.01M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.11/seresnet50-0644-10954a84.params.log)) |
| SE-ResNet-101 | 21.92 | 5.89 | 49,326,872 | 7,600.01M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.11/seresnet101-0589-4c10238d.params.log)) |
| SE-ResNet-152 | 21.48 | 5.77 | 66,821,848 | 11,324.62M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.11/seresnet152-0577-de6f099d.params.log)) |
| SE-ResNeXt-50 (32x4d) | 21.06 | 5.58 | 27,559,896 | 4,253.33M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.12/seresnext50_32x4d-0558-a49f8fb0.params.log)) |
| SE-ResNeXt-101 (32x4d) | 19.99 | 5.00 | 48,955,416 | 8,005.33M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.12/seresnext101_32x4d-0500-cf161260.params.log)) |
| SENet-154 | 18.84 | 4.65 | 115,088,984 | 20,742.40M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.13/senet154-0465-dd244507.params.log)) |
| PyramidNet-101 (a=360) | 22.72 | 6.52 | 42,455,070 | 8,706.81M | From [dyhan0920/Pyramid...PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.104/pyramidnet101_a360-0652-08d5a5d1.params.log)) |
| DiracNetV2-18 | 30.61 | 11.17 | 11,511,784 | 1,798.43M | From [[szagoruyko/diracnets]] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.111/diracnet18v2-1117-27601f6f.params.log)) |
| DiracNetV2-34 | 27.93 | 9.46 | 21,616,232 | 3,649.37M | From [[szagoruyko/diracnets]] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.111/diracnet34v2-0946-1faa6f12.params.log)) |
| DenseNet-121 | 25.11 | 7.80 | 7,978,856 | 2,852.39M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.3/densenet121-0780-49b72d04.params.log)) |
| DenseNet-161 | 22.40 | 6.18 | 28,681,000 | 7,761.25M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.3/densenet161-0618-52e30516.params.log)) |
| DenseNet-169 | 23.89 | 6.89 | 14,149,480 | 3,381.48M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.3/densenet169-0689-281ec06b.params.log)) |
| DenseNet-201 | 22.71 | 6.36 | 20,013,928 | 4,318.75M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.3/densenet201-0636-65b5d389.params.log)) |
| CondenseNet-74 (C=G=4) | 26.82 | 8.64 | 4,773,944 | 533.64M | From [ShichenLiu/CondenseNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.4/condensenet74_c4_g4-0864-cde68fa2.params.log)) |
| CondenseNet-74 (C=G=8) | 29.76 | 10.49 | 2,935,416 | 278.55M | From [ShichenLiu/CondenseNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.4/condensenet74_c8_g8-1049-4cf4a08e.params.log)) |
| WRN-50-2 | 22.15 | 6.12 | 68,849,128 | 11,412.82M | From [szagoruyko/functional-zoo] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.113/wrn50_2-0612-f8013e68.params.log)) |
| DPN-C-26 | 25.68 | 7.89 | 21,126,584 | 20,838.70M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnc26-0789-ee56ffab.params.log)) |
| DPN-C-42 | 23.80 | 6.92 | 31,234,744 | 31,236.97M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnc42-0692-f89c26d6.params.log)) |
| DPN-C-58 | 22.35 | 6.27 | 40,542,008 | 36,862.32M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnc58-0627-44cbf15c.params.log)) |
| DPN-D-22 | 26.67 | 8.52 | 16,393,752 | 16,626.00M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd22-0852-08574752.params.log)) |
| DPN-D-38 | 24.51 | 7.36 | 26,501,912 | 27,024.27M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd38-0736-c7d53bc0.params.log)) |
| DPN-D-54 | 22.05 | 6.27 | 35,809,176 | 32,649.62M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd54-0627-87d44c87.params.log)) |
| DPN-D-105 | 21.31 | 5.81 | 54,801,304 | 48,682.11M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd105-0581-ab12d662.params.log)) |
| DPN-68 | 23.57 | 7.00 | 12,611,602 | 2,338.71M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.17/dpn68-0700-3114719d.params.log)) |
| DPN-98 | 20.23 | 5.28 | 61,570,728 | 11,702.80M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.17/dpn98-0528-fa5d6fca.params.log)) |
| DPN-131 | 20.03 | 5.22 | 79,254,504 | 16,056.22M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.17/dpn131-0522-35ac2f82.params.log)) |
| DarkNet Tiny | 40.31 | 17.46 | 1,042,104 | 496.34M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.69/darknet_tiny-1746-16501793.params.log)) |
| DarkNet Ref | 38.00 | 16.68 | 7,319,416 | 365.55M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.64/darknet_ref-1668-3011b4e1.params.log)) |
| SqueezeNet v1.0 | 40.97 | 18.96 | 1,248,424 | 828.30M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.19/squeezenet_v1_0-1896-b69a4607.params.log)) |
| SqueezeNet v1.1 | 39.09 | 17.39 | 1,235,496 | 354.88M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.88/squeezenet_v1_1-1739-d7a1483a.params.log)) |
| SqueezeResNet v1.1 | 39.83 | 17.84 | 1,235,496 | 354.88M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.70/squeezeresnet_v1_1-1784-26064b82.params.log)) |
| ShuffleNetV2 x0.5 | 40.61 | 18.30 | 1,366,792 | 42.34M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.90/shufflenetv2_wd2-1830-156953de.params.log)) |
| ShuffleNetV2b x0.5 | 40.98 | 18.56 | 1,366,792 | 42.37M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.112/shufflenetv2b_wd2-1856-d1143ea2.params.log)) |
| ShuffleNetV2c x0.5 | 39.87 | 18.11 | 1,366,792 | 42.37M | From [tensorpack/tensorpack] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.91/shufflenetv2c_wd2-1811-979ce7d9.params.log)) |
| ShuffleNetV2 x1.0 | 33.76 | 13.22 | 2,278,604 | 147.92M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.93/shufflenetv2_w1-1322-04b52239.params.log)) |
| ShuffleNetV2c x1.0 | 30.74 | 11.38 | 2,279,760 | 148.85M | From [tensorpack/tensorpack] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.95/shufflenetv2c_w1-1138-646f3b78.params.log)) |
| ShuffleNetV2 x1.5 | 32.38 | 12.37 | 4,406,098 | 318.61M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.65/shufflenetv2_w3d2-1237-08c01388.params.log)) |
| ShuffleNetV2 x2.0 | 32.04 | 12.10 | 7,601,686 | 593.66M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.84/shufflenetv2_w2-1210-544b55d9.params.log)) |
| 108-MENet-8x1 (g=3) | 43.62 | 20.30 | 654,516 | 40.64M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.89/menet108_8x1_g3-2030-aa07f925.params.log)) |
| 128-MENet-8x1 (g=4) | 42.10 | 19.13 | 750,796 | 43.58M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.103/menet128_8x1_g4-1913-0c890a76.params.log)) |
| 228-MENet-12x1 (g=3) | 35.03 | 13.99 | 1,806,568 | 148.93M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet228_12x1_g3-1399-8c28d22f.params.log)) |
| 256-MENet-12x1 (g=4) | 34.49 | 13.90 | 1,888,240 | 146.11M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet256_12x1_g4-1390-4502f223.params.log)) |
| 348-MENet-12x1 (g=3) | 31.17 | 11.41 | 3,368,128 | 306.31M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet348_12x1_g3-1141-ac69b246.params.log)) |
| 352-MENet-12x1 (g=8) | 34.70 | 13.75 | 2,272,872 | 151.03M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet352_12x1_g8-1375-85779b8a.params.log)) |
| 456-MENet-24x1 (g=3) | 29.57 | 10.43 | 5,304,784 | 560.72M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet456_24x1_g3-1043-6e777068.params.log)) |
| MobileNet x0.25 | 45.78 | 22.18 | 470,072 | 42.30M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.62/mobilenet_wd4-2218-3185cdd2.params.log)) |
| MobileNet x0.5 | 36.12 | 14.81 | 1,331,592 | 152.04M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.66/mobilenet_wd2-1481-9f48baf6.params.log)) |
| MobileNet x0.75 | 32.71 | 12.28 | 2,585,560 | 329.22M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.7/mobilenet_w3d4-1228-dc11727a.params.log)) |
| MobileNet x1.0 | 29.25 | 10.03 | 4,231,976 | 573.83M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.7/mobilenet_w1-1003-b4fb8f1b.params.log)) |
| FD-MobileNet x0.25 | 56.19 | 31.38 | 383,160 | 12.44M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.68/fdmobilenet_wd4-3138-2fe432fd.params.log)) |
| FD-MobileNet x0.5 | 42.62 | 19.69 | 993,928 | 40.93M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.83/fdmobilenet_wd2-1969-242b9fa8.params.log)) |
| FD-MobileNet x1.0 | 35.95 | 14.72 | 2,901,288 | 146.08M | From [clavichord93/FD-MobileNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.8/fdmobilenet_w1-1472-a525b206.params.log)) |
| MobileNetV2 x0.25 | 48.89 | 25.24 | 1,516,392 | 32.22M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.9/mobilenetv2_wd4-2524-a2468611.params.log)) |
| MobileNetV2 x0.5 | 35.51 | 14.64 | 1,964,736 | 95.62M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.9/mobilenetv2_wd2-1464-02fe7ff2.params.log)) |
| MobileNetV2 x0.75 | 30.82 | 11.26 | 2,627,592 | 191.61M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.9/mobilenetv2_w3d4-1126-152672f5.params.log)) |
| MobileNetV2 x1.0 | 28.51 | 9.90 | 3,504,960 | 320.19M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.9/mobilenetv2_w1-0990-4e1a3878.params.log)) |
| MnasNet | 32.34 | 12.02 | 4,308,816 | 310.75M | From [zeusees/Mnasnet...Model] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.106/mnasnet-1202-993d5546.params.log)) |
| Xception | 20.99 | 5.56 | 22,855,952 | 8,385.86M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.115/xception-0556-bd2c1684.params.log)) |
| InceptionV3 | 21.22 | 5.59 | 23,834,568 | 5,746.72M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.92/inceptionv3-0559-6c087967.params.log)) |
| InceptionV4 | 20.60 | 5.25 | 42,679,816 | 12,314.17M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.105/inceptionv4-0525-f7aa9536.params.log)) |
| InceptionResNetV2 | 19.96 | 4.94 | 55,843,464 | 13,189.58M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.107/inceptionresnetv2-0494-3328f7fa.params.log)) |
| PolyNet | 19.09 | 4.53 | 95,366,600 | 34,768.84M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.96/polynet-0453-74280314.params.log)) |
| NASNet-A 4@1056 | 25.37 | 7.95 | 5,289,978 | 587.29M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.97/nasnet_4a1056-0795-5c78908e.params.log)) |
| NASNet-A 6@4032 | 18.17 | 4.24 | 88,753,150 | 24,021.18M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.101/nasnet_6a4032-0424-73cca5fe.params.log)) |
| PNASNet-5-Large | 17.90 | 4.28 | 86,057,668 | 25,169.47M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.114/pnasnet5large-0428-998a548f.params.log)) |

### For PyTorch

| Model | Top1 | Top5 | Params | FLOPs | Remarks |
| --- | ---: | ---: | ---: | ---: | --- |
| AlexNet | 43.48 | 20.93 | 61,100,840 | 715.49M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.108/alexnet-2093-6429d865.pth.log)) |
| VGG-11 | 30.98 | 11.37 | 132,863,336 | 7,622.65M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg11-1137-8a64fe7a.pth.log)) |
| VGG-13 | 30.07 | 10.75 | 133,047,848 | 11,326.85M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg13-1075-24178cab.pth.log)) |
| VGG-16 | 27.15 | 8.92 | 138,357,544 | 15,489.95M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg16-0892-10f44f68.pth.log)) |
| VGG-19 | 26.19 | 8.39 | 143,667,240 | 19,653.05M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg19-0839-d4e69a0d.pth.log)) |
| BN-VGG-11b | 29.63 | 10.19 | 132,868,840 | 7,622.65M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg11b-1019-98d7e914.pth.log)) |
| BN-VGG-13b | 28.41 | 9.63 | 133,053,736 | 11,326.85M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg13b-0963-cf9352f4.pth.log)) |
| BN-VGG-16b | 27.19 | 8.74 | 138,365,992 | 15,489.95M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg16b-0874-af4f2d0b.pth.log)) |
| BN-VGG-19b | 26.06 | 8.40 | 143,678,248 | 19,653.05M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg19b-0840-b6919f7f.pth.log)) |
| ResNet-10 | 37.46 | 15.85 | 5,418,792 | 892.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet10-1585-ef8a3ae3.pth.log)) |
| ResNet-12 | 36.18 | 14.80 | 5,492,776 | 1,124.23M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.30/resnet12-1480-c2263f73.pth.log)) |
| ResNet-14 | 33.17 | 12.71 | 5,788,200 | 1,355.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.40/resnet14-1271-568c392e.pth.log)) |
| ResNet-16 | 30.90 | 11.38 | 6,968,872 | 1,586.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.41/resnet16-1138-3a5aa7c0.pth.log)) |
| ResNet-18 x0.25 | 49.50 | 24.83 | 831,096 | 136.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.47/resnet18_wd4-2483-6ef2515c.pth.log)) |
| ResNet-18 x0.5 | 37.04 | 15.38 | 3,055,880 | 485.22M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.46/resnet18_wd2-1538-671466b5.pth.log)) |
| ResNet-18 x0.75 | 33.61 | 12.85 | 6,675,352 | 1,045.75M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.18/resnet18_w3d4-1285-94713e0e.pth.log)) |
| ResNet-18 | 29.52 | 10.21 | 11,689,512 | 1,818.21M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet18-1021-b0d7daea.pth.log)) |
| ResNet-34 | 25.66 | 8.18 | 21,797,672 | 3,669.16M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet34-0818-6f947d40.pth.log)) |
| ResNet-50 | 23.79 | 7.05 | 25,557,032 | 3,868.96M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet50-0705-f7a2027e.pth.log)) |
| ResNet-50b | 23.05 | 6.65 | 25,557,032 | 4,100.70M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet50b-0665-89691746.pth.log)) |
| ResNet-101 | 21.90 | 6.22 | 44,549,160 | 7,586.30M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet101-0622-ab0cf005.pth.log)) |
| ResNet-101b | 21.45 | 5.81 | 44,549,160 | 7,818.04M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet101b-0581-d983e682.pth.log)) |
| ResNet-152 | 21.26 | 5.82 | 60,192,808 | 11,304.85M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet152-0582-af1a3bd5.pth.log)) |
| ResNet-152b | 20.74 | 5.50 | 60,192,808 | 11,536.58M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet152b-0550-216604cf.pth.log)) |
| PreResNet-18 | 29.09 | 10.18 | 11,687,848 | 1,818.41M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.39/preresnet18-1018-98958fd2.pth.log)) |
| PreResNet-34 | 26.23 | 8.41 | 21,796,008 | 3,669.36M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet34-0841-b4dd761f.pth.log)) |
| PreResNet-50 | 23.70 | 6.85 | 25,549,480 | 3,869.16M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet50-0685-d81a7aca.pth.log)) |
| PreResNet-50b | 23.33 | 6.87 | 25,549,480 | 4,100.90M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet50b-0687-65be98fb.pth.log)) |
| PreResNet-101 | 21.74 | 5.91 | 44,541,608 | 7,586.50M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet101-0591-4bacff79.pth.log)) |
| PreResNet-101b | 21.95 | 6.03 | 44,541,608 | 7,818.24M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet101b-0603-b1e37a09.pth.log)) |
| PreResNet-152 | 20.94 | 5.55 | 60,185,256 | 11,305.05M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.14/preresnet152-0555-c842a030.pth.log)) |
| PreResNet-152b | 21.34 | 5.91 | 60,185,256 | 11,536.78M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet152b-0591-2c91ab2c.pth.log)) |
| PreResNet-200b | 21.33 | 5.88 | 64,666,280 | 15,040.27M | From [tornadomeet/ResNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.45/preresnet200b-0588-f7104ff3.pth.log)) |
| ResNeXt-101 (32x4d) | 21.81 | 6.11 | 44,177,704 | 7,991.62M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.10/resnext101_32x4d-0611-cf962440.pth.log)) |
| ResNeXt-101 (64x4d) | 21.04 | 5.75 | 83,455,272 | 15,491.88M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.10/resnext101_64x4d-0575-651abd02.pth.log)) |
| SE-ResNet-50 | 22.47 | 6.40 | 28,088,024 | 3,877.01M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.11/seresnet50-0640-8820f2af.pth.log)) |
| SE-ResNet-101 | 21.88 | 5.89 | 49,326,872 | 7,600.01M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.11/seresnet101-0589-5e6e831b.pth.log)) |
| SE-ResNet-152 | 21.48 | 5.76 | 66,821,848 | 11,324.62M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.11/seresnet152-0576-814cf72e.pth.log)) |
| SE-ResNeXt-50 (32x4d) | 21.00 | 5.54 | 27,559,896 | 4,253.33M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.12/seresnext50_32x4d-0554-99e0e9aa.pth.log)) |
| SE-ResNeXt-101 (32x4d) | 19.96 | 5.05 | 48,955,416 | 8,005.33M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.12/seresnext101_32x4d-0505-0924f0a2.pth.log)) |
| SENet-154 | 18.62 | 4.61 | 115,088,984 | 20,742.40M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.13/senet154-0461-6512228c.pth.log)) |
| PyramidNet-101 (a=360) | 21.98 | 6.20 | 42,455,070 | 8,706.81M | From [dyhan0920/Pyramid...PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.104/pyramidnet101_a360-0620-3a24427b.pth.log)) |
| DiracNetV2-18 | 31.47 | 11.70 | 11,511,784 | 1,798.43M | From [[szagoruyko/diracnets]] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.111/diracnet18v2-1170-e0673770.pth.log)) |
| DiracNetV2-34 | 28.75 | 9.93 | 21,616,232 | 3,649.37M | From [[szagoruyko/diracnets]] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.111/diracnet34v2-0993-a6a661c0.pth.log)) |
| DenseNet-121 | 25.57 | 8.03 | 7,978,856 | 2,852.39M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.3/densenet121-0803-f994107a.pth.log)) |
| DenseNet-161 | 22.86 | 6.44 | 28,681,000 | 7,761.25M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.3/densenet161-0644-c0fb22c8.pth.log)) |
| DenseNet-169 | 24.40 | 7.19 | 14,149,480 | 3,381.48M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.3/densenet169-0719-27139105.pth.log)) |
| DenseNet-201 | 23.10 | 6.63 | 20,013,928 | 4,318.75M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.3/densenet201-0663-71ece4ad.pth.log)) |
| CondenseNet-74 (C=G=4) | 26.25 | 8.28 | 4,773,944 | 533.64M | From [ShichenLiu/CondenseNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.4/condensenet74_c4_g4-0828-5ba55049.pth.log)) |
| CondenseNet-74 (C=G=8) | 28.93 | 10.06 | 2,935,416 | 278.55M | From [ShichenLiu/CondenseNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.4/condensenet74_c8_g8-1006-3574d874.pth.log)) |
| WRN-50-2 | 22.53 | 6.41 | 68,849,128 | 11,412.82M | From [szagoruyko/functional-zoo] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.113/wrn50_2-0641-83897ab9.pth.log)) |
| DPN-C-26 | 24.86 | 7.55 | 21,126,584 | 20,838.70M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnc26-0755-35405bd5.pth.log)) |
| DPN-C-42 | 22.94 | 6.57 | 31,234,744 | 31,236.97M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnc42-0657-7c99c460.pth.log)) |
| DPN-C-58 | 21.73 | 6.01 | 40,542,008 | 36,862.32M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnc58-0601-70ec1f56.pth.log)) |
| DPN-D-22 | 25.80 | 8.23 | 16,393,752 | 16,626.00M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd22-0823-5c2c6a0c.pth.log)) |
| DPN-D-38 | 23.79 | 6.95 | 26,501,912 | 27,024.27M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd38-0695-4630f0fb.pth.log)) |
| DPN-D-54 | 21.22 | 5.86 | 35,809,176 | 32,649.62M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd54-0586-bfdc1f88.pth.log)) |
| DPN-D-105 | 20.62 | 5.48 | 54,801,304 | 48,682.11M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd105-0548-a643f4dc.pth.log)) |
| DPN-68 | 24.17 | 7.27 | 12,611,602 | 2,338.71M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.17/dpn68-0727-43849233.pth.log)) |
| DPN-98 | 20.81 | 5.53 | 61,570,728 | 11,702.80M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.17/dpn98-0553-52c55969.pth.log)) |
| DPN-131 | 20.54 | 5.48 | 79,254,504 | 16,056.22M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.17/dpn131-0548-0c53e5b3.pth.log)) |
| DarkNet Tiny | 40.74 | 17.84 | 1,042,104 | 496.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.69/darknet_tiny-1784-4561e1ad.pth.log)) |
| DarkNet Ref | 38.58 | 17.18 | 7,319,416 | 365.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.64/darknet_ref-1718-034595b4.pth.log)) |
| SqueezeNet v1.0 | 41.31 | 19.32 | 1,248,424 | 828.30M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.19/squeezenet_v1_0-1932-e4017303.pth.log)) |
| SqueezeNet v1.1 | 39.31 | 17.72 | 1,235,496 | 354.88M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.88/squeezenet_v1_1-1772-25b77bc3.pth.log)) |
| SqueezeResNet v1.1 | 40.09 | 18.21 | 1,235,496 | 354.88M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.70/squeezeresnet_v1_1-1821-c27ed88f.pth.log)) |
| ShuffleNetV2 x0.5 | 40.99 | 18.65 | 1,366,792 | 42.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.90/shufflenetv2_wd2-1865-9c22238b.pth.log)) |
| ShuffleNetV2b x0.5 | 41.41 | 19.07 | 1,366,792 | 42.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.112/shufflenetv2b_wd2-1907-cf4fe43c.pth.log)) |
| ShuffleNetV2c x0.5 | 40.31 | 18.51 | 1,366,792 | 42.37M | From [tensorpack/tensorpack] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.94/shufflenetv2c_wd2-1851-e1d36c5d.pth.log)) |
| ShuffleNetV2 x1.0 | 34.26 | 13.54 | 2,278,604 | 147.92M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.93/shufflenetv2_w1-1354-73e7c9fd.pth.log)) |
| ShuffleNetV2c x1.0 | 30.98 | 11.61 | 2,279,760 | 148.85M | From [tensorpack/tensorpack] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.95/shufflenetv2c_w1-1161-8cdbbcc1.pth.log)) |
| ShuffleNetV2 x1.5 | 32.82 | 12.69 | 4,406,098 | 318.61M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.65/shufflenetv2_w3d2-1269-536ad5b1.pth.log)) |
| ShuffleNetV2 x2.0 | 32.45 | 12.49 | 7,601,686 | 593.66M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.84/shufflenetv2_w2-1249-b9f9e84c.pth.log)) |
| 108-MENet-8x1 (g=3) | 43.94 | 20.76 | 654,516 | 40.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.89/menet108_8x1_g3-2076-6acc82e4.pth.log)) |
| 128-MENet-8x1 (g=4) | 42.43 | 19.59 | 750,796 | 43.58M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.103/menet128_8x1_g4-1959-48fa80fc.pth.log)) |
| 228-MENet-12x1 (g=3) | 33.57 | 13.28 | 1,806,568 | 148.93M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet228_12x1_g3-1328-27991387.pth.log)) |
| 256-MENet-12x1 (g=4) | 33.41 | 13.26 | 1,888,240 | 146.11M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet256_12x1_g4-1326-e5d35476.pth.log)) |
| 348-MENet-12x1 (g=3) | 30.10 | 10.92 | 3,368,128 | 306.31M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet348_12x1_g3-1092-66be1a18.pth.log)) |
| 352-MENet-12x1 (g=8) | 33.31 | 13.08 | 2,272,872 | 151.03M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet352_12x1_g8-1308-e91ec72c.pth.log)) |
| 456-MENet-24x1 (g=3) | 28.40 | 9.93 | 5,304,784 | 560.72M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet456_24x1_g3-0993-cb9fd376.pth.log)) |
| MobileNet x0.25 | 46.26 | 22.49 | 470,072 | 42.30M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.62/mobilenet_wd4-2249-1ad5e8fe.pth.log)) |
| MobileNet x0.5 | 36.30 | 15.14 | 1,331,592 | 152.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.66/mobilenet_wd2-1514-cc15154a.pth.log)) |
| MobileNet x0.75 | 33.54 | 12.85 | 2,585,560 | 329.22M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.7/mobilenet_w3d4-1285-b8022fae.pth.log)) |
| MobileNet x1.0 | 29.86 | 10.36 | 4,231,976 | 573.83M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.7/mobilenet_w1-1036-34f7a0cb.pth.log)) |
| FD-MobileNet x0.25 | 55.77 | 31.32 | 383,160 | 12.44M | From [clavichord93/FD-MobileNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.8/fdmobilenet_wd4-3132-0b242eff.pth.log)) |
| FD-MobileNet x0.5 | 43.13 | 20.15 | 993,928 | 40.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.83/fdmobilenet_wd2-2015-414dbeed.pth.log)) |
| FD-MobileNet x1.0 | 34.70 | 14.05 | 2,901,288 | 146.08M | From [clavichord93/FD-MobileNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.8/fdmobilenet_w1-1405-a6538879.pth.log)) |
| MobileNetV2 x0.25 | 49.72 | 25.87 | 1,516,392 | 32.22M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.9/mobilenetv2_wd4-2587-189d4ea2.pth.log)) |
| MobileNetV2 x0.5 | 36.54 | 15.19 | 1,964,736 | 95.62M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.9/mobilenetv2_wd2-1519-d0937a23.pth.log)) |
| MobileNetV2 x0.75 | 31.89 | 11.76 | 2,627,592 | 191.61M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.9/mobilenetv2_w3d4-1176-1b966ff4.pth.log)) |
| MobileNetV2 x1.0 | 29.31 | 10.39 | 3,504,960 | 320.19M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.9/mobilenetv2_w1-1039-7532eb72.pth.log)) |
| MnasNet | 32.62 | 12.35 | 4,308,816 | 310.75M | From [zeusees/Mnasnet...Model] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.106/mnasnet-1235-58b6de49.pth.log)) |
| Xception | 20.97 | 5.49 | 22,855,952 | 8,385.86M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.115/xception-0549-e4f0232c.pth.log)) |
| InceptionV3 | 21.12 | 5.65 | 23,834,568 | 5,746.72M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.92/inceptionv3-0565-cf406180.pth.log)) |
| InceptionV4 | 20.64 | 5.29 | 42,679,816 | 12,314.17M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.105/inceptionv4-0529-5cb7b4e4.pth.log)) |
| InceptionResNetV2 | 19.93 | 4.90 | 55,843,464 | 13,189.58M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.107/inceptionresnetv2-0490-1d1b4d18.pth.log)) |
| PolyNet | 19.10 | 4.52 | 95,366,600 | 34,768.84M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.96/polynet-0452-6a1b295d.pth.log)) |
| NASNet-A 4@1056 | 25.68 | 8.16 | 5,289,978 | 587.29M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.97/nasnet_4a1056-0816-d21bbaf5.pth.log)) |
| NASNet-A 6@4032 | 18.14 | 4.21 | 88,753,150 | 24,021.18M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.101/nasnet_6a4032-0421-f354d28f.pth.log)) |
| PNASNet-5-Large | 17.88 | 4.28 | 86,057,668 | 25,169.47M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.114/pnasnet5large-0428-65de46eb.pth.log)) |

### For Chainer

| Model | Top1 | Top5 | Params | FLOPs | Remarks |
| --- | ---: | ---: | ---: | ---: | --- |
| AlexNet | 44.08 | 21.32 | 61,100,840 | 715.49M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.108/alexnet-2132-cea565f1.npz.log)) |
| VGG-11 | 31.89 | 11.79 | 132,863,336 | 7,622.65M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg11-1179-3cc057e6.npz.log)) |
| VGG-13 | 31.06 | 11.16 | 133,047,848 | 11,326.85M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg13-1116-e835ca5a.npz.log)) |
| VGG-16 | 26.75 | 8.70 | 138,357,544 | 15,489.95M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg16-0870-8741ff5c.npz.log)) |
| VGG-19 | 25.86 | 8.23 | 143,667,240 | 19,653.05M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg19-0823-18980884.npz.log)) |
| BN-VGG-11b | 30.37 | 10.60 | 132,868,840 | 7,622.65M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg11b-1060-8964402b.npz.log)) |
| BN-VGG-13b | 29.45 | 10.19 | 133,053,736 | 11,326.85M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg13b-1019-0121b0a4.npz.log)) |
| BN-VGG-16b | 26.89 | 8.63 | 138,365,992 | 15,489.95M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg16b-0863-cbaa2105.npz.log)) |
| BN-VGG-19b | 25.65 | 8.16 | 143,678,248 | 19,653.05M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg19b-0816-dc5e37a5.npz.log)) |
| ResNet-10 | 37.12 | 15.49 | 5,418,792 | 892.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet10-1549-b31f1135.npz.log)) |
| ResNet-12 | 35.86 | 14.48 | 5,492,776 | 1,124.23M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.30/resnet12-1448-11acb729.npz.log)) |
| ResNet-14 | 32.84 | 12.42 | 5,788,200 | 1,355.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.40/resnet14-1242-4e65746b.npz.log)) |
| ResNet-16 | 30.66 | 11.07 | 6,968,872 | 1,586.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.41/resnet16-1107-b1d7fb7d.npz.log)) |
| ResNet-18 x0.25 | 49.08 | 24.48 | 831,096 | 136.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.47/resnet18_wd4-2448-58c4a007.npz.log)) |
| ResNet-18 x0.5 | 36.55 | 14.99 | 3,055,880 | 485.22M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.46/resnet18_wd2-1499-542ed773.npz.log)) |
| ResNet-18 x0.75 | 33.27 | 12.56 | 6,675,352 | 1,045.75M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet18_w3d4-1256-ce2011df.npz.log)) |
| ResNet-18 | 29.08 | 9.97 | 11,689,512 | 1,818.21M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet18-0997-9862a84f.npz.log)) |
| ResNet-34 | 25.35 | 7.95 | 21,797,672 | 3,669.16M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet34-0795-0b392267.npz.log)) |
| ResNet-50 | 23.50 | 6.83 | 25,557,032 | 3,868.96M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet50-0683-9c795737.npz.log)) |
| ResNet-50b | 22.93 | 6.46 | 25,557,032 | 4,100.70M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet50b-0646-225a550e.npz.log)) |
| ResNet-101 | 21.65 | 6.01 | 44,549,160 | 7,586.30M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet101-0601-d8cddbea.npz.log)) |
| ResNet-101b | 21.16 | 5.59 | 44,549,160 | 7,818.04M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet101b-0559-b5c3b4b6.npz.log)) |
| ResNet-152 | 21.07 | 5.67 | 60,192,808 | 11,304.85M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet152-0567-62d194fc.npz.log)) |
| ResNet-152b | 20.44 | 5.39 | 60,192,808 | 11,536.58M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet152b-0539-2b175728.npz.log)) |
| PreResNet-18 | 28.66 | 9.92 | 11,687,848 | 1,818.41M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.39/preresnet18-0992-ad0c7511.npz.log)) |
| PreResNet-34 | 25.89 | 8.12 | 21,796,008 | 3,669.36M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet34-0812-829f5a23.npz.log)) |
| PreResNet-50 | 23.36 | 6.69 | 25,549,480 | 3,869.16M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet50-0669-40bd5e93.npz.log)) |
| PreResNet-50b | 23.08 | 6.67 | 25,549,480 | 4,100.90M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet50b-0667-b7d221ef.npz.log)) |
| PreResNet-101 | 21.45 | 5.75 | 44,541,608 | 7,586.50M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet101-0575-f6f6789a.npz.log)) |
| PreResNet-101b | 21.61 | 5.87 | 44,541,608 | 7,818.24M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet101b-0587-4211c5ab.npz.log)) |
| PreResNet-152 | 20.73 | 5.30 | 60,185,256 | 11,305.05M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet152-0530-021d99dc.npz.log)) |
| PreResNet-152b | 20.88 | 5.66 | 60,185,256 | 11,536.78M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet152b-0566-fdd337e7.npz.log)) |
| PreResNet-200b | 21.03 | 5.60 | 64,666,280 | 15,040.27M | From [tornadomeet/ResNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.45/preresnet200b-0560-f79bd952.npz.log)) |
| ResNeXt-101 (32x4d) | 21.11 | 5.69 | 44,177,704 | 7,991.62M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.26/resnext101_32x4d-0569-c6d1c30d.npz.log)) |
| ResNeXt-101 (64x4d) | 20.57 | 5.43 | 83,455,272 | 15,491.88M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.26/resnext101_64x4d-0543-dd8b7d96.npz.log)) |
| SE-ResNet-50 | 22.53 | 6.41 | 28,088,024 | 3,877.01M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.24/seresnet50-0641-f3d68cfc.npz.log)) |
| SE-ResNet-101 | 21.90 | 5.88 | 49,326,872 | 7,600.01M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.24/seresnet101-0588-e45a9f8f.npz.log)) |
| SE-ResNet-152 | 21.46 | 5.77 | 66,821,848 | 11,324.62M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.24/seresnet152-0577-a089ba52.npz.log)) |
| SE-ResNeXt-50 (32x4d) | 21.04 | 5.58 | 27,559,896 | 4,253.33M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.27/seresnext50_32x4d-0558-5c435c1b.npz.log)) |
| SE-ResNeXt-101 (32x4d) | 19.99 | 5.01 | 48,955,416 | 8,005.33M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.27/seresnext101_32x4d-0501-98ea6fc4.npz.log)) |
| SENet-154 | 18.79 | 4.63 | 115,088,984 | 20,742.40M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.28/senet154-0463-381d2494.npz.log)) |
| PyramidNet-101 (a=360) | 22.66 | 6.49 | 42,455,070 | 8,706.81M | From [dyhan0920/Pyramid...PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.104/pyramidnet101_a360-0649-b68c786b.npz.log)) |
| DiracNetV2-18 | 30.60 | 11.13 | 11,511,784 | 1,798.43M | From [[szagoruyko/diracnets]] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.111/diracnet18v2-1113-b85b43d1.npz.log)) |
| DiracNetV2-34 | 27.90 | 9.48 | 21,616,232 | 3,649.37M | From [[szagoruyko/diracnets]] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.111/diracnet34v2-0948-0245163a.npz.log)) |
| DenseNet-121 | 25.04 | 7.79 | 7,978,856 | 2,852.39M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.29/densenet121-0779-06d5ebbf.npz.log)) |
| DenseNet-161 | 22.36 | 6.20 | 28,681,000 | 7,761.25M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.29/densenet161-0620-6d05f3b9.npz.log)) |
| DenseNet-169 | 23.85 | 6.86 | 14,149,480 | 3,381.48M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.29/densenet169-0686-1978656b.npz.log)) |
| DenseNet-201 | 22.64 | 6.29 | 20,013,928 | 4,318.75M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.29/densenet201-0629-77702939.npz.log)) |
| CondenseNet-74 (C=G=4) | 26.81 | 8.61 | 4,773,944 | 533.64M | From [ShichenLiu/CondenseNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.36/condensenet74_c4_g4-0861-ef6077ec.npz.log)) |
| CondenseNet-74 (C=G=8) | 29.74 | 10.43 | 2,935,416 | 278.55M | From [ShichenLiu/CondenseNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.36/condensenet74_c8_g8-1043-277fbfb8.npz.log)) |
| WRN-50-2 | 22.06 | 6.13 | 68,849,128 | 11,412.82M | From [szagoruyko/functional-zoo] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.113/wrn50_2-0613-d0cd9171.npz.log)) |
| DPN-C-26 | 25.68 | 7.88 | 21,126,584 | 20,838.70M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnc26-0788-762c34c1.npz.log)) |
| DPN-C-42 | 23.72 | 6.93 | 31,234,744 | 31,236.97M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnc42-0693-ec938cc4.npz.log)) |
| DPN-C-58 | 22.35 | 6.29 | 40,542,008 | 36,862.32M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnc58-0629-063ef199.npz.log)) |
| DPN-D-22 | 26.65 | 8.50 | 16,393,752 | 16,626.00M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd22-0850-b25d4757.npz.log)) |
| DPN-D-38 | 24.53 | 7.36 | 26,501,912 | 27,024.27M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd38-0736-153481d6.npz.log)) |
| DPN-D-54 | 22.08 | 6.23 | 35,809,176 | 32,649.62M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd54-0623-31e8eeb8.npz.log)) |
| DPN-D-105 | 21.32 | 5.84 | 54,801,304 | 48,682.11M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd105-0584-c0d7657b.npz.log)) |
| DPN-68 | 23.61 | 7.01 | 12,611,602 | 2,338.71M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.34/dpn68-0701-ad8cd4ec.npz.log)) |
| DPN-98 | 20.80 | 5.53 | 61,570,728 | 11,702.80M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.34/dpn98-0553-9cd57335.npz.log)) |
| DPN-131 | 20.04 | 5.23 | 79,254,504 | 16,056.22M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.34/dpn131-0523-e3721599.npz.log)) |
| DarkNet Tiny | 40.33 | 17.46 | 1,042,104 | 496.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.69/darknet_tiny-1746-b04fa463.npz.log)) |
| DarkNet Ref | 38.09 | 16.71 | 7,319,416 | 365.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.64/darknet_ref-1671-b2d5721f.npz.log)) |
| SqueezeNet v1.0 | 41.01 | 18.96 | 1,248,424 | 828.30M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.20/squeezenet_v1_0-1896-6cbb35ce.npz.log)) |
| SqueezeNet v1.1 | 39.13 | 17.40 | 1,235,496 | 354.88M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.88/squeezenet_v1_1-1740-b236c204.npz.log)) |
| SqueezeResNet v1.1 | 39.85 | 17.87 | 1,235,496 | 354.88M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.70/squeezeresnet_v1_1-1787-f40e6051.npz.log)) |
| ShuffleNetV2 x0.5 | 43.45 | 20.73 | 1,366,792 | 42.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.90/shufflenetv2_wd2-2073-c5e5a23c.npz.log)) |
| ShuffleNetV2b x0.5 | 40.95 | 18.56 | 1,366,792 | 42.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.112/shufflenetv2b_wd2-1856-4d6e16de.npz.log)) |
| ShuffleNetV2c x0.5 | 39.82 | 18.14 | 1,366,792 | 42.37M | From [tensorpack/tensorpack] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.94/shufflenetv2c_wd2-1814-20fc1e3c.npz.log)) |
| ShuffleNetV2 x1.0 | 35.69 | 14.71 | 2,278,604 | 147.92M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.93/shufflenetv2_w1-1471-5698695f.npz.log)) |
| ShuffleNetV2c x1.0 | 30.74 | 11.37 | 2,279,760 | 148.85M | From [tensorpack/tensorpack] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.95/shufflenetv2c_w1-1137-2f59108a.npz.log)) |
| ShuffleNetV2 x1.5 | 33.96 | 13.37 | 4,406,098 | 318.61M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.65/shufflenetv2_w3d2-1337-66c1d6ed.npz.log)) |
| ShuffleNetV2 x2.0 | 33.21 | 13.03 | 7,601,686 | 593.66M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.84/shufflenetv2_w2-1303-349e42b5.npz.log)) |
| 108-MENet-8x1 (g=3) | 43.67 | 20.42 | 654,516 | 40.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.89/menet108_8x1_g3-2042-9e3ff283.npz.log)) |
| 128-MENet-8x1 (g=4) | 42.07 | 19.19 | 750,796 | 43.58M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.103/menet128_8x1_g4-1919-f6fd56fa.npz.log)) |
| 228-MENet-12x1 (g=3) | 34.93 | 14.01 | 1,806,568 | 148.93M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.33/menet228_12x1_g3-1401-07a0ace2.npz.log)) |
| 256-MENet-12x1 (g=4) | 34.44 | 13.91 | 1,888,240 | 146.11M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.33/menet256_12x1_g4-1391-ee68bd6f.npz.log)) |
| 348-MENet-12x1 (g=3) | 31.14 | 11.40 | 3,368,128 | 306.31M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.33/menet348_12x1_g3-1140-49feaea7.npz.log)) |
| 352-MENet-12x1 (g=8) | 34.62 | 13.68 | 2,272,872 | 151.03M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.33/menet352_12x1_g8-1368-2d523fac.npz.log)) |
| 456-MENet-24x1 (g=3) | 29.55 | 10.39 | 5,304,784 | 560.72M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.33/menet456_24x1_g3-1039-f68c36a2.npz.log)) |
| MobileNet x0.25 | 45.85 | 22.16 | 470,072 | 42.30M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.62/mobilenet_wd4-2216-09c50ab8.npz.log)) |
| MobileNet x0.5 | 36.15 | 14.86 | 1,331,592 | 152.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.66/mobilenet_wd2-1486-90e62dd6.npz.log)) |
| MobileNet x0.75 | 33.24 | 12.52 | 2,585,560 | 329.22M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.21/mobilenet_w3d4-1252-6675b58c.npz.log)) |
| MobileNet x1.0 | 29.71 | 10.31 | 4,231,976 | 573.83M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.21/mobilenet_w1-1031-3ecb405b.npz.log)) |
| FD-MobileNet x0.25 | 56.11 | 31.45 | 383,160 | 12.44M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.68/fdmobilenet_wd4-3145-6718fb07.npz.log)) |
| FD-MobileNet x0.5 | 42.68 | 19.76 | 993,928 | 40.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.83/fdmobilenet_wd2-1976-6299d442.npz.log)) |
| FD-MobileNet x1.0 | 35.94 | 14.70 | 2,901,288 | 146.08M | From [clavichord93/FD-MobileNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.25/fdmobilenet_w1-1470-b40709cb.npz.log)) |
| MobileNetV2 x0.25 | 49.11 | 25.49 | 1,516,392 | 32.22M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.31/mobilenetv2_wd4-2549-b5ff8bfd.npz.log)) |
| MobileNetV2 x0.5 | 35.96 | 14.98 | 1,964,736 | 95.62M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.31/mobilenetv2_wd2-1498-4b767a98.npz.log)) |
| MobileNetV2 x0.75 | 31.28 | 11.48 | 2,627,592 | 191.61M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.31/mobilenetv2_w3d4-1148-a6f852ea.npz.log)) |
| MobileNetV2 x1.0 | 28.87 | 10.05 | 3,504,960 | 320.19M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.31/mobilenetv2_w1-1005-3b6d1764.npz.log)) |
| MnasNet | 32.33 | 12.05 | 4,308,816 | 310.75M | From [zeusees/Mnasnet...Model] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.106/mnasnet-1205-7bc88b51.npz.log)) |
| Xception | 21.04 | 5.47 | 22,855,952 | 8,385.86M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.115/xception-0547-7a5be958.npz.log)) |
| InceptionV3 | 21.11 | 5.61 | 23,834,568 | 5,746.72M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.92/inceptionv3-0561-4ddea4df.npz.log)) |
| InceptionV4 | 20.62 | 5.26 | 42,679,816 | 12,314.17M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.105/inceptionv4-0526-02e53701.npz.log)) |
| InceptionResNetV2 | 19.93 | 4.92 | 55,843,464 | 13,189.58M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.107/inceptionresnetv2-0492-3d3de82b.npz.log)) |
| PolyNet | 19.08 | 4.50 | 95,366,600 | 34,768.84M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.96/polynet-0450-6dc7028b.npz.log)) |
| NASNet-A 4@1056 | 25.36 | 7.96 | 5,289,978 | 587.29M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.97/nasnet_4a1056-0796-f09950c0.npz.log)) |
| NASNet-A 6@4032 | 18.17 | 4.22 | 88,753,150 | 24,021.18M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.101/nasnet_6a4032-0422-d49d4663.npz.log)) |
| PNASNet-5-Large | 17.90 | 4.26 | 86,057,668 | 25,169.47M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.114/pnasnet5large-0426-3c2755dc.npz.log)) |

### For Keras

| Model | Top1 | Top5 | Params | FLOPs | Remarks |
| --- | ---: | ---: | ---: | ---: | --- |
| ResNet-10 | 37.09 | 15.54 | 5,418,792 | 892.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.49/resnet10-1554-294a0786.h5.log)) |
| ResNet-12 | 35.86 | 14.45 | 5,492,776 | 1,124.23M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.49/resnet12-1445-285da75b.h5.log)) |
| ResNet-14 | 32.85 | 12.42 | 5,788,200 | 1,355.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.49/resnet14-1242-e2ffca6e.h5.log)) |
| ResNet-16 | 30.67 | 11.09 | 6,968,872 | 1,586.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.49/resnet16-1109-8f70f97e.h5.log)) |
| ResNet-18 x0.25 | 49.14 | 24.45 | 831,096 | 136.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.49/resnet18_wd4-2445-dd6ba54d.h5.log)) |
| ResNet-18 x0.5 | 36.54 | 14.96 | 3,055,880 | 485.22M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.49/resnet18_wd2-1496-9bc78e3b.h5.log)) |
| ResNet-18 x0.75 | 33.24 | 12.54 | 6,675,352 | 1,045.75M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.49/resnet18_w3d4-1254-f6374cc3.h5.log)) |
| ResNet-18 | 29.13 | 9.94 | 11,689,512 | 1,818.21M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.49/resnet18-0994-3ff2352a.h5.log)) |
| ResNet-34 | 25.32 | 7.92 | 21,797,672 | 3,669.16M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.49/resnet34-0792-3ea662f5.h5.log)) |
| ResNet-50 | 23.49 | 6.87 | 25,557,032 | 3,868.96M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.49/resnet50-0687-9eb5e8d7.h5.log)) |
| ResNet-50b | 22.90 | 6.44 | 25,557,032 | 4,100.70M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.49/resnet50b-0644-fd813b71.h5.log)) |
| ResNet-101 | 21.64 | 5.99 | 44,549,160 | 7,586.30M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.49/resnet101-0599-ab428947.h5.log)) |
| ResNet-101b | 21.17 | 5.60 | 44,549,160 | 7,818.04M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.49/resnet101b-0560-241918fa.h5.log)) |
| ResNet-152 | 21.00 | 5.61 | 60,192,808 | 11,304.85M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.49/resnet152-0561-001efbff.h5.log)) |
| ResNet-152b | 20.53 | 5.37 | 60,192,808 | 11,536.58M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.49/resnet152b-0537-8870623c.h5.log)) |
| PreResNet-18 | 28.72 | 9.88 | 11,687,848 | 1,818.41M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.50/preresnet18-0988-36f6c05c.h5.log)) |
| PreResNet-34 | 25.86 | 8.11 | 21,796,008 | 3,669.36M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.50/preresnet34-0811-1663d695.h5.log)) |
| PreResNet-50 | 23.38 | 6.68 | 25,549,480 | 3,869.16M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.50/preresnet50-0668-90326d19.h5.log)) |
| PreResNet-50b | 23.14 | 6.63 | 25,549,480 | 4,100.90M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.50/preresnet50b-0663-c30588ee.h5.log)) |
| PreResNet-101 | 21.43 | 5.75 | 44,541,608 | 7,586.50M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.50/preresnet101-0575-5dff088d.h5.log)) |
| PreResNet-101b | 21.71 | 5.88 | 44,541,608 | 7,818.24M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.50/preresnet101b-0588-fad1f60c.h5.log)) |
| PreResNet-152 | 20.69 | 5.31 | 60,185,256 | 11,305.05M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.50/preresnet152-0531-a5ac128d.h5.log)) |
| PreResNet-152b | 20.99 | 5.76 | 60,185,256 | 11,536.78M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.50/preresnet152b-0576-ea9dda1e.h5.log)) |
| PreResNet-200b | 21.09 | 5.64 | 64,666,280 | 15,040.27M | From [tornadomeet/ResNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.50/preresnet200b-0564-9172d4c0.h5.log)) |
| ResNeXt-101 (32x4d) | 21.30 | 5.78 | 44,177,704 | 7,991.62M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.51/resnext101_32x4d-0578-7623f640.h5.log)) |
| ResNeXt-101 (64x4d) | 20.59 | 5.41 | 83,455,272 | 15,491.88M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.51/resnext101_64x4d-0541-7b58eaae.h5.log)) |
| SE-ResNet-50 | 22.50 | 6.43 | 28,088,024 | 3,877.01M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.52/seresnet50-0643-fabfa406.h5.log)) |
| SE-ResNet-101 | 21.92 | 5.88 | 49,326,872 | 7,600.01M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.52/seresnet101-0588-933d3415.h5.log)) |
| SE-ResNet-152 | 21.46 | 5.77 | 66,821,848 | 11,324.62M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.52/seresnet152-0577-d25ced7d.h5.log)) |
| SE-ResNeXt-50 (32x4d) | 21.05 | 5.57 | 27,559,896 | 4,253.33M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.53/seresnext50_32x4d-0557-997ef4dd.h5.log)) |
| SE-ResNeXt-101 (32x4d) | 19.98 | 4.99 | 48,955,416 | 8,005.33M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.53/seresnext101_32x4d-0499-59e4e584.h5.log)) |
| SENet-154 | 18.83 | 4.65 | 115,088,984 | 20,742.40M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.54/senet154-0465-962aeede.h5.log)) |
| DenseNet-121 | 25.09 | 7.80 | 7,978,856 | 2,852.39M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.55/densenet121-0780-52b0611c.h5.log)) |
| DenseNet-161 | 22.39 | 6.18 | 28,681,000 | 7,761.25M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.55/densenet161-0618-070fcb45.h5.log)) |
| DenseNet-169 | 23.88 | 6.89 | 14,149,480 | 3,381.48M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.55/densenet169-0689-ae41b4a6.h5.log)) |
| DenseNet-201 | 22.69 | 6.35 | 20,013,928 | 4,318.75M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.55/densenet201-0635-cf3afbb2.h5.log)) |
| DarkNet Tiny | 40.31 | 17.46 | 1,042,104 | 496.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.69/darknet_tiny-1746-147e949b.h5.log)) |
| DarkNet Ref | 37.99 | 16.68 | 7,319,416 | 365.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.64/darknet_ref-1668-2ef080bb.h5.log)) |
| SqueezeNet v1.0 | 41.07 | 19.04 | 1,248,424 | 828.30M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.57/squeezenet_v1_0-1904-c2c87509.h5.log)) |
| SqueezeNet v1.1 | 39.08 | 17.39 | 1,235,496 | 354.88M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.88/squeezenet_v1_1-1739-b9a8f9ea.h5.log)) |
| SqueezeResNet v1.1 | 39.82 | 17.84 | 1,235,496 | 354.88M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.70/squeezeresnet_v1_1-1784-43ee9cbb.h5.log)) |
| ShuffleNetV2 x0.5 | 40.76 | 18.40 | 1,366,792 | 42.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.90/shufflenetv2_wd2-1840-9b4b0964.h5.log)) |
| ShuffleNetV2 x1.0 | 33.79 | 13.38 | 2,278,604 | 147.92M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.93/shufflenetv2_w1-1338-cb35f09e.h5.log)) |
| ShuffleNetV2 x1.5 | 32.46 | 12.47 | 4,406,098 | 318.61M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.65/shufflenetv2_w3d2-1247-f7f813b4.h5.log)) |
| ShuffleNetV2 x2.0 | 31.91 | 12.23 | 7,601,686 | 593.66M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.84/shufflenetv2_w2-1223-63291468.h5.log)) |
| 108-MENet-8x1 (g=3) | 43.61 | 20.31 | 654,516 | 40.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.89/menet108_8x1_g3-2031-a4d43433.h5.log)) |
| 128-MENet-8x1 (g=4) | 42.08 | 19.14 | 750,796 | 43.58M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.103/menet128_8x1_g4-1914-5bb8f228.h5.log)) |
| 228-MENet-12x1 (g=3) | 35.02 | 14.01 | 1,806,568 | 148.93M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.58/menet228_12x1_g3-1401-954b3ba0.h5.log)) |
| 256-MENet-12x1 (g=4) | 34.48 | 13.91 | 1,888,240 | 146.11M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.58/menet256_12x1_g4-1391-a63a606a.h5.log)) |
| 348-MENet-12x1 (g=3) | 31.17 | 11.42 | 3,368,128 | 306.31M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.58/menet348_12x1_g3-1142-0715c866.h5.log)) |
| 352-MENet-12x1 (g=8) | 34.69 | 13.75 | 2,272,872 | 151.03M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.58/menet352_12x1_g8-1375-9007c933.h5.log)) |
| 456-MENet-24x1 (g=3) | 29.55 | 10.44 | 5,304,784 | 560.72M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.58/menet456_24x1_g3-1044-c090af59.h5.log)) |
| MobileNet x0.25 | 45.80 | 22.17 | 470,072 | 42.30M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.62/mobilenet_wd4-2217-fb7abda8.h5.log)) |
| MobileNet x0.5 | 36.11 | 14.81 | 1,331,592 | 152.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.66/mobilenet_wd2-1481-b78fec33.h5.log)) |
| MobileNet x0.75 | 32.71 | 12.28 | 2,585,560 | 329.22M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.59/mobilenet_w3d4-1228-09a1eb55.h5.log)) |
| MobileNet x1.0 | 29.24 | 10.03 | 4,231,976 | 573.83M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.59/mobilenet_w1-1003-ec69d89b.h5.log)) |
| FD-MobileNet x0.25 | 56.17 | 31.37 | 383,160 | 12.44M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.68/fdmobilenet_wd4-3137-153934e4.h5.log)) |
| FD-MobileNet x0.5 | 42.61 | 19.69 | 993,928 | 40.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.83/fdmobilenet_wd2-1969-5678a212.h5.log)) |
| FD-MobileNet x1.0 | 35.95 | 14.73 | 2,901,288 | 146.08M | From [clavichord93/FD-MobileNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.60/fdmobilenet_w1-1473-680e603f.h5.log)) |
| MobileNetV2 x0.25 | 48.86 | 25.24 | 1,516,392 | 32.22M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.61/mobilenetv2_wd4-2524-a8ea2889.h5.log)) |
| MobileNetV2 x0.5 | 35.51 | 14.65 | 1,964,736 | 95.62M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.61/mobilenetv2_wd2-1465-774d5bca.h5.log)) |
| MobileNetV2 x0.75 | 30.81 | 11.26 | 2,627,592 | 191.61M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.61/mobilenetv2_w3d4-1126-f2f664da.h5.log)) |
| MobileNetV2 x1.0 | 28.50 | 9.90 | 3,504,960 | 320.19M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.61/mobilenetv2_w1-0990-cbb8be96.h5.log)) |
| MnasNet | 32.32 | 12.03 | 4,308,816 | 310.75M | From [zeusees/Mnasnet...Model] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.106/mnasnet-1203-76505508.h5.log)) |

### For TensorFlow

| Model | Top1 | Top5 | Params | FLOPs | Remarks |
| --- | ---: | ---: | ---: | ---: | --- |
| ResNet-10 | 37.11 | 15.52 | 5,418,792 | 892.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.72/resnet10-1552-e2c11848.tf.npz.log)) |
| ResNet-12 | 35.82 | 14.50 | 5,492,776 | 1,124.23M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.72/resnet12-1450-8865f58b.tf.npz.log)) |
| ResNet-14 | 32.83 | 12.45 | 5,788,200 | 1,355.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.72/resnet14-1245-8596c8f1.tf.npz.log)) |
| ResNet-16 | 30.66 | 11.05 | 6,968,872 | 1,586.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.72/resnet16-1105-8ee84db2.tf.npz.log)) |
| ResNet-18 x0.25 | 49.12 | 24.50 | 831,096 | 136.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.72/resnet18_wd4-2450-b536eea5.tf.npz.log)) |
| ResNet-18 x0.5 | 36.51 | 14.93 | 3,055,880 | 485.22M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.72/resnet18_wd2-1493-dfb5d150.tf.npz.log)) |
| ResNet-18 x0.75 | 33.28 | 12.50 | 6,675,352 | 1,045.75M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.72/resnet18_w3d4-1250-2040e339.tf.npz.log)) |
| ResNet-18 | 29.10 | 9.99 | 11,689,512 | 1,818.21M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.72/resnet18-0999-b2422403.tf.npz.log)) |
| ResNet-34 | 25.32 | 7.93 | 21,797,672 | 3,669.16M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.72/resnet34-0793-aaf4f066.tf.npz.log)) |
| ResNet-50 | 23.48 | 6.87 | 25,557,032 | 3,868.96M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.72/resnet50-0687-3fd48a1a.tf.npz.log)) |
| ResNet-50b | 22.97 | 6.48 | 25,557,032 | 4,100.70M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.72/resnet50b-0648-6a8d5eda.tf.npz.log)) |
| ResNet-101 | 21.61 | 6.01 | 44,549,160 | 7,586.30M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.72/resnet101-0601-3fc260bc.tf.npz.log)) |
| ResNet-101b | 21.22 | 5.57 | 44,549,160 | 7,818.04M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.72/resnet101b-0557-8731697c.tf.npz.log)) |
| ResNet-152 | 20.99 | 5.59 | 60,192,808 | 11,304.85M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.72/resnet152-0559-d3c4d7b2.tf.npz.log)) |
| ResNet-152b | 20.55 | 5.35 | 60,192,808 | 11,536.58M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.72/resnet152b-0535-bcccd3d7.tf.npz.log)) |
| PreResNet-18 | 28.75 | 9.88 | 11,687,848 | 1,818.41M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.73/preresnet18-0988-3295cbda.tf.npz.log)) |
| PreResNet-34 | 25.82 | 8.08 | 21,796,008 | 3,669.36M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.73/preresnet34-0808-ceab73cc.tf.npz.log)) |
| PreResNet-50 | 23.42 | 6.68 | 25,549,480 | 3,869.16M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.73/preresnet50-0668-822837cf.tf.npz.log)) |
| PreResNet-50b | 23.12 | 6.61 | 25,549,480 | 4,100.90M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.73/preresnet50b-0661-49f158a2.tf.npz.log)) |
| PreResNet-101 | 21.49 | 5.72 | 44,541,608 | 7,586.50M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.73/preresnet101-0572-cd61594e.tf.npz.log)) |
| PreResNet-101b | 21.70 | 5.91 | 44,541,608 | 7,818.24M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.73/preresnet101b-0591-93ae5e69.tf.npz.log)) |
| PreResNet-152 | 20.63 | 5.29 | 60,185,256 | 11,305.05M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.73/preresnet152-0529-b761f286.tf.npz.log)) |
| PreResNet-152b | 20.95 | 5.76 | 60,185,256 | 11,536.78M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.73/preresnet152b-0576-c036165c.tf.npz.log)) |
| PreResNet-200b | 21.12 | 5.60 | 64,666,280 | 15,040.27M | From [tornadomeet/ResNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.73/preresnet200b-0560-881e0e28.tf.npz.log)) |
| ResNeXt-101 (32x4d) | 21.33 | 5.80 | 44,177,704 | 7,991.62M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.74/resnext101_32x4d-0580-bf746cb6.tf.npz.log)) |
| ResNeXt-101 (64x4d) | 20.59 | 5.43 | 83,455,272 | 15,491.88M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.74/resnext101_64x4d-0543-f51ffdb0.tf.npz.log)) |
| SE-ResNet-50 | 22.53 | 6.43 | 28,088,024 | 3,877.01M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.75/seresnet50-0643-e022e5b9.tf.npz.log)) |
| SE-ResNet-101 | 21.92 | 5.89 | 49,326,872 | 7,600.01M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.75/seresnet101-0589-305d2301.tf.npz.log)) |
| SE-ResNet-152 | 21.48 | 5.78 | 66,821,848 | 11,324.62M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.75/seresnet152-0578-d06ab6d9.tf.npz.log)) |
| SE-ResNeXt-50 (32x4d) | 21.01 | 5.53 | 27,559,896 | 4,253.33M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.76/seresnext50_32x4d-0553-20723214.tf.npz.log)) |
| SE-ResNeXt-101 (32x4d) | 19.99 | 4.97 | 48,955,416 | 8,005.33M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.76/seresnext101_32x4d-0497-268d7d22.tf.npz.log)) |
| SENet-154 | 18.77 | 4.63 | 115,088,984 | 20,742.40M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.86/senet154-0463-c86eaaed.tf.npz.log)) |
| DenseNet-121 | 25.16 | 7.82 | 7,978,856 | 2,852.39M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.77/densenet121-0782-1bfa61d4.tf.npz.log)) |
| DenseNet-161 | 22.40 | 6.17 | 28,681,000 | 7,761.25M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.77/densenet161-0617-9deca33a.tf.npz.log)) |
| DenseNet-169 | 23.93 | 6.87 | 14,149,480 | 3,381.48M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.77/densenet169-0687-23910539.tf.npz.log)) |
| DenseNet-201 | 22.70 | 6.35 | 20,013,928 | 4,318.75M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.77/densenet201-0635-5eda7895.tf.npz.log)) |
| DarkNet Tiny | 40.35 | 17.51 | 1,042,104 | 496.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.71/darknet_tiny-1751-750ff8d9.tf.npz.log)) |
| DarkNet Ref | 37.99 | 16.72 | 7,319,416 | 365.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.71/darknet_ref-1672-3c8ed62a.tf.npz.log)) |
| SqueezeNet v1.0 | 41.13 | 19.02 | 1,248,424 | 828.30M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.78/squeezenet_v1_0-1902-694730ac.tf.npz.log)) |
| SqueezeNet v1.1 | 39.14 | 17.39 | 1,235,496 | 354.88M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.88/squeezenet_v1_1-1739-48945577.tf.npz.log)) |
| SqueezeResNet v1.1 | 39.75 | 17.92 | 1,235,496 | 354.88M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.79/squeezeresnet_v1_1-1792-44c17928.tf.npz.log)) |
| ShuffleNetV2 x0.5 | 40.88 | 18.44 | 1,366,792 | 42.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.90/shufflenetv2_wd2-1844-2bd8a314.tf.npz.log)) |
| ShuffleNetV2b x0.5 | 41.03 | 18.59 | 1,366,792 | 42.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.112/shufflenetv2b_wd2-1859-67249edb.tf.npz.log)) |
| ShuffleNetV2c x0.5 | 39.93 | 18.11 | 1,366,792 | 42.37M | From [tensorpack/tensorpack] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.91/shufflenetv2c_wd2-1811-98435af9.tf.npz.log)) |
| ShuffleNetV2 x1.0 | 33.81 | 13.40 | 2,278,604 | 147.92M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.93/shufflenetv2_w1-1340-b5b2d8c6.tf.npz.log)) |
| ShuffleNetV2c x1.0 | 30.77 | 11.39 | 2,279,760 | 148.85M | From [tensorpack/tensorpack] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.95/shufflenetv2c_w1-1139-47dd03c8.tf.npz.log)) |
| ShuffleNetV2 x1.5 | 32.51 | 12.50 | 4,406,098 | 318.61M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.85/shufflenetv2_w3d2-1250-5dd7b5b1.tf.npz.log)) |
| ShuffleNetV2 x2.0 | 31.99 | 12.26 | 7,601,686 | 593.66M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.85/shufflenetv2_w2-1226-f66f6987.tf.npz.log)) |
| 108-MENet-8x1 (g=3) | 43.67 | 20.32 | 654,516 | 40.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.89/menet108_8x1_g3-2032-4e9e89e1.tf.npz.log)) |
| 128-MENet-8x1 (g=4) | 42.04 | 19.15 | 750,796 | 43.58M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.103/menet128_8x1_g4-1915-148105f4.tf.npz.log)) |
| 228-MENet-12x1 (g=3) | 35.01 | 14.05 | 1,806,568 | 148.93M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.87/menet228_12x1_g3-1405-724b4422.tf.npz.log)) |
| 256-MENet-12x1 (g=4) | 34.48 | 13.95 | 1,888,240 | 146.11M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.87/menet256_12x1_g4-1395-d0ce72b1.tf.npz.log)) |
| 348-MENet-12x1 (g=3) | 31.19 | 11.41 | 3,368,128 | 306.31M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.87/menet348_12x1_g3-1141-f90f3c12.tf.npz.log)) |
| 352-MENet-12x1 (g=8) | 34.65 | 13.71 | 2,272,872 | 151.03M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.87/menet352_12x1_g8-1371-3621d3c0.tf.npz.log)) |
| 456-MENet-24x1 (g=3) | 29.56 | 10.46 | 5,304,784 | 560.72M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.87/menet456_24x1_g3-1046-6d70fb21.tf.npz.log)) |
| MobileNet x0.25 | 45.78 | 22.21 | 470,072 | 42.30M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.80/mobilenet_wd4-2221-15ee9820.tf.npz.log)) |
| MobileNet x0.5 | 36.18 | 14.84 | 1,331,592 | 152.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.80/mobilenet_wd2-1484-84d1de07.tf.npz.log)) |
| MobileNet x0.75 | 32.70 | 12.27 | 2,585,560 | 329.22M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.80/mobilenet_w3d4-1227-7f3f25dc.tf.npz.log)) |
| MobileNet x1.0 | 29.30 | 10.04 | 4,231,976 | 573.83M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.80/mobilenet_w1-1004-fa4fdc2e.tf.npz.log)) |
| FD-MobileNet x0.25 | 56.08 | 31.44 | 383,160 | 12.44M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.81/fdmobilenet_wd4-3144-3febaec9.tf.npz.log)) |
| FD-MobileNet x0.5 | 42.67 | 19.70 | 993,928 | 40.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.83/fdmobilenet_wd2-1970-d778e687.tf.npz.log)) |
| FD-MobileNet x1.0 | 36.02 | 14.76 | 2,901,288 | 146.08M | From [clavichord93/FD-MobileNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.81/fdmobilenet_w1-1476-4db4956b.tf.npz.log)) |
| MobileNetV2 x0.25 | 48.87 | 25.26 | 1,516,392 | 32.22M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.82/mobilenetv2_wd4-2526-b1697003.tf.npz.log)) |
| MobileNetV2 x0.5 | 35.51 | 14.60 | 1,964,736 | 95.62M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.82/mobilenetv2_wd2-1460-12376d24.tf.npz.log)) |
| MobileNetV2 x0.75 | 30.79 | 11.24 | 2,627,592 | 191.61M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.82/mobilenetv2_w3d4-1124-3531c997.tf.npz.log)) |
| MobileNetV2 x1.0 | 28.53 | 9.90 | 3,504,960 | 320.19M | Converted from Gluon Model Zoo ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.82/mobilenetv2_w1-0990-e80f9fe4.tf.npz.log)) |
| MnasNet | 32.34 | 12.07 | 4,308,816 | 310.75M | From [zeusees/Mnasnet...Model] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.106/mnasnet-1207-929dc499.tf.npz.log)) |

[tornadomeet/ResNet]: https://github.com/tornadomeet/ResNet
[Cadene/pretrained...pytorch]: https://github.com/Cadene/pretrained-models.pytorch
[ShichenLiu/CondenseNet]: https://github.com/ShichenLiu/CondenseNet
[clavichord93/MENet]: https://github.com/clavichord93/MENet
[clavichord93/FD-MobileNet]: https://github.com/clavichord93/FD-MobileNet
[tensorpack/tensorpack]: https://github.com/tensorpack/tensorpack
[dyhan0920/Pyramid...PyTorch]: https://github.com/dyhan0920/PyramidNet-PyTorch
[zeusees/Mnasnet...Model]: https://github.com/zeusees/Mnasnet-Pretrained-Model
[szagoruyko/diracnets]: https://github.com/szagoruyko/diracnets
[szagoruyko/functional-zoo]: https://github.com/szagoruyko/functional-zoo
[fyu/drn]: https://github.com/fyu/drn