# Large-scale image classification networks for embedded systems

[![Build Status](https://travis-ci.org/osmr/imgclsmob.svg?branch=master)](https://travis-ci.org/osmr/imgclsmob)
[![GitHub License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-2.7%2C3.6-lightgrey.svg)](https://github.com/osmr/imgclsmob)

This repository contains several classification models on MXNet/Gluon, PyTorch, Chainer, Keras, and TensorFlow, with
scripts for training/validating/converting models. All models are designed for using with ImageNet-1k dataset.

## List of models

- AlexNet (['One weird trick for parallelizing convolutional neural networks'](https://arxiv.org/abs/1404.5997))
- ZFNet (['Visualizing and Understanding Convolutional Networks'](https://arxiv.org/abs/1311.2901))
- VGG/BN-VGG (['Very Deep Convolutional Networks for Large-Scale Image Recognition'](https://arxiv.org/abs/1409.1556))
- BN-Inception (['Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift'](https://arxiv.org/abs/1502.03167))
- ResNet (['Deep Residual Learning for Image Recognition'](https://arxiv.org/abs/1512.03385))
- PreResNet (['Identity Mappings in Deep Residual Networks'](https://arxiv.org/abs/1603.05027))
- ResNeXt (['Aggregated Residual Transformations for Deep Neural Networks'](http://arxiv.org/abs/1611.05431))
- SENet/SE-ResNet/SE-PreResNet/SE-ResNeXt (['Squeeze-and-Excitation Networks'](https://arxiv.org/abs/1709.01507))
- IBN-ResNet/IBN-ResNeXt/IBN-DenseNet (['Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net'](https://arxiv.org/abs/1807.09441))
- AirNet/AirNeXt (['Attention Inspiring Receptive-Fields Network for Learning Invariant Representations'](https://ieeexplore.ieee.org/document/8510896))
- BAM-ResNet (['BAM: Bottleneck Attention Module'](https://arxiv.org/abs/1807.06514))
- CBAM-ResNet (['CBAM: Convolutional Block Attention Module'](https://arxiv.org/abs/1807.06521))
- ResAttNet (['Residual Attention Network for Image Classification'](https://arxiv.org/abs/1704.06904))
- PyramidNet (['Deep Pyramidal Residual Networks'](https://arxiv.org/abs/1610.02915))
- DiracNetV2 (['DiracNets: Training Very Deep Neural Networks Without Skip-Connections'](https://arxiv.org/abs/1706.00388))
- DenseNet (['Densely Connected Convolutional Networks'](https://arxiv.org/abs/1608.06993))
- CondenseNet (['CondenseNet: An Efficient DenseNet using Learned Group Convolutions'](https://arxiv.org/abs/1711.09224))
- SparseNet (['Sparsely Aggregated Convolutional Networks'](https://arxiv.org/abs/1801.05895))
- PeleeNet (['Pelee: A Real-Time Object Detection System on Mobile Devices'](https://arxiv.org/abs/1804.06882))
- WRN (['Wide Residual Networks'](https://arxiv.org/abs/1605.07146))
- DRN-C/DRN-D (['Dilated Residual Networks'](https://arxiv.org/abs/1705.09914))
- DPN (['Dual Path Networks'](https://arxiv.org/abs/1707.01629))
- DarkNet Ref/Tiny/19 (['Darknet: Open source neural networks in c'](https://github.com/pjreddie/darknet))
- DarkNet-53 (['YOLOv3: An Incremental Improvement'](https://arxiv.org/abs/1804.02767))
- ChannelNet (['ChannelNets: Compact and Efficient Convolutional Neural Networks via Channel-Wise Convolutions'](https://arxiv.org/abs/1809.01330))
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
- DARTS (['DARTS: Differentiable Architecture Search'](https://arxiv.org/abs/1806.09055))
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
- All models require the same ordinary normalization.
- FLOPs/2 is the number of FLOPs divided by two to be similar to the number of MACs.

### For Gluon

| Model | Top1 | Top5 | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | ---: | --- |
| AlexNet | 44.12 | 21.26 | 61,100,840 | 714.83M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.108/alexnet-2126-9cb87ebd.params.log)) |
| VGG-11 | 31.91 | 11.76 | 132,863,336 | 7,615.87M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg11-1176-95dd287d.params.log)) |
| VGG-13 | 31.06 | 11.12 | 133,047,848 | 11,317.65M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg13-1112-a0db3c6c.params.log)) |
| VGG-16 | 26.78 | 8.69 | 138,357,544 | 15,480.10M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg16-0869-57a2556f.params.log)) |
| VGG-19 | 25.88 | 8.23 | 143,667,240 | 19,642.55M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg19-0823-0e2a1e0a.params.log)) |
| BN-VGG-11b | 30.34 | 10.57 | 132,868,840 | 7,630.72M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg11b-1057-b2d8f382.params.log)) |
| BN-VGG-13b | 29.48 | 10.16 | 133,053,736 | 11,342.14M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg13b-1016-f384ff52.params.log)) |
| BN-VGG-16b | 26.89 | 8.65 | 138,365,992 | 15,507.20M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg16b-0865-b5e33db8.params.log)) |
| BN-VGG-19b | 25.66 | 8.15 | 143,678,248 | 19,672.26M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg19b-0815-3a0e43e6.params.log)) |
| BN-Inception | 25.09 | 7.76 | 11,295,240 | 2,048.06M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.139/bninception-0776-8314001b.params.log)) |
| ResNet-10 | 37.09 | 15.55 | 5,418,792 | 894.04M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet10-1555-cfb0a76d.params.log)) |
| ResNet-12 | 35.86 | 14.46 | 5,492,776 | 1,126.25M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.30/resnet12-1446-9ce715b0.params.log)) |
| ResNet-14 | 32.85 | 12.41 | 5,788,200 | 1,357.94M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.40/resnet14-1241-a8955ff3.params.log)) |
| ResNet-16 | 30.68 | 11.10 | 6,968,872 | 1,589.34M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.41/resnet16-1110-1be996d1.params.log)) |
| ResNet-18 x0.25 | 49.16 | 24.45 | 831,096 | 137.32M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.47/resnet18_wd4-2445-28d15cf4.params.log)) |
| ResNet-18 x0.5 | 36.54 | 14.96 | 3,055,880 | 486.49M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.46/resnet18_wd2-1496-d839c509.params.log)) |
| ResNet-18 x0.75 | 33.25 | 12.54 | 6,675,352 | 1,047.53M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.18/resnet18_w3d4-1254-d6548612.params.log)) |
| ResNet-18 | 28.21 | 9.67 | 11,689,512 | 1,820.41M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.151/resnet18-0967-dbc58dee.params.log)) |
| ResNet-34 | 25.34 | 7.92 | 21,797,672 | 3,672.68M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet34-0792-5b875f49.params.log)) |
| ResNet-50 | 22.65 | 6.41 | 25,557,032 | 3,877.95M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.147/resnet50-0641-1eaa883b.params.log)) |
| ResNet-50b | 22.32 | 6.18 | 25,557,032 | 4,110.48M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.146/resnet50b-0618-8e2541fb.params.log)) |
| ResNet-101 | 21.66 | 5.99 | 44,549,160 | 7,597.95M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet101-0599-a6d3a5f4.params.log)) |
| ResNet-101b | 20.79 | 5.39 | 44,549,160 | 7,830.48M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.145/resnet101b-0539-7406d858.params.log)) |
| ResNet-152 | 20.76 | 5.35 | 60,192,808 | 11,321.85M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.144/resnet152-0535-bbdd7ed1.params.log)) |
| ResNet-152b | 20.31 | 5.25 | 60,192,808 | 11,554.38M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.143/resnet152b-0525-6f30d0d9.params.log)) |
| PreResNet-18 | 28.16 | 9.51 | 11,687,848 | 1,820.56M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.140/preresnet18-0951-71279a0b.params.log)) |
| PreResNet-34 | 25.88 | 8.11 | 21,796,008 | 3,672.83M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet34-0811-f8fe98a2.params.log)) |
| PreResNet-50 | 23.39 | 6.68 | 25,549,480 | 3,875.44M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet50-0668-4940c94b.params.log)) |
| PreResNet-50b | 23.16 | 6.64 | 25,549,480 | 4,107.97M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet50b-0664-2fcfddb1.params.log)) |
| PreResNet-101 | 21.45 | 5.75 | 44,541,608 | 7,595.44M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet101-0575-e2887e53.params.log)) |
| PreResNet-101b | 21.73 | 5.88 | 44,541,608 | 7,827.97M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet101b-0588-1015145a.params.log)) |
| PreResNet-152 | 20.70 | 5.32 | 60,185,256 | 11,319.34M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.14/preresnet152-0532-31505f71.params.log)) |
| PreResNet-152b | 21.00 | 5.75 | 60,185,256 | 11,551.87M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet152b-0575-dc303191.params.log)) |
| PreResNet-200b | 21.10 | 5.64 | 64,666,280 | 15,068.63M | From [tornadomeet/ResNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.45/preresnet200b-0564-38f849a6.params.log)) |
| ResNeXt-101 (32x4d) | 21.32 | 5.79 | 44,177,704 | 8,003.45M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.10/resnext101_32x4d-0579-9afbfdbc.params.log)) |
| ResNeXt-101 (64x4d) | 20.60 | 5.41 | 83,455,272 | 15,500.27M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.10/resnext101_64x4d-0541-0d4fd87b.params.log)) |
| SE-ResNet-50 | 22.51 | 6.44 | 28,088,024 | 3,880.49M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.11/seresnet50-0644-10954a84.params.log)) |
| SE-ResNet-101 | 21.92 | 5.89 | 49,326,872 | 7,602.76M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.11/seresnet101-0589-4c10238d.params.log)) |
| SE-ResNet-152 | 21.48 | 5.77 | 66,821,848 | 11,328.52M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.11/seresnet152-0577-de6f099d.params.log)) |
| SE-ResNeXt-50 (32x4d) | 21.06 | 5.58 | 27,559,896 | 4,258.40M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.12/seresnext50_32x4d-0558-a49f8fb0.params.log)) |
| SE-ResNeXt-101 (32x4d) | 19.99 | 5.00 | 48,955,416 | 8,008.26M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.12/seresnext101_32x4d-0500-cf161260.params.log)) |
| SENet-154 | 18.84 | 4.65 | 115,088,984 | 20,745.78M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.13/senet154-0465-dd244507.params.log)) |
| IBN-ResNet-50 | 23.56 | 6.68 | 25,557,032 | 4,110.48M | From [XingangPan/IBN-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.127/ibn_resnet50-0668-db527596.params.log)) |
| IBN-ResNet-101 | 21.89 | 5.87 | 44,549,160 | 7,830.48M | From [XingangPan/IBN-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.127/ibn_resnet101-0587-946e7f10.params.log)) |
| IBN(b)-ResNet-50 | 23.91 | 6.97 | 25,558,568 | 4,112.89M | From [XingangPan/IBN-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.127/ibnb_resnet50-0697-0aea51d2.params.log)) |
| IBN-ResNeXt-101 (32x4d) | 21.43 | 5.62 | 44,177,704 | 8,003.45M | From [XingangPan/IBN-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.127/ibn_resnext101_32x4d-0562-05ddba79.params.log)) |
| IBN-DenseNet-121 | 24.98 | 7.47 | 7,978,856 | 2,872.13M | From [XingangPan/IBN-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.127/ibn_densenet121-0747-1434d379.params.log)) |
| IBN-DenseNet-169 | 23.78 | 6.82 | 14,149,480 | 3,403.89M | From [XingangPan/IBN-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.127/ibn_densenet169-0682-6d7c48c5.params.log)) |
| AirNet50-1x64d (r=2) | 22.48 | 6.21 | 27,425,864 | 4,772.11M | From [soeaver/AirNet-PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.120/airnet50_1x64d_r2-0621-347358cc.params.log)) |
| AirNet50-1x64d (r=16) | 22.91 | 6.46 | 25,714,952 | 4,399.97M | From [soeaver/AirNet-PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.120/airnet50_1x64d_r16-0646-0b847b99.params.log)) |
| AirNeXt50-32x4d (r=2) | 21.51 | 5.75 | 27,604,296 | 5,339.58M | From [soeaver/AirNet-PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.120/airnext50_32x4d_r2-0575-ab104fb5.params.log)) |
| BAM-ResNet-50 | 23.68 | 6.96 | 25,915,099 | 4,196.09M | From [Jongchan/attention-module] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.124/bam_resnet50-0696-7e573b61.params.log)) |
| CBAM-ResNet-50 | 23.02 | 6.38 | 28,089,624 | 4,116.97M | From [Jongchan/attention-module] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.125/cbam_resnet50-0638-78be5665.params.log)) |
| PyramidNet-101 (a=360) | 22.72 | 6.52 | 42,455,070 | 8,743.54M | From [dyhan0920/Pyramid...PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.104/pyramidnet101_a360-0652-08d5a5d1.params.log)) |
| DiracNetV2-18 | 30.61 | 11.17 | 11,511,784 | 1,796.62M | From [szagoruyko/diracnets] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.111/diracnet18v2-1117-27601f6f.params.log)) |
| DiracNetV2-34 | 27.93 | 9.46 | 21,616,232 | 3,646.93M | From [szagoruyko/diracnets] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.111/diracnet34v2-0946-1faa6f12.params.log)) |
| DenseNet-121 | 25.11 | 7.80 | 7,978,856 | 2,872.13M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.3/densenet121-0780-49b72d04.params.log)) |
| DenseNet-161 | 22.40 | 6.18 | 28,681,000 | 7,793.16M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.3/densenet161-0618-52e30516.params.log)) |
| DenseNet-169 | 23.89 | 6.89 | 14,149,480 | 3,403.89M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.3/densenet169-0689-281ec06b.params.log)) |
| DenseNet-201 | 22.71 | 6.36 | 20,013,928 | 4,347.15M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.3/densenet201-0636-65b5d389.params.log)) |
| CondenseNet-74 (C=G=4) | 26.82 | 8.64 | 4,773,944 | 546.06M | From [ShichenLiu/CondenseNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.4/condensenet74_c4_g4-0864-cde68fa2.params.log)) |
| CondenseNet-74 (C=G=8) | 29.76 | 10.49 | 2,935,416 | 291.52M | From [ShichenLiu/CondenseNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.4/condensenet74_c8_g8-1049-4cf4a08e.params.log)) |
| PeleeNet | 31.71 | 11.25 | 2,802,248 | 514.87M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.141/peleenet-1125-38d4fb24.params.log)) |
| WRN-50-2 | 22.15 | 6.12 | 68,849,128 | 11,405.42M | From [szagoruyko/functional-zoo] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.113/wrn50_2-0612-f8013e68.params.log)) |
| DRN-C-26 | 25.68 | 7.89 | 21,126,584 | 16,993.90M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnc26-0789-ee56ffab.params.log)) |
| DRN-C-42 | 23.80 | 6.92 | 31,234,744 | 25,093.75M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnc42-0692-f89c26d6.params.log)) |
| DRN-C-58 | 22.35 | 6.27 | 40,542,008 | 32,489.94M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnc58-0627-44cbf15c.params.log)) |
| DRN-D-22 | 26.67 | 8.52 | 16,393,752 | 13,051.33M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd22-0852-08574752.params.log)) |
| DRN-D-38 | 24.51 | 7.36 | 26,501,912 | 21,151.19M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd38-0736-c7d53bc0.params.log)) |
| DRN-D-54 | 22.05 | 6.27 | 35,809,176 | 28,547.38M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd54-0627-87d44c87.params.log)) |
| DRN-D-105 | 21.31 | 5.81 | 54,801,304 | 43,442.43M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd105-0581-ab12d662.params.log)) |
| DPN-68 | 23.57 | 7.00 | 12,611,602 | 2,351.84M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.17/dpn68-0700-3114719d.params.log)) |
| DPN-98 | 20.23 | 5.28 | 61,570,728 | 11,716.51M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.17/dpn98-0528-fa5d6fca.params.log)) |
| DPN-131 | 20.03 | 5.22 | 79,254,504 | 16,076.15M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.17/dpn131-0522-35ac2f82.params.log)) |
| DarkNet Tiny | 40.31 | 17.46 | 1,042,104 | 500.85M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.69/darknet_tiny-1746-16501793.params.log)) |
| DarkNet Ref | 38.00 | 16.68 | 7,319,416 | 367.59M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.64/darknet_ref-1668-3011b4e1.params.log)) |
| DarkNet-53 | 21.44 | 5.56 | 41,609,928 | 7,133.86M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.150/darknet53-0556-e9486353.params.log)) |
| SqueezeNet v1.0 | 38.73 | 17.34 | 1,248,424 | 823.67M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.128/squeezenet_v1_0-1734-e6f8b0e8.params.log)) |
| SqueezeNet v1.1 | 39.09 | 17.39 | 1,235,496 | 352.02M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.88/squeezenet_v1_1-1739-d7a1483a.params.log)) |
| SqueezeResNet v1.1 | 39.83 | 17.84 | 1,235,496 | 352.02M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.70/squeezeresnet_v1_1-1784-26064b82.params.log)) |
| 1.0-SqNxt-23 | 45.78 | 21.55 | 724,056 | 287.28M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.138/sqnxt23_w1-2155-ae90c345.params.log)) |
| ShuffleNet x0.25 (g=1) | 62.00 | 36.77 | 209,746 | 12.35M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.134/shufflenet_g1_wd4-3677-ee58f368.params.log)) |
| ShuffleNet x0.25 (g=3) | 61.34 | 36.17 | 305,902 | 13.09M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.135/shufflenet_g3_wd4-3617-bd08e3ed.params.log)) |
| ShuffleNetV2 x0.5 | 40.61 | 18.30 | 1,366,792 | 43.31M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.90/shufflenetv2_wd2-1830-156953de.params.log)) |
| ShuffleNetV2b x0.5 | 40.98 | 18.56 | 1,366,792 | 43.31M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.112/shufflenetv2b_wd2-1856-d1143ea2.params.log)) |
| ShuffleNetV2c x0.5 | 39.87 | 18.11 | 1,366,792 | 43.31M | From [tensorpack/tensorpack] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.91/shufflenetv2c_wd2-1811-979ce7d9.params.log)) |
| ShuffleNetV2 x1.0 | 30.94 | 11.23 | 2,278,604 | 149.72M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.133/shufflenetv2_w1-1123-27435039.params.log)) |
| ShuffleNetV2c x1.0 | 30.74 | 11.38 | 2,279,760 | 150.62M | From [tensorpack/tensorpack] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.95/shufflenetv2c_w1-1138-646f3b78.params.log)) |
| ShuffleNetV2 x1.5 | 32.38 | 12.37 | 4,406,098 | 320.77M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.65/shufflenetv2_w3d2-1237-08c01388.params.log)) |
| ShuffleNetV2 x2.0 | 32.04 | 12.10 | 7,601,686 | 595.84M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.84/shufflenetv2_w2-1210-544b55d9.params.log)) |
| 108-MENet-8x1 (g=3) | 43.62 | 20.30 | 654,516 | 42.68M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.89/menet108_8x1_g3-2030-aa07f925.params.log)) |
| 128-MENet-8x1 (g=4) | 42.10 | 19.13 | 750,796 | 45.98M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.103/menet128_8x1_g4-1913-0c890a76.params.log)) |
| 228-MENet-12x1 (g=3) | 33.86 | 12.89 | 1,806,568 | 152.93M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.131/menet228_12x1_g3-1289-2dc2eec7.params.log)) |
| 256-MENet-12x1 (g=4) | 34.49 | 13.90 | 1,888,240 | 150.65M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet256_12x1_g4-1390-4502f223.params.log)) |
| 348-MENet-12x1 (g=3) | 31.17 | 11.41 | 3,368,128 | 312.00M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet348_12x1_g3-1141-ac69b246.params.log)) |
| 352-MENet-12x1 (g=8) | 34.70 | 13.75 | 2,272,872 | 157.35M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet352_12x1_g8-1375-85779b8a.params.log)) |
| 456-MENet-24x1 (g=3) | 29.57 | 10.43 | 5,304,784 | 567.90M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet456_24x1_g3-1043-6e777068.params.log)) |
| MobileNet x0.25 | 45.78 | 22.18 | 470,072 | 44.09M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.62/mobilenet_wd4-2218-3185cdd2.params.log)) |
| MobileNet x0.5 | 34.79 | 13.65 | 1,331,592 | 155.42M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.148/mobilenet_wd2-1365-59441ede.params.log)) |
| MobileNet x0.75 | 29.85 | 10.51 | 2,585,560 | 333.99M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.130/mobilenet_w3d4-1051-6361d4b4.params.log)) |
| MobileNet x1.0 | 26.72 | 8.71 | 4,231,976 | 579.80M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.149/mobilenet_w1-0871-63fb089c.params.log)) |
| FD-MobileNet x0.25 | 56.19 | 31.38 | 383,160 | 12.95M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.68/fdmobilenet_wd4-3138-2fe432fd.params.log)) |
| FD-MobileNet x0.5 | 42.62 | 19.69 | 993,928 | 41.84M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.83/fdmobilenet_wd2-1969-242b9fa8.params.log)) |
| FD-MobileNet x1.0 | 34.42 | 13.73 | 2,901,288 | 147.46M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.129/fdmobilenet_w1-1373-c81e1b43.params.log)) |
| MobileNetV2 x0.25 | 48.08 | 24.12 | 1,516,392 | 34.24M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.137/mobilenetv2_wd4-2412-d92b5b2d.params.log)) |
| MobileNetV2 x0.5 | 35.51 | 14.64 | 1,964,736 | 100.13M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.9/mobilenetv2_wd2-1464-02fe7ff2.params.log)) |
| MobileNetV2 x0.75 | 30.82 | 11.26 | 2,627,592 | 198.50M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.9/mobilenetv2_w3d4-1126-152672f5.params.log)) |
| MobileNetV2 x1.0 | 28.51 | 9.90 | 3,504,960 | 329.36M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.9/mobilenetv2_w1-0990-4e1a3878.params.log)) |
| IGCV3 x0.25 | 53.43 | 28.30 | 1,534,020 | 41.29M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.142/igcv3_wd4-2830-71abf6e0.params.log)) |
| IGCV3 x0.5 | 39.41 | 17.03 | 1,985,528 | 111.12M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.132/igcv3_wd2-1703-145b7089.params.log)) |
| IGCV3 x1.0 | 28.22 | 9.54 | 3,491,688 | 340.79M | From [homles11/IGCV3] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.126/igcv3_w1-0954-ae026c8c.params.log)) |
| MnasNet | 31.32 | 11.44 | 4,308,816 | 317.67M | From [zeusees/Mnasnet...Model] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.117/mnasnet-1144-c972fec0.params.log)) |
| DARTS | 27.23 | 8.97 | 4,718,752 | 539.86M | From [quark0/darts] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.118/darts-0897-aafd6452.params.log)) |
| Xception | 20.99 | 5.56 | 22,855,952 | 8,403.63M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.115/xception-0556-bd2c1684.params.log)) |
| InceptionV3 | 21.22 | 5.59 | 23,834,568 | 5,743.06M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.92/inceptionv3-0559-6c087967.params.log)) |
| InceptionV4 | 20.60 | 5.25 | 42,679,816 | 12,304.93M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.105/inceptionv4-0525-f7aa9536.params.log)) |
| InceptionResNetV2 | 19.96 | 4.94 | 55,843,464 | 13,188.64M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.107/inceptionresnetv2-0494-3328f7fa.params.log)) |
| PolyNet | 19.09 | 4.53 | 95,366,600 | 34,821.34M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.96/polynet-0453-74280314.params.log)) |
| NASNet-A 4@1056 | 25.37 | 7.95 | 5,289,978 | 584.90M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.97/nasnet_4a1056-0795-5c78908e.params.log)) |
| NASNet-A 6@4032 | 18.17 | 4.24 | 88,753,150 | 23,976.44M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.101/nasnet_6a4032-0424-73cca5fe.params.log)) |
| PNASNet-5-Large | 17.90 | 4.28 | 86,057,668 | 25,140.77M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.114/pnasnet5large-0428-998a548f.params.log)) |

### For PyTorch

| Model | Top1 | Top5 | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | ---: | --- |
| AlexNet | 43.48 | 20.93 | 61,100,840 | 714.83M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.108/alexnet-2093-6429d865.pth.log)) |
| VGG-11 | 30.98 | 11.37 | 132,863,336 | 7,615.87M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg11-1137-8a64fe7a.pth.log)) |
| VGG-13 | 30.07 | 10.75 | 133,047,848 | 11,317.65M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg13-1075-24178cab.pth.log)) |
| VGG-16 | 27.15 | 8.92 | 138,357,544 | 15,480.10M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg16-0892-10f44f68.pth.log)) |
| VGG-19 | 26.19 | 8.39 | 143,667,240 | 19,642.55M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg19-0839-d4e69a0d.pth.log)) |
| BN-VGG-11b | 29.63 | 10.19 | 132,868,840 | 7,630.72M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg11b-1019-98d7e914.pth.log)) |
| BN-VGG-13b | 28.41 | 9.63 | 133,053,736 | 11,342.14M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg13b-0963-cf9352f4.pth.log)) |
| BN-VGG-16b | 27.19 | 8.74 | 138,365,992 | 15,507.20M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg16b-0874-af4f2d0b.pth.log)) |
| BN-VGG-19b | 26.06 | 8.40 | 143,678,248 | 19,672.26M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg19b-0840-b6919f7f.pth.log)) |
| BN-Inception | 25.39 | 8.04 | 11,295,240 | 2,048.06M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.139/bninception-0804-99ff8708.pth.log)) |
| ResNet-10 | 37.46 | 15.85 | 5,418,792 | 894.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet10-1585-ef8a3ae3.pth.log)) |
| ResNet-12 | 36.18 | 14.80 | 5,492,776 | 1,126.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.30/resnet12-1480-c2263f73.pth.log)) |
| ResNet-14 | 33.17 | 12.71 | 5,788,200 | 1,357.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.40/resnet14-1271-568c392e.pth.log)) |
| ResNet-16 | 30.90 | 11.38 | 6,968,872 | 1,589.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.41/resnet16-1138-3a5aa7c0.pth.log)) |
| ResNet-18 x0.25 | 49.50 | 24.83 | 831,096 | 137.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.47/resnet18_wd4-2483-6ef2515c.pth.log)) |
| ResNet-18 x0.5 | 37.04 | 15.38 | 3,055,880 | 486.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.46/resnet18_wd2-1538-671466b5.pth.log)) |
| ResNet-18 x0.75 | 33.61 | 12.85 | 6,675,352 | 1,047.53M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.18/resnet18_w3d4-1285-94713e0e.pth.log)) |
| ResNet-18 | 28.53 | 9.90 | 11,689,512 | 1,820.41M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.151/resnet18-0990-1cf297b3.pth.log)) |
| ResNet-34 | 25.66 | 8.18 | 21,797,672 | 3,672.68M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet34-0818-6f947d40.pth.log)) |
| ResNet-50 | 22.96 | 6.58 | 25,557,032 | 3,877.95M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.147/resnet50-0658-828686d7.pth.log)) |
| ResNet-50b | 22.61 | 6.45 | 25,557,032 | 4,110.48M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.146/resnet50b-0645-a53df64c.pth.log)) |
| ResNet-101 | 21.90 | 6.22 | 44,549,160 | 7,597.95M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet101-0622-ab0cf005.pth.log)) |
| ResNet-101b | 20.88 | 5.61 | 44,549,160 | 7,830.48M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.145/resnet101b-0561-9fbf0696.pth.log)) |
| ResNet-152 | 21.01 | 5.50 | 60,192,808 | 11,321.85M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.144/resnet152-0550-800b2cb1.pth.log)) |
| ResNet-152b | 20.56 | 5.34 | 60,192,808 | 11,554.38M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.143/resnet152b-0534-e02a8bf7.pth.log)) |
| PreResNet-18 | 28.43 | 9.72 | 11,687,848 | 1,820.56M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.140/preresnet18-0972-5651bc2d.pth.log)) |
| PreResNet-34 | 26.23 | 8.41 | 21,796,008 | 3,672.83M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet34-0841-b4dd761f.pth.log)) |
| PreResNet-50 | 23.70 | 6.85 | 25,549,480 | 3,875.44M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet50-0685-d81a7aca.pth.log)) |
| PreResNet-50b | 23.33 | 6.87 | 25,549,480 | 4,107.97M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet50b-0687-65be98fb.pth.log)) |
| PreResNet-101 | 21.74 | 5.91 | 44,541,608 | 7,595.44M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet101-0591-4bacff79.pth.log)) |
| PreResNet-101b | 21.95 | 6.03 | 44,541,608 | 7,827.97M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet101b-0603-b1e37a09.pth.log)) |
| PreResNet-152 | 20.94 | 5.55 | 60,185,256 | 11,319.34M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.14/preresnet152-0555-c842a030.pth.log)) |
| PreResNet-152b | 21.34 | 5.91 | 60,185,256 | 11,551.87M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet152b-0591-2c91ab2c.pth.log)) |
| PreResNet-200b | 21.33 | 5.88 | 64,666,280 | 15,068.63M | From [tornadomeet/ResNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.45/preresnet200b-0588-f7104ff3.pth.log)) |
| ResNeXt-101 (32x4d) | 21.81 | 6.11 | 44,177,704 | 8,003.45M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.10/resnext101_32x4d-0611-cf962440.pth.log)) |
| ResNeXt-101 (64x4d) | 21.04 | 5.75 | 83,455,272 | 15,500.27M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.10/resnext101_64x4d-0575-651abd02.pth.log)) |
| SE-ResNet-50 | 22.47 | 6.40 | 28,088,024 | 3,880.49M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.11/seresnet50-0640-8820f2af.pth.log)) |
| SE-ResNet-101 | 21.88 | 5.89 | 49,326,872 | 7,602.76M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.11/seresnet101-0589-5e6e831b.pth.log)) |
| SE-ResNet-152 | 21.48 | 5.76 | 66,821,848 | 11,328.52M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.11/seresnet152-0576-814cf72e.pth.log)) |
| SE-ResNeXt-50 (32x4d) | 21.00 | 5.54 | 27,559,896 | 4,258.40M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.12/seresnext50_32x4d-0554-99e0e9aa.pth.log)) |
| SE-ResNeXt-101 (32x4d) | 19.96 | 5.05 | 48,955,416 | 8,008.26M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.12/seresnext101_32x4d-0505-0924f0a2.pth.log)) |
| SENet-154 | 18.62 | 4.61 | 115,088,984 | 20,745.78M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.13/senet154-0461-6512228c.pth.log)) |
| IBN-ResNet-50 | 22.76 | 6.41 | 25,557,032 | 4,110.48M | From [XingangPan/IBN-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.127/ibn_resnet50-0641-e48a1fe5.pth.log)) |
| IBN-ResNet-101 | 21.29 | 5.61 | 44,549,160 | 7,830.48M | From [XingangPan/IBN-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.127/ibn_resnet101-0561-5279c78a.pth.log)) |
| IBN(b)-ResNet-50 | 23.64 | 6.86 | 25,558,568 | 4,112.89M | From [XingangPan/IBN-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.127/ibnb_resnet50-0686-e138995e.pth.log)) |
| IBN-ResNeXt-101 (32x4d) | 20.88 | 5.42 | 44,177,704 | 8,003.45M | From [XingangPan/IBN-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.127/ibn_resnext101_32x4d-0542-b5233c66.pth.log)) |
| IBN-DenseNet-121 | 24.47 | 7.25 | 7,978,856 | 2,872.13M | From [XingangPan/IBN-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.127/ibn_densenet121-0725-b90b0615.pth.log)) |
| IBN-DenseNet-169 | 23.25 | 6.51 | 14,149,480 | 3,403.89M | From [XingangPan/IBN-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.127/ibn_densenet169-0651-96dd755e.pth.log)) |
| AirNet50-1x64d (r=2) | 21.84 | 5.90 | 27,425,864 | 4,772.11M | From [soeaver/AirNet-PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.120/airnet50_1x64d_r2-0590-3ec42212.pth.log)) |
| AirNet50-1x64d (r=16) | 22.11 | 6.19 | 25,714,952 | 4,399.97M | From [soeaver/AirNet-PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.120/airnet50_1x64d_r16-0619-090179e7.pth.log)) |
| AirNeXt50-32x4d (r=2) | 20.87 | 5.51 | 27,604,296 | 5,339.58M | From [soeaver/AirNet-PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.120/airnext50_32x4d_r2-0551-c68156e5.pth.log)) |
| BAM-ResNet-50 | 23.14 | 6.58 | 25,915,099 | 4,196.09M | From [Jongchan/attention-module] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.124/bam_resnet50-0658-96a37c82.pth.log)) |
| CBAM-ResNet-50 | 22.38 | 6.05 | 28,089,624 | 4,116.97M | From [Jongchan/attention-module] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.125/cbam_resnet50-0605-a1172fe6.pth.log)) |
| PyramidNet-101 (a=360) | 21.98 | 6.20 | 42,455,070 | 8,743.54M | From [dyhan0920/Pyramid...PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.104/pyramidnet101_a360-0620-3a24427b.pth.log)) |
| DiracNetV2-18 | 31.47 | 11.70 | 11,511,784 | 1,796.62M | From [szagoruyko/diracnets] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.111/diracnet18v2-1170-e0673770.pth.log)) |
| DiracNetV2-34 | 28.75 | 9.93 | 21,616,232 | 3,646.93M | From [szagoruyko/diracnets] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.111/diracnet34v2-0993-a6a661c0.pth.log)) |
| DenseNet-121 | 25.57 | 8.03 | 7,978,856 | 2,872.13M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.3/densenet121-0803-f994107a.pth.log)) |
| DenseNet-161 | 22.86 | 6.44 | 28,681,000 | 7,793.16M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.3/densenet161-0644-c0fb22c8.pth.log)) |
| DenseNet-169 | 24.40 | 7.19 | 14,149,480 | 3,403.89M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.3/densenet169-0719-27139105.pth.log)) |
| DenseNet-201 | 23.10 | 6.63 | 20,013,928 | 4,347.15M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.3/densenet201-0663-71ece4ad.pth.log)) |
| CondenseNet-74 (C=G=4) | 26.25 | 8.28 | 4,773,944 | 546.06M | From [ShichenLiu/CondenseNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.4/condensenet74_c4_g4-0828-5ba55049.pth.log)) |
| CondenseNet-74 (C=G=8) | 28.93 | 10.06 | 2,935,416 | 291.52M | From [ShichenLiu/CondenseNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.4/condensenet74_c8_g8-1006-3574d874.pth.log)) |
| PeleeNet | 31.81 | 11.51 | 2,802,248 | 514.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.141/peleenet-1151-9c47b802.pth.log)) |
| WRN-50-2 | 22.53 | 6.41 | 68,849,128 | 11,405.42M | From [szagoruyko/functional-zoo] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.113/wrn50_2-0641-83897ab9.pth.log)) |
| DRN-C-26 | 24.86 | 7.55 | 21,126,584 | 16,993.90M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnc26-0755-35405bd5.pth.log)) |
| DRN-C-42 | 22.94 | 6.57 | 31,234,744 | 25,093.75M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnc42-0657-7c99c460.pth.log)) |
| DRN-C-58 | 21.73 | 6.01 | 40,542,008 | 32,489.94M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnc58-0601-70ec1f56.pth.log)) |
| DRN-D-22 | 25.80 | 8.23 | 16,393,752 | 13,051.33M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd22-0823-5c2c6a0c.pth.log)) |
| DRN-D-38 | 23.79 | 6.95 | 26,501,912 | 21,151.19M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd38-0695-4630f0fb.pth.log)) |
| DRN-D-54 | 21.22 | 5.86 | 35,809,176 | 28,547.38M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd54-0586-bfdc1f88.pth.log)) |
| DRN-D-105 | 20.62 | 5.48 | 54,801,304 | 43,442.43M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd105-0548-a643f4dc.pth.log)) |
| DPN-68 | 24.17 | 7.27 | 12,611,602 | 2,351.84M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.17/dpn68-0727-43849233.pth.log)) |
| DPN-98 | 20.81 | 5.53 | 61,570,728 | 11,716.51M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.17/dpn98-0553-52c55969.pth.log)) |
| DPN-131 | 20.54 | 5.48 | 79,254,504 | 16,076.15M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.17/dpn131-0548-0c53e5b3.pth.log)) |
| DarkNet Tiny | 40.74 | 17.84 | 1,042,104 | 500.85M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.69/darknet_tiny-1784-4561e1ad.pth.log)) |
| DarkNet Ref | 38.58 | 17.18 | 7,319,416 | 367.59M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.64/darknet_ref-1718-034595b4.pth.log)) |
| DarkNet-53 | 21.75 | 5.64 | 41,609,928 | 7,133.86M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.150/darknet53-0564-b36bef6b.pth.log)) |
| SqueezeNet v1.0 | 39.29 | 17.66 | 1,248,424 | 823.67M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.128/squeezenet_v1_0-1766-afdbcf1a.pth.log)) |
| SqueezeNet v1.1 | 39.31 | 17.72 | 1,235,496 | 352.02M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.88/squeezenet_v1_1-1772-25b77bc3.pth.log)) |
| SqueezeResNet v1.1 | 40.09 | 18.21 | 1,235,496 | 352.02M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.70/squeezeresnet_v1_1-1821-c27ed88f.pth.log)) |
| 1.0-SqNxt-23 | 46.33 | 21.96 | 724,056 | 287.28M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.138/sqnxt23_w1-2196-7c35871b.pth.log)) |
| ShuffleNet x0.25 (g=1) | 62.44 | 37.29 | 209,746 | 12.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.134/shufflenet_g1_wd4-3729-47dbd0f2.pth.log)) |
| ShuffleNet x0.25 (g=3) | 61.74 | 36.53 | 305,902 | 13.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.135/shufflenet_g3_wd4-3653-6abdd65e.pth.log)) |
| ShuffleNetV2 x0.5 | 40.99 | 18.65 | 1,366,792 | 43.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.90/shufflenetv2_wd2-1865-9c22238b.pth.log)) |
| ShuffleNetV2b x0.5 | 41.41 | 19.07 | 1,366,792 | 43.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.112/shufflenetv2b_wd2-1907-cf4fe43c.pth.log)) |
| ShuffleNetV2c x0.5 | 40.31 | 18.51 | 1,366,792 | 43.31M | From [tensorpack/tensorpack] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.94/shufflenetv2c_wd2-1851-e1d36c5d.pth.log)) |
| ShuffleNetV2 x1.0 | 31.44 | 11.63 | 2,278,604 | 149.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.133/shufflenetv2_w1-1163-c71dfb7a.pth.log)) |
| ShuffleNetV2c x1.0 | 30.98 | 11.61 | 2,279,760 | 150.62M | From [tensorpack/tensorpack] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.95/shufflenetv2c_w1-1161-8cdbbcc1.pth.log)) |
| ShuffleNetV2 x1.5 | 32.82 | 12.69 | 4,406,098 | 320.77M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.65/shufflenetv2_w3d2-1269-536ad5b1.pth.log)) |
| ShuffleNetV2 x2.0 | 32.45 | 12.49 | 7,601,686 | 595.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.84/shufflenetv2_w2-1249-b9f9e84c.pth.log)) |
| 108-MENet-8x1 (g=3) | 43.94 | 20.76 | 654,516 | 42.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.89/menet108_8x1_g3-2076-6acc82e4.pth.log)) |
| 128-MENet-8x1 (g=4) | 42.43 | 19.59 | 750,796 | 45.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.103/menet128_8x1_g4-1959-48fa80fc.pth.log)) |
| 228-MENet-12x1 (g=3) | 34.11 | 13.16 | 1,806,568 | 152.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.131/menet228_12x1_g3-1316-5b670c42.pth.log)) |
| 256-MENet-12x1 (g=4) | 33.41 | 13.26 | 1,888,240 | 150.65M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet256_12x1_g4-1326-e5d35476.pth.log)) |
| 348-MENet-12x1 (g=3) | 30.10 | 10.92 | 3,368,128 | 312.00M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet348_12x1_g3-1092-66be1a18.pth.log)) |
| 352-MENet-12x1 (g=8) | 33.31 | 13.08 | 2,272,872 | 157.35M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet352_12x1_g8-1308-e91ec72c.pth.log)) |
| 456-MENet-24x1 (g=3) | 28.40 | 9.93 | 5,304,784 | 567.90M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.6/menet456_24x1_g3-0993-cb9fd376.pth.log)) |
| MobileNet x0.25 | 46.26 | 22.49 | 470,072 | 44.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.62/mobilenet_wd4-2249-1ad5e8fe.pth.log)) |
| MobileNet x0.5 | 35.11 | 13.95 | 1,331,592 | 155.42M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.148/mobilenet_wd2-1395-364b4ca6.pth.log)) |
| MobileNet x0.75 | 30.14 | 10.76 | 2,585,560 | 333.99M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.130/mobilenet_w3d4-1076-d801bcae.pth.log)) |
| MobileNet x1.0 | 27.08 | 9.00 | 4,231,976 | 579.80M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.149/mobilenet_w1-0900-45d81a1b.pth.log)) |
| FD-MobileNet x0.25 | 55.77 | 31.32 | 383,160 | 12.95M | From [clavichord93/FD-MobileNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.8/fdmobilenet_wd4-3132-0b242eff.pth.log)) |
| FD-MobileNet x0.5 | 43.13 | 20.15 | 993,928 | 41.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.83/fdmobilenet_wd2-2015-414dbeed.pth.log)) |
| FD-MobileNet x1.0 | 34.70 | 14.05 | 2,901,288 | 147.46M | From [clavichord93/FD-MobileNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.8/fdmobilenet_w1-1405-a6538879.pth.log)) |
| MobileNetV2 x0.25 | 48.34 | 24.51 | 1,516,392 | 34.24M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.137/mobilenetv2_wd4-2451-05e1e3a2.pth.log)) |
| MobileNetV2 x0.5 | 36.54 | 15.19 | 1,964,736 | 100.13M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.9/mobilenetv2_wd2-1519-d0937a23.pth.log)) |
| MobileNetV2 x0.75 | 31.89 | 11.76 | 2,627,592 | 198.50M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.9/mobilenetv2_w3d4-1176-1b966ff4.pth.log)) |
| MobileNetV2 x1.0 | 29.31 | 10.39 | 3,504,960 | 329.36M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.9/mobilenetv2_w1-1039-7532eb72.pth.log)) |
| IGCV3 x0.25 | 53.70 | 28.71 | 1,534,020 | 41.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.142/igcv3_wd4-2871-c9f28301.pth.log)) |
| IGCV3 x0.5 | 39.75 | 17.32 | 1,985,528 | 111.12M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.132/igcv3_wd2-1732-8c504f44.pth.log)) |
| IGCV3 x1.0 | 28.40 | 9.84 | 3,491,688 | 340.79M | From [homles11/IGCV3] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.126/igcv3_w1-0984-5f099cc8.pth.log)) |
| MnasNet | 31.58 | 11.74 | 4,308,816 | 317.67M | From [zeusees/Mnasnet...Model] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.117/mnasnet-1174-e8ec017c.pth.log)) |
| DARTS | 26.70 | 8.74 | 4,718,752 | 539.86M | From [quark0/darts] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.118/darts-0874-74f0c7b6.pth.log)) |
| Xception | 20.97 | 5.49 | 22,855,952 | 8,403.63M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.115/xception-0549-e4f0232c.pth.log)) |
| InceptionV3 | 21.12 | 5.65 | 23,834,568 | 5,743.06M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.92/inceptionv3-0565-cf406180.pth.log)) |
| InceptionV4 | 20.64 | 5.29 | 42,679,816 | 12,304.93M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.105/inceptionv4-0529-5cb7b4e4.pth.log)) |
| InceptionResNetV2 | 19.93 | 4.90 | 55,843,464 | 13,188.64M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.107/inceptionresnetv2-0490-1d1b4d18.pth.log)) |
| PolyNet | 19.10 | 4.52 | 95,366,600 | 34,821.34M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.96/polynet-0452-6a1b295d.pth.log)) |
| NASNet-A 4@1056 | 25.68 | 8.16 | 5,289,978 | 584.90M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.97/nasnet_4a1056-0816-d21bbaf5.pth.log)) |
| NASNet-A 6@4032 | 18.14 | 4.21 | 88,753,150 | 23,976.44M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.101/nasnet_6a4032-0421-f354d28f.pth.log)) |
| PNASNet-5-Large | 17.88 | 4.28 | 86,057,668 | 25,140.77M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.114/pnasnet5large-0428-65de46eb.pth.log)) |

### For Chainer

| Model | Top1 | Top5 | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | ---: | --- |
| AlexNet | 44.08 | 21.32 | 61,100,840 | 714.83M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.108/alexnet-2132-cea565f1.npz.log)) |
| VGG-11 | 31.89 | 11.79 | 132,863,336 | 7,615.87M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg11-1179-3cc057e6.npz.log)) |
| VGG-13 | 31.06 | 11.16 | 133,047,848 | 11,317.65M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg13-1116-e835ca5a.npz.log)) |
| VGG-16 | 26.75 | 8.70 | 138,357,544 | 15,480.10M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg16-0870-8741ff5c.npz.log)) |
| VGG-19 | 25.86 | 8.23 | 143,667,240 | 19,642.55M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.109/vgg19-0823-18980884.npz.log)) |
| BN-VGG-11b | 30.37 | 10.60 | 132,868,840 | 7,630.72M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg11b-1060-8964402b.npz.log)) |
| BN-VGG-13b | 29.45 | 10.19 | 133,053,736 | 11,342.14M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg13b-1019-0121b0a4.npz.log)) |
| BN-VGG-16b | 26.89 | 8.63 | 138,365,992 | 15,507.20M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg16b-0863-cbaa2105.npz.log)) |
| BN-VGG-19b | 25.65 | 8.16 | 143,678,248 | 19,672.26M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.110/bn_vgg19b-0816-dc5e37a5.npz.log)) |
| BN-Inception | 25.08 | 7.78 | 11,295,240 | 2,048.06M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.139/bninception-0778-99f685c2.npz.log)) |
| ResNet-10 | 37.12 | 15.49 | 5,418,792 | 894.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet10-1549-b31f1135.npz.log)) |
| ResNet-12 | 35.86 | 14.48 | 5,492,776 | 1,126.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.30/resnet12-1448-11acb729.npz.log)) |
| ResNet-14 | 32.84 | 12.42 | 5,788,200 | 1,357.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.40/resnet14-1242-4e65746b.npz.log)) |
| ResNet-16 | 30.66 | 11.07 | 6,968,872 | 1,589.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.41/resnet16-1107-b1d7fb7d.npz.log)) |
| ResNet-18 x0.25 | 49.08 | 24.48 | 831,096 | 137.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.47/resnet18_wd4-2448-58c4a007.npz.log)) |
| ResNet-18 x0.5 | 36.55 | 14.99 | 3,055,880 | 486.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.46/resnet18_wd2-1499-542ed773.npz.log)) |
| ResNet-18 x0.75 | 33.27 | 12.56 | 6,675,352 | 1,047.53M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet18_w3d4-1256-ce2011df.npz.log)) |
| ResNet-18 | 28.22 | 9.67 | 11,689,512 | 1,820.41M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.151/resnet18-0967-53f35c84.npz.log)) |
| ResNet-34 | 25.35 | 7.95 | 21,797,672 | 3,672.68M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet34-0795-0b392267.npz.log)) |
| ResNet-50 | 22.61 | 6.41 | 25,557,032 | 3,877.95M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.147/resnet50-0641-ca0cd7a1.npz.log)) |
| ResNet-50b | 22.34 | 6.18 | 25,557,032 | 4,110.48M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.146/resnet50b-0618-42fffef9.npz.log)) |
| ResNet-101 | 21.65 | 6.01 | 44,549,160 | 7,597.95M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.22/resnet101-0601-d8cddbea.npz.log)) |
| ResNet-101b | 20.79 | 5.40 | 44,549,160 | 7,830.48M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.145/resnet101b-0540-af300066.npz.log)) |
| ResNet-152 | 20.74 | 5.35 | 60,192,808 | 11,321.85M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.144/resnet152-0535-64c1daa7.npz.log)) |
| ResNet-152b | 20.29 | 5.27 | 60,192,808 | 11,554.38M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.143/resnet152b-0527-6efec251.npz.log)) |
| PreResNet-18 | 28.17 | 9.54 | 11,687,848 | 1,820.56M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.140/preresnet18-0954-21e4811a.npz.log)) |
| PreResNet-34 | 25.89 | 8.12 | 21,796,008 | 3,672.83M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet34-0812-829f5a23.npz.log)) |
| PreResNet-50 | 23.36 | 6.69 | 25,549,480 | 3,875.44M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet50-0669-40bd5e93.npz.log)) |
| PreResNet-50b | 23.08 | 6.67 | 25,549,480 | 4,107.97M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet50b-0667-b7d221ef.npz.log)) |
| PreResNet-101 | 21.45 | 5.75 | 44,541,608 | 7,595.44M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet101-0575-f6f6789a.npz.log)) |
| PreResNet-101b | 21.61 | 5.87 | 44,541,608 | 7,827.97M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet101b-0587-4211c5ab.npz.log)) |
| PreResNet-152 | 20.73 | 5.30 | 60,185,256 | 11,319.34M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet152-0530-021d99dc.npz.log)) |
| PreResNet-152b | 20.88 | 5.66 | 60,185,256 | 11,551.87M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.23/preresnet152b-0566-fdd337e7.npz.log)) |
| PreResNet-200b | 21.03 | 5.60 | 64,666,280 | 15,068.63M | From [tornadomeet/ResNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.45/preresnet200b-0560-f79bd952.npz.log)) |
| ResNeXt-101 (32x4d) | 21.11 | 5.69 | 44,177,704 | 8,003.45M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.26/resnext101_32x4d-0569-c6d1c30d.npz.log)) |
| ResNeXt-101 (64x4d) | 20.57 | 5.43 | 83,455,272 | 15,500.27M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.26/resnext101_64x4d-0543-dd8b7d96.npz.log)) |
| SE-ResNet-50 | 22.53 | 6.41 | 28,088,024 | 3,880.49M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.24/seresnet50-0641-f3d68cfc.npz.log)) |
| SE-ResNet-101 | 21.90 | 5.88 | 49,326,872 | 7,602.76M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.24/seresnet101-0588-e45a9f8f.npz.log)) |
| SE-ResNet-152 | 21.46 | 5.77 | 66,821,848 | 11,328.52M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.24/seresnet152-0577-a089ba52.npz.log)) |
| SE-ResNeXt-50 (32x4d) | 21.04 | 5.58 | 27,559,896 | 4,258.40M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.27/seresnext50_32x4d-0558-5c435c1b.npz.log)) |
| SE-ResNeXt-101 (32x4d) | 19.99 | 5.01 | 48,955,416 | 8,008.26M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.27/seresnext101_32x4d-0501-98ea6fc4.npz.log)) |
| SENet-154 | 18.79 | 4.63 | 115,088,984 | 20,745.78M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.28/senet154-0463-381d2494.npz.log)) |
| AirNet50-1x64d (r=2) | 22.46 | 6.20 | 27,425,864 | 4,772.11M | From [soeaver/AirNet-PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.120/airnet50_1x64d_r2-0620-b6a9359d.npz.log)) |
| AirNet50-1x64d (r=16) | 22.89 | 6.50 | 25,714,952 | 4,399.97M | From [soeaver/AirNet-PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.120/airnet50_1x64d_r16-0650-95da530f.npz.log)) |
| AirNeXt50-32x4d (r=2) | 21.50 | 5.73 | 27,604,296 | 5,339.58M | From [soeaver/AirNet-PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.120/airnext50_32x4d_r2-0573-160860f7.npz.log)) |
| BAM-ResNet-50 | 23.71 | 6.97 | 25,915,099 | 4,196.09M | From [Jongchan/attention-module] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.124/bam_resnet50-0697-a8c65533.npz.log)) |
| CBAM-ResNet-50 | 22.99 | 6.40 | 28,089,624 | 4,116.97M | From [Jongchan/attention-module] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.125/cbam_resnet50-0640-b2314d97.npz.log)) |
| PyramidNet-101 (a=360) | 22.66 | 6.49 | 42,455,070 | 8,743.54M | From [dyhan0920/Pyramid...PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.104/pyramidnet101_a360-0649-b68c786b.npz.log)) |
| DiracNetV2-18 | 30.60 | 11.13 | 11,511,784 | 1,796.62M | From [szagoruyko/diracnets] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.111/diracnet18v2-1113-b85b43d1.npz.log)) |
| DiracNetV2-34 | 27.90 | 9.48 | 21,616,232 | 3,646.93M | From [szagoruyko/diracnets] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.111/diracnet34v2-0948-0245163a.npz.log)) |
| DenseNet-121 | 25.04 | 7.79 | 7,978,856 | 2,872.13M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.29/densenet121-0779-06d5ebbf.npz.log)) |
| DenseNet-161 | 22.36 | 6.20 | 28,681,000 | 7,793.16M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.29/densenet161-0620-6d05f3b9.npz.log)) |
| DenseNet-169 | 23.85 | 6.86 | 14,149,480 | 3,403.89M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.29/densenet169-0686-1978656b.npz.log)) |
| DenseNet-201 | 22.64 | 6.29 | 20,013,928 | 4,347.15M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.29/densenet201-0629-77702939.npz.log)) |
| CondenseNet-74 (C=G=4) | 26.81 | 8.61 | 4,773,944 | 546.06M | From [ShichenLiu/CondenseNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.36/condensenet74_c4_g4-0861-ef6077ec.npz.log)) |
| CondenseNet-74 (C=G=8) | 29.74 | 10.43 | 2,935,416 | 291.52M | From [ShichenLiu/CondenseNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.36/condensenet74_c8_g8-1043-277fbfb8.npz.log)) |
| PeleeNet | 31.61 | 11.27 | 2,802,248 | 514.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.141/peleenet-1127-ef057fc9.npz.log)) |
| WRN-50-2 | 22.06 | 6.13 | 68,849,128 | 11,405.42M | From [szagoruyko/functional-zoo] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.113/wrn50_2-0613-d0cd9171.npz.log)) |
| DRN-C-26 | 25.68 | 7.88 | 21,126,584 | 16,993.90M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnc26-0788-762c34c1.npz.log)) |
| DRN-C-42 | 23.72 | 6.93 | 31,234,744 | 25,093.75M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnc42-0693-ec938cc4.npz.log)) |
| DRN-C-58 | 22.35 | 6.29 | 40,542,008 | 32,489.94M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnc58-0629-063ef199.npz.log)) |
| DRN-D-22 | 26.65 | 8.50 | 16,393,752 | 13,051.33M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd22-0850-b25d4757.npz.log)) |
| DRN-D-38 | 24.53 | 7.36 | 26,501,912 | 21,151.19M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd38-0736-153481d6.npz.log)) |
| DRN-D-54 | 22.08 | 6.23 | 35,809,176 | 28,547.38M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd54-0623-31e8eeb8.npz.log)) |
| DRN-D-105 | 21.32 | 5.84 | 54,801,304 | 43,442.43M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.116/drnd105-0584-c0d7657b.npz.log)) |
| DPN-68 | 23.61 | 7.01 | 12,611,602 | 2,351.84M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.34/dpn68-0701-ad8cd4ec.npz.log)) |
| DPN-98 | 20.80 | 5.53 | 61,570,728 | 11,716.51M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.34/dpn98-0553-9cd57335.npz.log)) |
| DPN-131 | 20.04 | 5.23 | 79,254,504 | 16,076.15M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.34/dpn131-0523-e3721599.npz.log)) |
| DarkNet Tiny | 40.33 | 17.46 | 1,042,104 | 500.85M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.69/darknet_tiny-1746-b04fa463.npz.log)) |
| DarkNet Ref | 38.09 | 16.71 | 7,319,416 | 367.59M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.64/darknet_ref-1671-b2d5721f.npz.log)) |
| DarkNet-53 | 21.41 | 5.56 | 41,609,928 | 7,133.86M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.150/darknet53-0556-42c57951.npz.log)) |
| SqueezeNet v1.0 | 38.76 | 17.38 | 1,248,424 | 823.67M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.128/squeezenet_v1_0-1738-4c55a6a5.npz.log)) |
| SqueezeNet v1.1 | 39.13 | 17.40 | 1,235,496 | 352.02M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.88/squeezenet_v1_1-1740-b236c204.npz.log)) |
| SqueezeResNet v1.1 | 39.85 | 17.87 | 1,235,496 | 352.02M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.70/squeezeresnet_v1_1-1787-f40e6051.npz.log)) |
| 1.0-SqNxt-23 | 46.55 | 22.13 | 724,056 | 287.28M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.138/sqnxt23_w1-2213-e1404b06.npz.log)) |
| ShuffleNet x0.25 (g=1) | 62.04 | 36.81 | 209,746 | 12.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.134/shufflenet_g1_wd4-3681-15d3e787.npz.log)) |
| ShuffleNet x0.25 (g=3) | 61.30 | 36.16 | 305,902 | 13.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.135/shufflenet_g3_wd4-3616-064f7f7f.npz.log)) |
| ShuffleNetV2 x0.5 | 43.45 | 20.73 | 1,366,792 | 43.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.90/shufflenetv2_wd2-2073-c5e5a23c.npz.log)) |
| ShuffleNetV2b x0.5 | 40.95 | 18.56 | 1,366,792 | 43.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.112/shufflenetv2b_wd2-1856-4d6e16de.npz.log)) |
| ShuffleNetV2c x0.5 | 39.82 | 18.14 | 1,366,792 | 43.31M | From [tensorpack/tensorpack] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.94/shufflenetv2c_wd2-1814-20fc1e3c.npz.log)) |
| ShuffleNetV2 x1.0 | 33.39 | 12.98 | 2,278,604 | 149.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.133/shufflenetv2_w1-1298-3830a2da.npz.log)) |
| ShuffleNetV2c x1.0 | 30.74 | 11.37 | 2,279,760 | 150.62M | From [tensorpack/tensorpack] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.95/shufflenetv2c_w1-1137-2f59108a.npz.log)) |
| ShuffleNetV2 x1.5 | 33.96 | 13.37 | 4,406,098 | 320.77M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.65/shufflenetv2_w3d2-1337-66c1d6ed.npz.log)) |
| ShuffleNetV2 x2.0 | 33.21 | 13.03 | 7,601,686 | 595.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.84/shufflenetv2_w2-1303-349e42b5.npz.log)) |
| 108-MENet-8x1 (g=3) | 43.67 | 20.42 | 654,516 | 42.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.89/menet108_8x1_g3-2042-9e3ff283.npz.log)) |
| 128-MENet-8x1 (g=4) | 42.07 | 19.19 | 750,796 | 45.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.103/menet128_8x1_g4-1919-f6fd56fa.npz.log)) |
| 228-MENet-12x1 (g=3) | 33.86 | 13.01 | 1,806,568 | 152.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.131/menet228_12x1_g3-1301-39c25ca3.npz.log)) |
| 256-MENet-12x1 (g=4) | 34.44 | 13.91 | 1,888,240 | 150.65M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.33/menet256_12x1_g4-1391-ee68bd6f.npz.log)) |
| 348-MENet-12x1 (g=3) | 31.14 | 11.40 | 3,368,128 | 312.00M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.33/menet348_12x1_g3-1140-49feaea7.npz.log)) |
| 352-MENet-12x1 (g=8) | 34.62 | 13.68 | 2,272,872 | 157.35M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.33/menet352_12x1_g8-1368-2d523fac.npz.log)) |
| 456-MENet-24x1 (g=3) | 29.55 | 10.39 | 5,304,784 | 567.90M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.33/menet456_24x1_g3-1039-f68c36a2.npz.log)) |
| MobileNet x0.25 | 45.85 | 22.16 | 470,072 | 44.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.62/mobilenet_wd4-2216-09c50ab8.npz.log)) |
| MobileNet x0.5 | 34.83 | 13.66 | 1,331,592 | 155.42M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.148/mobilenet_wd2-1366-dca3a965.npz.log)) |
| MobileNet x0.75 | 29.86 | 10.53 | 2,585,560 | 333.99M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.130/mobilenet_w3d4-1053-d7ec3192.npz.log)) |
| MobileNet x1.0 | 26.75 | 8.71 | 4,231,976 | 579.80M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.149/mobilenet_w1-0871-51b7833b.npz.log)) |
| FD-MobileNet x0.25 | 56.11 | 31.45 | 383,160 | 12.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.68/fdmobilenet_wd4-3145-6718fb07.npz.log)) |
| FD-MobileNet x0.5 | 42.68 | 19.76 | 993,928 | 41.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.83/fdmobilenet_wd2-1976-6299d442.npz.log)) |
| FD-MobileNet x1.0 | 34.44 | 13.74 | 2,901,288 | 147.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.129/fdmobilenet_w1-1374-99c7854b.npz.log)) |
| MobileNetV2 x0.25 | 48.10 | 24.11 | 1,516,392 | 34.24M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.137/mobilenetv2_wd4-2411-9fc398d3.npz.log)) |
| MobileNetV2 x0.5 | 35.96 | 14.98 | 1,964,736 | 100.13M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.31/mobilenetv2_wd2-1498-4b767a98.npz.log)) |
| MobileNetV2 x0.75 | 31.28 | 11.48 | 2,627,592 | 198.50M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.31/mobilenetv2_w3d4-1148-a6f852ea.npz.log)) |
| MobileNetV2 x1.0 | 28.87 | 10.05 | 3,504,960 | 329.36M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.31/mobilenetv2_w1-1005-3b6d1764.npz.log)) |
| IGCV3 x0.25 | 53.36 | 28.28 | 1,534,020 | 41.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.142/igcv3_wd4-2828-25942192.npz.log)) |
| IGCV3 x0.5 | 39.36 | 17.04 | 1,985,528 | 111.12M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.132/igcv3_wd2-1704-86246558.npz.log)) |
| IGCV3 x1.0 | 28.20 | 9.55 | 3,491,688 | 340.79M | From [homles11/IGCV3] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.126/igcv3_w1-0955-1c00ac33.npz.log)) |
| MnasNet | 31.27 | 11.44 | 4,308,816 | 317.67M | From [zeusees/Mnasnet...Model] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.117/mnasnet-1144-688e523d.npz.log)) |
| DARTS | 27.29 | 8.97 | 4,718,752 | 539.86M | From [quark0/darts] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.118/darts-0897-8986fe64.npz.log)) |
| Xception | 21.04 | 5.47 | 22,855,952 | 8,403.63M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.115/xception-0547-7a5be958.npz.log)) |
| InceptionV3 | 21.11 | 5.61 | 23,834,568 | 5,743.06M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.92/inceptionv3-0561-4ddea4df.npz.log)) |
| InceptionV4 | 20.62 | 5.26 | 42,679,816 | 12,304.93M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.105/inceptionv4-0526-02e53701.npz.log)) |
| InceptionResNetV2 | 19.93 | 4.92 | 55,843,464 | 13,188.64M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.107/inceptionresnetv2-0492-3d3de82b.npz.log)) |
| PolyNet | 19.08 | 4.50 | 95,366,600 | 34,821.34M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.96/polynet-0450-6dc7028b.npz.log)) |
| NASNet-A 4@1056 | 25.36 | 7.96 | 5,289,978 | 584.90M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.97/nasnet_4a1056-0796-f09950c0.npz.log)) |
| NASNet-A 6@4032 | 18.17 | 4.22 | 88,753,150 | 23,976.44M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.101/nasnet_6a4032-0422-d49d4663.npz.log)) |
| PNASNet-5-Large | 17.90 | 4.26 | 86,057,668 | 25,140.77M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.114/pnasnet5large-0426-3c2755dc.npz.log)) |

### For Keras

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
| ResNet-18 | 28.20 | 9.67 | 11,689,512 | 1,820.41M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.151/resnet18-0967-4a0a9110.h5.log)) |
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
| 1.0-SqNxt-23 | 45.97 | 21.67 | 724,056 | 287.28M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.138/sqnxt23_w1-2167-81f731e5.h5.log)) |
| ShuffleNet x0.25 (g=1) | 62.00 | 36.76 | 209,746 | 12.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.134/shufflenet_g1_wd4-3676-cb39b773.h5.log)) |
| ShuffleNet x0.25 (g=3) | 61.32 | 36.15 | 305,902 | 13.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.135/shufflenet_g3_wd4-3615-21150468.h5.log)) |
| ShuffleNetV2 x0.5 | 40.76 | 18.40 | 1,366,792 | 43.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.90/shufflenetv2_wd2-1840-9b4b0964.h5.log)) |
| ShuffleNetV2 x1.0 | 31.02 | 11.33 | 2,278,604 | 149.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.133/shufflenetv2_w1-1133-bcba973e.h5.log)) |
| ShuffleNetV2 x1.5 | 32.46 | 12.47 | 4,406,098 | 320.77M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.65/shufflenetv2_w3d2-1247-f7f813b4.h5.log)) |
| ShuffleNetV2 x2.0 | 31.91 | 12.23 | 7,601,686 | 595.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.84/shufflenetv2_w2-1223-63291468.h5.log)) |
| 108-MENet-8x1 (g=3) | 43.61 | 20.31 | 654,516 | 42.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.89/menet108_8x1_g3-2031-a4d43433.h5.log)) |
| 128-MENet-8x1 (g=4) | 42.08 | 19.14 | 750,796 | 45.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.103/menet128_8x1_g4-1914-5bb8f228.h5.log)) |
| 228-MENet-12x1 (g=3) | 33.85 | 12.88 | 1,806,568 | 152.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.131/menet228_12x1_g3-1288-c2eeac24.h5.log)) |
| 256-MENet-12x1 (g=4) | 34.48 | 13.91 | 1,888,240 | 150.65M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.58/menet256_12x1_g4-1391-a63a606a.h5.log)) |
| 348-MENet-12x1 (g=3) | 31.17 | 11.42 | 3,368,128 | 312.00M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.58/menet348_12x1_g3-1142-0715c866.h5.log)) |
| 352-MENet-12x1 (g=8) | 34.69 | 13.75 | 2,272,872 | 157.35M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.58/menet352_12x1_g8-1375-9007c933.h5.log)) |
| 456-MENet-24x1 (g=3) | 29.55 | 10.44 | 5,304,784 | 567.90M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.58/menet456_24x1_g3-1044-c090af59.h5.log)) |
| MobileNet x0.25 | 45.80 | 22.17 | 470,072 | 44.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.62/mobilenet_wd4-2217-fb7abda8.h5.log)) |
| MobileNet x0.5 | 34.80 | 13.66 | 1,331,592 | 155.42M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.148/mobilenet_wd2-1366-e823a654.h5.log)) |
| MobileNet x0.75 | 29.85 | 10.51 | 2,585,560 | 333.99M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.130/mobilenet_w3d4-1051-d200ad45.h5.log)) |
| MobileNet x1.0 | 26.71 | 8.71 | 4,231,976 | 579.80M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.149/mobilenet_w1-0871-cccedf06.h5.log)) |
| FD-MobileNet x0.25 | 56.17 | 31.37 | 383,160 | 12.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.68/fdmobilenet_wd4-3137-153934e4.h5.log)) |
| FD-MobileNet x0.5 | 42.61 | 19.69 | 993,928 | 41.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.83/fdmobilenet_wd2-1969-5678a212.h5.log)) |
| FD-MobileNet x1.0 | 34.42 | 13.74 | 2,901,288 | 147.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.129/fdmobilenet_w1-1374-21b24355.h5.log)) |
| MobileNetV2 x0.25 | 48.06 | 24.12 | 1,516,392 | 34.24M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.137/mobilenetv2_wd4-2412-62273372.h5.log)) |
| MobileNetV2 x0.5 | 35.51 | 14.65 | 1,964,736 | 100.13M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.61/mobilenetv2_wd2-1465-774d5bca.h5.log)) |
| MobileNetV2 x0.75 | 30.81 | 11.26 | 2,627,592 | 198.50M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.61/mobilenetv2_w3d4-1126-f2f664da.h5.log)) |
| MobileNetV2 x1.0 | 28.50 | 9.90 | 3,504,960 | 329.36M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.61/mobilenetv2_w1-0990-cbb8be96.h5.log)) |
| IGCV3 x0.25 | 53.41 | 28.29 | 1,534,020 | 41.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.142/igcv3_wd4-2829-00072caf.h5.log)) |
| IGCV3 x0.5 | 39.39 | 17.04 | 1,985,528 | 111.12M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.132/igcv3_wd2-1704-b8961ca3.h5.log)) |
| IGCV3 x1.0 | 28.21 | 9.55 | 3,491,688 | 340.79M | From [homles11/IGCV3] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.126/igcv3_w1-0955-e2bde79d.h5.log)) |
| MnasNet | 31.30 | 11.45 | 4,308,816 | 317.67M | From [zeusees/Mnasnet...Model] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.117/mnasnet-1145-11b6acf1.h5.log)) |

### For TensorFlow

| Model | Top1 | Top5 | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | ---: | --- |
| AlexNet | 44.07 | 21.32 | 61,100,840 | 714.83M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.121/alexnet-2132-e3d8a249.tf.npz.log)) |
| VGG-11 | 31.89 | 11.73 | 132,863,336 | 7,615.87M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.122/vgg11-1173-ea0bf3a5.tf.npz.log)) |
| VGG-13 | 31.03 | 11.15 | 133,047,848 | 11,317.65M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.122/vgg13-1115-f01687c1.tf.npz.log)) |
| VGG-16 | 26.77 | 8.68 | 138,357,544 | 15,480.10M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.122/vgg16-0868-f6cadf2c.tf.npz.log)) |
| VGG-19 | 25.93 | 8.23 | 143,667,240 | 19,642.55M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.122/vgg19-0823-99580f95.tf.npz.log)) |
| BN-VGG-11b | 30.34 | 10.58 | 132,868,840 | 7,630.72M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.123/bn_vgg11b-1058-44558265.tf.npz.log)) |
| BN-VGG-13b | 29.47 | 10.15 | 133,053,736 | 11,342.14M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.123/bn_vgg13b-1015-999e47a6.tf.npz.log)) |
| BN-VGG-16b | 26.83 | 8.66 | 138,365,992 | 15,507.20M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.123/bn_vgg16b-0866-1f8251aa.tf.npz.log)) |
| BN-VGG-19b | 25.62 | 8.17 | 143,678,248 | 19,672.26M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.123/bn_vgg19b-0817-784e4c39.tf.npz.log)) |
| ResNet-10 | 37.11 | 15.52 | 5,418,792 | 894.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.72/resnet10-1552-e2c11848.tf.npz.log)) |
| ResNet-12 | 35.82 | 14.50 | 5,492,776 | 1,126.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.72/resnet12-1450-8865f58b.tf.npz.log)) |
| ResNet-14 | 32.83 | 12.45 | 5,788,200 | 1,357.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.72/resnet14-1245-8596c8f1.tf.npz.log)) |
| ResNet-16 | 30.66 | 11.05 | 6,968,872 | 1,589.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.72/resnet16-1105-8ee84db2.tf.npz.log)) |
| ResNet-18 x0.25 | 49.12 | 24.50 | 831,096 | 137.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.72/resnet18_wd4-2450-b536eea5.tf.npz.log)) |
| ResNet-18 x0.5 | 36.51 | 14.93 | 3,055,880 | 486.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.72/resnet18_wd2-1493-dfb5d150.tf.npz.log)) |
| ResNet-18 x0.75 | 33.28 | 12.50 | 6,675,352 | 1,047.53M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.72/resnet18_w3d4-1250-2040e339.tf.npz.log)) |
| ResNet-18 | 28.25 | 9.69 | 11,689,512 | 1,820.41M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.151/resnet18-0969-e8b92f28.tf.npz.log)) |
| ResNet-34 | 25.32 | 7.93 | 21,797,672 | 3,672.68M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.72/resnet34-0793-aaf4f066.tf.npz.log)) |
| ResNet-50 | 22.61 | 6.42 | 25,557,032 | 3,877.95M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.147/resnet50-0642-39e88383.tf.npz.log)) |
| ResNet-50b | 22.36 | 6.21 | 25,557,032 | 4,110.48M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.146/resnet50b-0621-22a3e9a9.tf.npz.log)) |
| ResNet-101 | 21.61 | 6.01 | 44,549,160 | 7,597.95M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.72/resnet101-0601-3fc260bc.tf.npz.log)) |
| ResNet-101b | 20.81 | 5.40 | 44,549,160 | 7,830.48M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.145/resnet101b-0540-4e2ec57c.tf.npz.log)) |
| ResNet-152 | 20.73 | 5.35 | 60,192,808 | 11,321.85M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.144/resnet152-0535-b21844fc.tf.npz.log)) |
| ResNet-152b | 20.27 | 5.23 | 60,192,808 | 11,554.38M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.143/resnet152b-0523-da1f46f3.tf.npz.log)) |
| PreResNet-18 | 28.21 | 9.49 | 11,687,848 | 1,820.56M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.140/preresnet18-0949-692e6c11.tf.npz.log)) |
| PreResNet-34 | 25.82 | 8.08 | 21,796,008 | 3,672.83M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.73/preresnet34-0808-ceab73cc.tf.npz.log)) |
| PreResNet-50 | 23.42 | 6.68 | 25,549,480 | 3,875.44M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.73/preresnet50-0668-822837cf.tf.npz.log)) |
| PreResNet-50b | 23.12 | 6.61 | 25,549,480 | 4,107.97M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.73/preresnet50b-0661-49f158a2.tf.npz.log)) |
| PreResNet-101 | 21.49 | 5.72 | 44,541,608 | 7,595.44M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.73/preresnet101-0572-cd61594e.tf.npz.log)) |
| PreResNet-101b | 21.70 | 5.91 | 44,541,608 | 7,827.97M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.73/preresnet101b-0591-93ae5e69.tf.npz.log)) |
| PreResNet-152 | 20.63 | 5.29 | 60,185,256 | 11,319.34M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.73/preresnet152-0529-b761f286.tf.npz.log)) |
| PreResNet-152b | 20.95 | 5.76 | 60,185,256 | 11,551.87M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.73/preresnet152b-0576-c036165c.tf.npz.log)) |
| PreResNet-200b | 21.12 | 5.60 | 64,666,280 | 15,068.63M | From [tornadomeet/ResNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.73/preresnet200b-0560-881e0e28.tf.npz.log)) |
| ResNeXt-101 (32x4d) | 21.33 | 5.80 | 44,177,704 | 8,003.45M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.74/resnext101_32x4d-0580-bf746cb6.tf.npz.log)) |
| ResNeXt-101 (64x4d) | 20.59 | 5.43 | 83,455,272 | 15,500.27M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.74/resnext101_64x4d-0543-f51ffdb0.tf.npz.log)) |
| SE-ResNet-50 | 22.53 | 6.43 | 28,088,024 | 3,880.49M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.75/seresnet50-0643-e022e5b9.tf.npz.log)) |
| SE-ResNet-101 | 21.92 | 5.89 | 49,326,872 | 7,602.76M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.75/seresnet101-0589-305d2301.tf.npz.log)) |
| SE-ResNet-152 | 21.48 | 5.78 | 66,821,848 | 11,328.52M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.75/seresnet152-0578-d06ab6d9.tf.npz.log)) |
| SE-ResNeXt-50 (32x4d) | 21.01 | 5.53 | 27,559,896 | 4,258.40M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.76/seresnext50_32x4d-0553-20723214.tf.npz.log)) |
| SE-ResNeXt-101 (32x4d) | 19.99 | 4.97 | 48,955,416 | 8,008.26M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.76/seresnext101_32x4d-0497-268d7d22.tf.npz.log)) |
| SENet-154 | 18.77 | 4.63 | 115,088,984 | 20,745.78M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.86/senet154-0463-c86eaaed.tf.npz.log)) |
| DenseNet-121 | 25.16 | 7.82 | 7,978,856 | 2,872.13M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.77/densenet121-0782-1bfa61d4.tf.npz.log)) |
| DenseNet-161 | 22.40 | 6.17 | 28,681,000 | 7,793.16M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.77/densenet161-0617-9deca33a.tf.npz.log)) |
| DenseNet-169 | 23.93 | 6.87 | 14,149,480 | 3,403.89M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.77/densenet169-0687-23910539.tf.npz.log)) |
| DenseNet-201 | 22.70 | 6.35 | 20,013,928 | 4,347.15M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.77/densenet201-0635-5eda7895.tf.npz.log)) |
| DarkNet Tiny | 40.35 | 17.51 | 1,042,104 | 500.85M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.71/darknet_tiny-1751-750ff8d9.tf.npz.log)) |
| DarkNet Ref | 37.99 | 16.72 | 7,319,416 | 367.59M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.71/darknet_ref-1672-3c8ed62a.tf.npz.log)) |
| DarkNet-53 | 21.42 | 5.55 | 41,609,928 | 7,133.86M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.150/darknet53-0555-49816dbf.tf.npz.log)) |
| SqueezeNet v1.0 | 39.18 | 17.58 | 1,248,424 | 823.67M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.128/squeezenet_v1_0-1758-fc6384ff.tf.npz.log)) |
| SqueezeNet v1.1 | 39.14 | 17.39 | 1,235,496 | 352.02M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.88/squeezenet_v1_1-1739-48945577.tf.npz.log)) |
| SqueezeResNet v1.1 | 39.75 | 17.92 | 1,235,496 | 352.02M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.79/squeezeresnet_v1_1-1792-44c17928.tf.npz.log)) |
| 1.0-SqNxt-23 | 48.14 | 23.51 | 724,056 | 287.28M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.138/sqnxt23_w1-2351-c353f458.tf.npz.log)) |
| ShuffleNet x0.25 (g=1) | 62.03 | 36.80 | 209,746 | 12.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.134/shufflenet_g1_wd4-3680-3d985635.tf.npz.log)) |
| ShuffleNet x0.25 (g=3) | 61.33 | 36.17 | 305,902 | 13.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.135/shufflenet_g3_wd4-3617-8f00e642.tf.npz.log)) |
| ShuffleNetV2 x0.5 | 40.88 | 18.44 | 1,366,792 | 43.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.90/shufflenetv2_wd2-1844-2bd8a314.tf.npz.log)) |
| ShuffleNetV2b x0.5 | 41.03 | 18.59 | 1,366,792 | 43.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.112/shufflenetv2b_wd2-1859-67249edb.tf.npz.log)) |
| ShuffleNetV2c x0.5 | 39.93 | 18.11 | 1,366,792 | 43.31M | From [tensorpack/tensorpack] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.91/shufflenetv2c_wd2-1811-98435af9.tf.npz.log)) |
| ShuffleNetV2 x1.0 | 31.02 | 11.31 | 2,278,604 | 149.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.133/shufflenetv2_w1-1131-6a728e21.tf.npz.log)) |
| ShuffleNetV2c x1.0 | 30.77 | 11.39 | 2,279,760 | 150.62M | From [tensorpack/tensorpack] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.95/shufflenetv2c_w1-1139-47dd03c8.tf.npz.log)) |
| ShuffleNetV2 x1.5 | 32.51 | 12.50 | 4,406,098 | 320.77M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.85/shufflenetv2_w3d2-1250-5dd7b5b1.tf.npz.log)) |
| ShuffleNetV2 x2.0 | 31.99 | 12.26 | 7,601,686 | 595.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.85/shufflenetv2_w2-1226-f66f6987.tf.npz.log)) |
| 108-MENet-8x1 (g=3) | 43.67 | 20.32 | 654,516 | 42.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.89/menet108_8x1_g3-2032-4e9e89e1.tf.npz.log)) |
| 128-MENet-8x1 (g=4) | 42.04 | 19.15 | 750,796 | 45.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.103/menet128_8x1_g4-1915-148105f4.tf.npz.log)) |
| 228-MENet-12x1 (g=3) | 33.85 | 12.92 | 1,806,568 | 152.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.131/menet228_12x1_g3-1292-e594e8bb.tf.npz.log)) |
| 256-MENet-12x1 (g=4) | 34.48 | 13.95 | 1,888,240 | 150.65M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.87/menet256_12x1_g4-1395-d0ce72b1.tf.npz.log)) |
| 348-MENet-12x1 (g=3) | 31.19 | 11.41 | 3,368,128 | 312.00M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.87/menet348_12x1_g3-1141-f90f3c12.tf.npz.log)) |
| 352-MENet-12x1 (g=8) | 34.65 | 13.71 | 2,272,872 | 157.35M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.87/menet352_12x1_g8-1371-3621d3c0.tf.npz.log)) |
| 456-MENet-24x1 (g=3) | 29.56 | 10.46 | 5,304,784 | 567.90M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.87/menet456_24x1_g3-1046-6d70fb21.tf.npz.log)) |
| MobileNet x0.25 | 45.78 | 22.21 | 470,072 | 44.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.80/mobilenet_wd4-2221-15ee9820.tf.npz.log)) |
| MobileNet x0.5 | 34.74 | 13.66 | 1,331,592 | 155.42M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.148/mobilenet_wd2-1366-9ece12a0.tf.npz.log)) |
| MobileNet x0.75 | 29.82 | 10.49 | 2,585,560 | 333.99M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.130/mobilenet_w3d4-1049-3139bba7.tf.npz.log)) |
| MobileNet x1.0 | 26.75 | 8.74 | 4,231,976 | 579.80M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.149/mobilenet_w1-0874-6562a6c8.tf.npz.log)) |
| FD-MobileNet x0.25 | 56.08 | 31.44 | 383,160 | 12.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.81/fdmobilenet_wd4-3144-3febaec9.tf.npz.log)) |
| FD-MobileNet x0.5 | 42.67 | 19.70 | 993,928 | 41.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.83/fdmobilenet_wd2-1970-d778e687.tf.npz.log)) |
| FD-MobileNet x1.0 | 34.47 | 13.74 | 2,901,288 | 147.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.129/fdmobilenet_w1-1374-9f999806.tf.npz.log)) |
| MobileNetV2 x0.25 | 48.18 | 24.16 | 1,516,392 | 34.24M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.137/mobilenetv2_wd4-2416-ae7e5137.tf.npz.log)) |
| MobileNetV2 x0.5 | 35.51 | 14.60 | 1,964,736 | 100.13M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.82/mobilenetv2_wd2-1460-12376d24.tf.npz.log)) |
| MobileNetV2 x0.75 | 30.79 | 11.24 | 2,627,592 | 198.50M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.82/mobilenetv2_w3d4-1124-3531c997.tf.npz.log)) |
| MobileNetV2 x1.0 | 28.53 | 9.90 | 3,504,960 | 329.36M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.82/mobilenetv2_w1-0990-e80f9fe4.tf.npz.log)) |
| IGCV3 x0.25 | 53.39 | 28.35 | 1,534,020 | 41.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.142/igcv3_wd4-2835-b41fb3c7.tf.npz.log)) |
| IGCV3 x0.5 | 39.38 | 17.05 | 1,985,528 | 111.12M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.132/igcv3_wd2-1705-de0b98d9.tf.npz.log)) |
| IGCV3 x1.0 | 28.17 | 9.55 | 3,491,688 | 340.79M | From [homles11/IGCV3] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.126/igcv3_w1-0955-cb263e3a.tf.npz.log)) |
| MnasNet | 31.29 | 11.44 | 4,308,816 | 317.67M | From [zeusees/Mnasnet...Model] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.117/mnasnet-1144-f2b84fc4.tf.npz.log)) |

[dmlc/gluon-cv]: https://github.com/dmlc/gluon-cv
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
[quark0/darts]: https://github.com/quark0/darts
[homles11/IGCV3]: https://github.com/homles11/IGCV3
[soeaver/AirNet-PyTorch]: https://github.com/soeaver/AirNet-PyTorch
[Jongchan/attention-module]: https://github.com/Jongchan/attention-module
[XingangPan/IBN-Net]: https://github.com/XingangPan/IBN-Net