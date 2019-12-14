# Large-scale image classification models on TensorFlow 1.x

[![PyPI](https://img.shields.io/pypi/v/tensorflowcv.svg)](https://pypi.python.org/pypi/tensorflowcv)
[![Downloads](https://pepy.tech/badge/tensorflowcv)](https://pepy.tech/project/tensorflowcv)

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

## Installation

To use the models in your project, simply install the `tensorflowcv` package with `tensorflow`:
```
pip install tensorflowcv tensorflow>=1.11.0
```
To enable/disable different hardware supports, check out TensorFlow installation [instructions](https://www.tensorflow.org).

## Usage

Example of using a pretrained ResNet-18 model (with `channels_first` data format):
```
from tensorflowcv.model_provider import get_model as tfcv_get_model
from tensorflowcv.model_provider import init_variables_from_state_dict as tfcv_init_variables_from_state_dict
import tensorflow as tf
import numpy as np

net = tfcv_get_model("resnet18", pretrained=True, data_format="channels_first")
x = tf.placeholder(dtype=tf.float32, shape=(None, 3, 224, 224), name="xx")
y_net = net(x)

with tf.Session() as sess:
    tfcv_init_variables_from_state_dict(sess=sess, state_dict=net.state_dict)
    x_value = np.zeros((1, 3, 224, 224), np.float32)
    y = sess.run(y_net, feed_dict={x: x_value})
```

## Pretrained models (ImageNet-1K)

Some remarks:
- Top1/Top5 are the standard 1-crop Top-1/Top-5 errors (in percents) on the validation subset of the ImageNet-1K dataset.
- FLOPs/2 is the number of FLOPs divided by two to be similar to the number of MACs.
- Remark `Converted from GL model` means that the model was trained on `MXNet/Gluon` and then converted to TensorFlow.

| Model | Top1 | Top5 | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | ---: | --- |
| AlexNet | 40.44 | 17.88 | 62,378,344 | 1,132.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.394/alexnet-1788-d3cd2a5a.tf.npz.log)) |
| AlexNet-b | 41.05 | 18.53 | 61,100,840 | 714.83M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.384/alexnetb-1853-58a51cd1.tf.npz.log)) |
| ZFNet | 39.51 | 17.15 | 62,357,608 | 1,170.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.395/zfnet-1715-a18747ef.tf.npz.log)) |
| ZFNet-b | 36.30 | 14.82 | 107,627,624 | 2,479.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.400/zfnetb-1482-2624da31.tf.npz.log)) |
| VGG-11 | 29.58 | 10.15 | 132,863,336 | 7,615.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.381/vgg11-1015-b87e9dbc.tf.npz.log)) |
| VGG-13 | 28.34 | 9.46 | 133,047,848 | 11,317.65M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.388/vgg13-0946-f1411e1f.tf.npz.log)) |
| VGG-16 | 26.65 | 8.30 | 138,357,544 | 15,480.10M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.401/vgg16-0830-e63ead2e.tf.npz.log)) |
| VGG-19 | 25.60 | 7.68 | 143,667,240 | 19,642.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.420/vgg19-0768-cf2a33c6.tf.npz.log)) |
| BN-VGG-11 | 28.55 | 9.36 | 132,866,088 | 7,630.21M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.339/bn_vgg11-0936-4ff8667b.tf.npz.log)) |
| BN-VGG-13 | 27.73 | 8.88 | 133,050,792 | 11,341.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.353/bn_vgg13-0888-0a49f871.tf.npz.log)) |
| BN-VGG-16 | 25.48 | 7.55 | 138,361,768 | 15,506.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.359/bn_vgg16-0755-9948c82d.tf.npz.log)) |
| BN-VGG-19 | 23.88 | 6.89 | 143,672,744 | 19,671.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.360/bn_vgg19-0689-8a3197c6.tf.npz.log)) |
| BN-VGG-11b | 29.21 | 9.79 | 132,868,840 | 7,630.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.407/bn_vgg11b-0979-6a3890a4.tf.npz.log)) |
| BN-VGG-13b | 29.47 | 10.15 | 133,053,736 | 11,342.14M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.123/bn_vgg13b-1015-999e47a6.tf.npz.log)) |
| BN-VGG-16b | 26.83 | 8.66 | 138,365,992 | 15,507.20M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.123/bn_vgg16b-0866-1f8251aa.tf.npz.log)) |
| BN-VGG-19b | 25.62 | 8.17 | 143,678,248 | 19,672.26M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.123/bn_vgg19b-0817-784e4c39.tf.npz.log)) |
| ResNet-10 | 34.62 | 13.90 | 5,418,792 | 894.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.248/resnet10-1390-7fff13ae.tf.npz.log)) |
| ResNet-12 | 33.36 | 13.00 | 5,492,776 | 1,126.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.253/resnet12-1300-9539494f.tf.npz.log)) |
| ResNet-14 | 32.22 | 12.25 | 5,788,200 | 1,357.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.256/resnet14-1225-d1fb0f76.tf.npz.log)) |
| ResNet-BC-14b | 30.25 | 11.21 | 10,064,936 | 1,479.12M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.309/resnetbc14b-1121-45f5a6d8.tf.npz.log)) |
| ResNet-16 | 30.25 | 10.86 | 6,968,872 | 1,589.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.259/resnet16-1086-5ac8e7da.tf.npz.log)) |
| ResNet-18 x0.25 | 39.32 | 17.41 | 3,937,400 | 270.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.262/resnet18_wd4-1741-4aafd009.tf.npz.log)) |
| ResNet-18 x0.5 | 33.36 | 12.87 | 5,804,296 | 608.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.263/resnet18_wd2-1287-dac8e632.tf.npz.log)) |
| ResNet-18 x0.75 | 30.00 | 10.69 | 8,476,056 | 1,129.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.266/resnet18_w3d4-1069-d22e6604.tf.npz.log)) |
| ResNet-18 | 28.16 | 9.56 | 11,689,512 | 1,820.41M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.153/resnet18-0956-b4fc7198.tf.npz.log)) |
| ResNet-26 | 26.13 | 8.38 | 17,960,232 | 2,746.79M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.305/resnet26-0838-f647811d.tf.npz.log)) |
| ResNet-BC-26b | 24.81 | 7.57 | 15,995,176 | 2,356.67M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.313/resnetbc26b-0757-55c88013.tf.npz.log)) |
| ResNet-34 | 24.57 | 7.42 | 21,797,672 | 3,672.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.291/resnet34-0742-8faa0ab2.tf.npz.log)) |
| ResNet-BC-38b | 23.51 | 6.73 | 21,925,416 | 3,234.21M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.328/resnetbc38b-0673-324ac8fe.tf.npz.log)) |
| ResNet-50 | 22.14 | 6.05 | 25,557,032 | 3,877.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.329/resnet50-0605-34177a2e.tf.npz.log)) |
| ResNet-50b | 22.03 | 6.09 | 25,557,032 | 4,110.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.308/resnet50b-0609-4b684173.tf.npz.log)) |
| ResNet-101 | 21.61 | 6.01 | 44,549,160 | 7,597.95M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.72/resnet101-0601-3fc260bc.tf.npz.log)) |
| ResNet-101b | 20.29 | 5.07 | 44,549,160 | 7,830.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.357/resnet101b-0507-527dca37.tf.npz.log)) |
| ResNet-152 | 20.73 | 5.35 | 60,192,808 | 11,321.85M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.144/resnet152-0535-b21844fc.tf.npz.log)) |
| ResNet-152b | 19.61 | 4.85 | 60,192,808 | 11,554.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.378/resnet152b-0485-36964f48.tf.npz.log)) |
| PreResNet-10 | 34.71 | 14.01 | 5,417,128 | 894.19M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.249/preresnet10-1401-3a2eed3b.tf.npz.log)) |
| PreResNet-12 | 33.58 | 13.21 | 5,491,112 | 1,126.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.257/preresnet12-1321-0c424c40.tf.npz.log)) |
| PreResNet-14 | 32.30 | 12.16 | 5,786,536 | 1,358.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.260/preresnet14-1216-fda0747f.tf.npz.log)) |
| PreResNet-BC-14b | 30.80 | 11.53 | 10,057,384 | 1,476.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.315/preresnetbc14b-1153-00da991c.tf.npz.log)) |
| PreResNet-16 | 30.23 | 10.82 | 6,967,208 | 1,589.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.261/preresnet16-1082-865af98b.tf.npz.log)) |
| PreResNet-18 x0.25 | 39.66 | 17.76 | 3,935,960 | 270.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.272/preresnet18_wd4-1776-82bea5e8.tf.npz.log)) |
| PreResNet-18 x0.5 | 33.73 | 13.18 | 5,802,440 | 608.73M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.273/preresnet18_wd2-1318-44f39f41.tf.npz.log)) |
| PreResNet-18 x0.75 | 29.93 | 10.71 | 8,473,784 | 1,129.51M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.274/preresnet18_w3d4-1071-380470ee.tf.npz.log)) |
| PreResNet-18 | 28.21 | 9.49 | 11,687,848 | 1,820.56M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.140/preresnet18-0949-692e6c11.tf.npz.log)) |
| PreResNet-26 | 26.02 | 8.33 | 17,958,568 | 2,746.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.316/preresnet26-0833-8de37e08.tf.npz.log)) |
| PreResNet-BC-26b | 25.24 | 7.89 | 15,987,624 | 2,354.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.325/preresnetbc26b-0789-993dd84a.tf.npz.log)) |
| PreResNet-34 | 24.53 | 7.54 | 21,796,008 | 3,672.83M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.300/preresnet34-0754-9d563584.tf.npz.log)) |
| PreResNet-BC-38b | 22.70 | 6.34 | 21,917,864 | 3,231.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.348/preresnetbc38b-0634-f22aa1c3.tf.npz.log)) |
| PreResNet-50 | 22.19 | 6.25 | 25,549,480 | 3,875.44M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.330/preresnet50-0625-06130b12.tf.npz.log)) |
| PreResNet-50b | 22.36 | 6.31 | 25,549,480 | 4,107.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.307/preresnet50b-0631-9fc00073.tf.npz.log)) |
| PreResNet-101 | 21.49 | 5.72 | 44,541,608 | 7,595.44M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.73/preresnet101-0572-cd61594e.tf.npz.log)) |
| PreResNet-101b | 20.80 | 5.39 | 44,541,608 | 7,827.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.351/preresnet101b-0539-c0b9e129.tf.npz.log)) |
| PreResNet-152 | 20.63 | 5.29 | 60,185,256 | 11,319.34M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.73/preresnet152-0529-b761f286.tf.npz.log)) |
| PreResNet-152b | 19.87 | 5.00 | 60,185,256 | 11,551.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.386/preresnet152b-0500-7ae9df4b.tf.npz.log)) |
| PreResNet-200b | 21.12 | 5.60 | 64,666,280 | 15,068.63M | From [tornadomeet/ResNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.73/preresnet200b-0560-881e0e28.tf.npz.log)) |
| PreResNet-269b | 20.73 | 5.55 | 102,065,832 | 20,101.11M | From [soeaver/mxnet-model] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.239/preresnet269b-0555-c799eaf2.tf.npz.log)) |
| ResNeXt-14 (16x4d) | 31.65 | 12.24 | 7,127,336 | 1,045.77M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.370/resnext14_16x4d-1224-3f603dde.tf.npz.log)) |
| ResNeXt-14 (32x2d) | 32.11 | 12.46 | 7,029,416 | 1,031.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.371/resnext14_32x2d-1246-df7d6b8a.tf.npz.log)) |
| ResNeXt-14 (32x4d) | 29.95 | 11.13 | 9,411,880 | 1,603.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.327/resnext14_32x4d-1113-cac0dad5.tf.npz.log)) |
| ResNeXt-26 (32x2d) | 26.29 | 8.49 | 9,924,136 | 1,461.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.373/resnext26_32x2d-0849-2dee5d79.tf.npz.log)) |
| ResNeXt-26 (32x4d) | 23.94 | 7.17 | 15,389,480 | 2,488.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.332/resnext26_32x4d-0717-594567d2.tf.npz.log)) |
| ResNeXt-50 (32x4d) | 20.65 | 5.46 | 25,028,904 | 4,255.86M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.417/resnext50_32x4d-0546-c0817d9b.tf.npz.log)) |
| ResNeXt-101 (32x4d) | 19.61 | 4.93 | 44,177,704 | 8,003.45M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.417/resnext101_32x4d-0493-de52ea63.tf.npz.log)) |
| ResNeXt-101 (64x4d) | 19.27 | 4.85 | 83,455,272 | 15,500.27M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.417/resnext101_64x4d-0485-ddff97a9.tf.npz.log)) |
| SE-ResNet-10 | 33.56 | 13.36 | 5,463,332 | 894.27M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.354/seresnet10-1336-d4a0a9d3.tf.npz.log)) |
| SE-ResNet-18 | 27.89 | 9.23 | 11,778,592 | 1,820.88M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.355/seresnet18-0923-7aa519d2.tf.npz.log)) |
| SE-ResNet-26 | 25.44 | 8.09 | 18,093,852 | 2,747.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.363/seresnet26-0809-b2a8b74f.tf.npz.log)) |
| SE-ResNet-BC-26b | 23.44 | 6.81 | 17,395,976 | 2,359.58M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.366/seresnetbc26b-0681-692ccde3.tf.npz.log)) |
| SE-ResNet-BC-38b | 21.43 | 5.78 | 24,026,616 | 3,238.58M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.374/seresnetbc38b-0578-2d787dc4.tf.npz.log)) |
| SE-ResNet-50 | 22.53 | 6.43 | 28,088,024 | 3,880.49M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.75/seresnet50-0643-e022e5b9.tf.npz.log)) |
| SE-ResNet-50b | 20.60 | 5.33 | 28,088,024 | 4,115.78M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.387/seresnet50b-0533-539e58be.tf.npz.log)) |
| SE-ResNet-101 | 21.92 | 5.89 | 49,326,872 | 7,602.76M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.75/seresnet101-0589-305d2301.tf.npz.log)) |
| SE-ResNet-152 | 21.48 | 5.78 | 66,821,848 | 11,328.52M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.75/seresnet152-0578-d06ab6d9.tf.npz.log)) |
| SE-PreResNet-10 | 33.67 | 13.09 | 5,461,668 | 894.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.377/sepreresnet10-1309-b0162a2e.tf.npz.log)) |
| SE-PreResNet-18 | 27.67 | 9.41 | 11,776,928 | 1,821.03M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.380/sepreresnet18-0941-5606cb35.tf.npz.log)) |
| SE-PreResNet-BC-26b | 22.96 | 6.34 | 17,388,424 | 2,357.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.399/sepreresnetbc26b-0634-d903397d.tf.npz.log)) |
| SE-PreResNet-BC-38b | 21.37 | 5.64 | 24,019,064 | 3,236.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.409/sepreresnetbc38b-0564-262a4a2e.tf.npz.log)) |
| SE-ResNeXt-50 (32x4d) | 19.95 | 5.07 | 27,559,896 | 4,261.16M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.418/seresnext50_32x4d-0507-982a4cb8.tf.npz.log)) |
| SE-ResNeXt-101 (32x4d) | 19.02 | 4.61 | 48,955,416 | 8,012.73M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.418/seresnext101_32x4d-0461-b84ec20a.tf.npz.log)) |
| SE-ResNeXt-101 (64x4d) | 19.02 | 4.65 | 88,232,984 | 15,509.54M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.418/seresnext101_64x4d-0465-b16029e6.tf.npz.log)) |
| SENet-16 | 25.34 | 8.03 | 31,366,168 | 5,081.30M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.341/senet16-0803-366c58ce.tf.npz.log)) |
| SENet-28 | 21.68 | 5.94 | 36,453,768 | 5,732.71M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.356/senet28-0594-98ba8cc2.tf.npz.log)) |
| SENet-154 | 18.77 | 4.63 | 115,088,984 | 20,745.78M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.86/senet154-0463-c86eaaed.tf.npz.log)) |
| DenseNet-121 | 23.21 | 6.88 | 7,978,856 | 2,872.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.314/densenet121-0688-e3bccdc5.tf.npz.log)) |
| DenseNet-161 | 22.40 | 6.17 | 28,681,000 | 7,793.16M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.77/densenet161-0617-9deca33a.tf.npz.log)) |
| DenseNet-169 | 22.13 | 6.06 | 14,149,480 | 3,403.89M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.406/densenet169-0606-fcbb5c86.tf.npz.log)) |
| DenseNet-201 | 22.70 | 6.35 | 20,013,928 | 4,347.15M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.77/densenet201-0635-5eda7895.tf.npz.log)) |
| DarkNet Tiny | 40.35 | 17.51 | 1,042,104 | 500.85M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.71/darknet_tiny-1751-750ff8d9.tf.npz.log)) |
| DarkNet Ref | 37.99 | 16.72 | 7,319,416 | 367.59M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.71/darknet_ref-1672-3c8ed62a.tf.npz.log)) |
| DarkNet-53 | 21.42 | 5.55 | 41,609,928 | 7,133.86M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.150/darknet53-0555-49816dbf.tf.npz.log)) |
| SqueezeNet v1.0 | 39.18 | 17.58 | 1,248,424 | 823.67M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.128/squeezenet_v1_0-1758-fc6384ff.tf.npz.log)) |
| SqueezeNet v1.1 | 39.14 | 17.39 | 1,235,496 | 352.02M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.88/squeezenet_v1_1-1739-48945577.tf.npz.log)) |
| SqueezeResNet v1.0 | 39.36 | 17.82 | 1,248,424 | 823.67M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.178/squeezeresnet_v1_0-1782-bafdf6ae.tf.npz.log)) |
| SqueezeResNet v1.1 | 39.75 | 17.92 | 1,235,496 | 352.02M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.79/squeezeresnet_v1_1-1792-44c17928.tf.npz.log)) |
| 1.0-SqNxt-23 | 45.41 | 21.08 | 724,056 | 287.28M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.171/sqnxt23_w1-2108-62670200.tf.npz.log)) |
| 1.0-SqNxt-23v5 | 44.68 | 20.77 | 921,816 | 285.82M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.172/sqnxt23v5_w1-2077-ebc0c53d.tf.npz.log)) |
| 1.5-SqNxt-23 | 37.11 | 15.09 | 1,511,824 | 552.39M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.210/sqnxt23_w3d2-1509-8fbdcd6d.tf.npz.log)) |
| 1.5-SqNxt-23v5 | 37.33 | 15.39 | 1,953,616 | 550.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.212/sqnxt23v5_w3d2-1539-ae14d7b8.tf.npz.log)) |
| 2.0-SqNxt-23 | 32.44 | 12.35 | 2,583,752 | 898.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.240/sqnxt23_w2-1235-ea1ae9b7.tf.npz.log)) |
| 2.0-SqNxt-23v5 | 32.19 | 12.13 | 3,366,344 | 897.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.216/sqnxt23v5_w2-1213-d12c9b33.tf.npz.log)) |
| ShuffleNet x0.25 (g=1) | 62.03 | 36.80 | 209,746 | 12.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.134/shufflenet_g1_wd4-3680-3d985635.tf.npz.log)) |
| ShuffleNet x0.25 (g=3) | 61.33 | 36.17 | 305,902 | 13.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.135/shufflenet_g3_wd4-3617-8f00e642.tf.npz.log)) |
| ShuffleNet x0.5 (g=1) | 46.25 | 22.31 | 534,484 | 41.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.174/shufflenet_g1_wd2-2231-d5356e3b.tf.npz.log)) |
| ShuffleNet x0.5 (g=3) | 43.89 | 20.63 | 718,324 | 41.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.167/shufflenet_g3_wd2-2063-db302789.tf.npz.log)) |
| ShuffleNet x0.75 (g=1) | 39.26 | 16.78 | 975,214 | 86.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.218/shufflenet_g1_w3d4-1678-ca175843.tf.npz.log)) |
| ShuffleNet x0.75 (g=3) | 37.89 | 16.13 | 1,238,266 | 85.82M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.219/shufflenet_g3_w3d4-1613-f7a106be.tf.npz.log)) |
| ShuffleNet x1.0 (g=1) | 34.48 | 13.51 | 1,531,936 | 148.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.223/shufflenet_g1_w1-1351-2f36fdbc.tf.npz.log)) |
| ShuffleNet x1.0 (g=2) | 33.97 | 13.33 | 1,733,848 | 147.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.241/shufflenet_g2_w1-1333-24d32ea2.tf.npz.log)) |
| ShuffleNet x1.0 (g=3) | 33.93 | 13.32 | 1,865,728 | 145.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.244/shufflenet_g3_w1-1332-cc1781c4.tf.npz.log)) |
| ShuffleNet x1.0 (g=4) | 33.89 | 13.13 | 1,968,344 | 143.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.245/shufflenet_g4_w1-1313-25dd6c89.tf.npz.log)) |
| ShuffleNet x1.0 (g=8) | 33.65 | 13.21 | 2,434,768 | 150.76M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.250/shufflenet_g8_w1-1321-854a60f4.tf.npz.log)) |
| ShuffleNetV2 x0.5 | 40.88 | 18.44 | 1,366,792 | 43.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.90/shufflenetv2_wd2-1844-2bd8a314.tf.npz.log)) |
| ShuffleNetV2 x1.0 | 31.02 | 11.31 | 2,278,604 | 149.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.133/shufflenetv2_w1-1131-6a728e21.tf.npz.log)) |
| ShuffleNetV2 x1.5 | 27.33 | 9.23 | 4,406,098 | 320.77M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.288/shufflenetv2_w3d2-0923-6b8c6c3c.tf.npz.log)) |
| ShuffleNetV2 x2.0 | 25.80 | 8.21 | 7,601,686 | 595.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.301/shufflenetv2_w2-0821-274b770f.tf.npz.log)) |
| ShuffleNetV2b x0.5 | 39.80 | 17.84 | 1,366,792 | 43.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.158/shufflenetv2b_wd2-1784-fd5df5a3.tf.npz.log)) |
| ShuffleNetV2b x1.0 | 30.40 | 11.04 | 2,279,760 | 150.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.161/shufflenetv2b_w1-1104-6df32bad.tf.npz.log)) |
| ShuffleNetV2b x1.5 | 26.92 | 8.80 | 4,410,194 | 323.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.203/shufflenetv2b_w3d2-0880-9ce6d2b7.tf.npz.log)) |
| ShuffleNetV2b x2.0 | 25.20 | 8.10 | 7,611,290 | 603.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.242/shufflenetv2b_w2-0810-164690ed.tf.npz.log)) |
| 108-MENet-8x1 (g=3) | 43.67 | 20.32 | 654,516 | 42.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.89/menet108_8x1_g3-2032-4e9e89e1.tf.npz.log)) |
| 128-MENet-8x1 (g=4) | 42.04 | 19.15 | 750,796 | 45.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.103/menet128_8x1_g4-1915-148105f4.tf.npz.log)) |
| 160-MENet-8x1 (g=8) | 43.53 | 20.28 | 850,120 | 45.63M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.154/menet160_8x1_g8-2028-7ff635d1.tf.npz.log)) |
| 228-MENet-12x1 (g=3) | 33.85 | 12.92 | 1,806,568 | 152.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.131/menet228_12x1_g3-1292-e594e8bb.tf.npz.log)) |
| 256-MENet-12x1 (g=4) | 32.19 | 12.19 | 1,888,240 | 150.65M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.152/menet256_12x1_g4-1219-25b42dc0.tf.npz.log)) |
| 348-MENet-12x1 (g=3) | 27.87 | 9.35 | 3,368,128 | 312.00M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.173/menet348_12x1_g3-0935-bd4f0502.tf.npz.log)) |
| 352-MENet-12x1 (g=8) | 31.31 | 11.69 | 2,272,872 | 157.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.198/menet352_12x1_g8-1169-c983d04f.tf.npz.log)) |
| 456-MENet-24x1 (g=3) | 24.96 | 7.79 | 5,304,784 | 567.90M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.237/menet456_24x1_g3-0779-adc7145f.tf.npz.log)) |
| MobileNet x0.25 | 45.78 | 22.21 | 470,072 | 44.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.80/mobilenet_wd4-2221-15ee9820.tf.npz.log)) |
| MobileNet x0.5 | 33.85 | 13.31 | 1,331,592 | 155.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.156/mobilenet_wd2-1331-4c5b66f1.tf.npz.log)) |
| MobileNet x0.75 | 29.82 | 10.49 | 2,585,560 | 333.99M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.130/mobilenet_w3d4-1049-3139bba7.tf.npz.log)) |
| MobileNet x1.0 | 26.45 | 8.67 | 4,231,976 | 579.80M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.155/mobilenet_w1-0867-83beb02e.tf.npz.log)) |
| FD-MobileNet x0.25 | 55.50 | 30.50 | 383,160 | 12.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.177/fdmobilenet_wd4-3050-e441d715.tf.npz.log)) |
| FD-MobileNet x0.5 | 42.67 | 19.70 | 993,928 | 41.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.83/fdmobilenet_wd2-1970-d778e687.tf.npz.log)) |
| FD-MobileNet x0.75 | 37.95 | 16.02 | 1,833,304 | 86.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.159/fdmobilenet_w3d4-1602-91d5bf30.tf.npz.log)) |
| FD-MobileNet x1.0 | 33.78 | 13.18 | 2,901,288 | 147.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.162/fdmobilenet_w1-1318-da6a9808.tf.npz.log)) |
| MobileNetV2 x0.25 | 48.18 | 24.16 | 1,516,392 | 34.24M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.137/mobilenetv2_wd4-2416-ae7e5137.tf.npz.log)) |
| MobileNetV2 x0.5 | 35.56 | 14.46 | 1,964,736 | 100.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.170/mobilenetv2_wd2-1446-696501bd.tf.npz.log)) |
| MobileNetV2 x0.75 | 29.80 | 10.44 | 2,627,592 | 198.50M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.230/mobilenetv2_w3d4-1044-0a8633ac.tf.npz.log)) |
| MobileNetV2 x1.0 | 26.79 | 8.62 | 3,504,960 | 329.36M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.213/mobilenetv2_w1-0862-03daae54.tf.npz.log)) |
| IGCV3 x0.25 | 53.39 | 28.35 | 1,534,020 | 41.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.142/igcv3_wd4-2835-b41fb3c7.tf.npz.log)) |
| IGCV3 x0.5 | 39.38 | 17.05 | 1,985,528 | 111.12M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.132/igcv3_wd2-1705-de0b98d9.tf.npz.log)) |
| IGCV3 x0.75 | 30.80 | 10.96 | 2,638,084 | 210.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.207/igcv3_w3d4-1096-b8650159.tf.npz.log)) |
| IGCV3 x1.0 | 27.67 | 9.03 | 3,491,688 | 340.79M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.243/igcv3_w1-0903-a69c216f.tf.npz.log)) |
| MnasNet-B1 | 25.73 | 8.00 | 4,383,312 | 326.30M |  From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.419/mnasnet_b1-0800-a21e7b11.tf.npz.log)) |
| MnasNet-A1 | 25.00 | 7.56 | 3,887,038 | 326.07M |  From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.419/mnasnet_a1-0756-2903749f.tf.npz.log)) |

[dmlc/gluon-cv]: https://github.com/dmlc/gluon-cv
[tornadomeet/ResNet]: https://github.com/tornadomeet/ResNet
[Cadene/pretrained...pytorch]: https://github.com/Cadene/pretrained-models.pytorch
[tensorpack/tensorpack]: https://github.com/tensorpack/tensorpack
[clavichord93/MENet]: https://github.com/clavichord93/MENet
[zeusees/Mnasnet...Model]: https://github.com/zeusees/Mnasnet-Pretrained-Model
[soeaver/mxnet-model]: https://github.com/soeaver/mxnet-model
[rwightman/pyt...models]: https://github.com/rwightman/pytorch-image-models