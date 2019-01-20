# Large-scale image classification models on TensorFlow

[![PyPI](https://img.shields.io/pypi/v/tensorflowcv.svg)](https://pypi.python.org/pypi/tensorflowcv)
[![Downloads](https://pepy.tech/badge/tensorflowcv)](https://pepy.tech/project/tensorflowcv)

This is a collection of large-scale image classification models. Many of them are pretrained on ImageNet-1K dataset
and loaded automatically during use. All pretrained models require the same ordinary normalization. Scripts for
training/evaluating/converting models are in the [`imgclsmob`](https://github.com/osmr/imgclsmob) repo.

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

To use the models in your project, simply install the `tensorflowcv` package with `tensorflow-gpu`:
```
pip install tensorflowcv tensorflow-gpu>=1.11.0
```
To enable/disable different hardware supports, check out TensorFlow installation [instructions](https://www.tensorflow.org).

Note that the models use NCHW data format. The current version of TensorFlow cannot work with them on CPU.

## Usage

Example of using the pretrained ResNet-18 model:
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
- Top1/Top5 are the standard 1-crop Top-1/Top-5 errors (in percents) on the validation subset of the ImageNet-1K dataset.
- FLOPs/2 is the number of FLOPs divided by two to be similar to the number of MACs.
- Remark `Converted from GL model` means that the model was trained on `MXNet/Gluon` and then converted to Keras.

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
| ResNet-18 | 28.16 | 9.56 | 11,689,512 | 1,820.41M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.153/resnet18-0956-b4fc7198.tf.npz.log)) |
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
| 1.0-SqNxt-23 | 45.41 | 21.08 | 724,056 | 287.28M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.171/sqnxt23_w1-2108-62670200.tf.npz.log)) |
| 1.0-SqNxt-23v5 | 44.68 | 20.77 | 921,816 | 285.82M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.172/sqnxt23v5_w1-2077-ebc0c53d.tf.npz.log)) |
| ShuffleNet x0.25 (g=1) | 62.03 | 36.80 | 209,746 | 12.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.134/shufflenet_g1_wd4-3680-3d985635.tf.npz.log)) |
| ShuffleNet x0.25 (g=3) | 61.33 | 36.17 | 305,902 | 13.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.135/shufflenet_g3_wd4-3617-8f00e642.tf.npz.log)) |
| ShuffleNet x0.5 (g=3) | 43.89 | 20.63 | 718,324 | 41.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.167/shufflenet_g3_wd2-2063-db302789.tf.npz.log)) |
| ShuffleNetV2 x0.5 | 40.88 | 18.44 | 1,366,792 | 43.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.90/shufflenetv2_wd2-1844-2bd8a314.tf.npz.log)) |
| ShuffleNetV2 x1.0 | 31.02 | 11.31 | 2,278,604 | 149.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.133/shufflenetv2_w1-1131-6a728e21.tf.npz.log)) |
| ShuffleNetV2 x1.5 | 32.51 | 12.50 | 4,406,098 | 320.77M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.85/shufflenetv2_w3d2-1250-5dd7b5b1.tf.npz.log)) |
| ShuffleNetV2 x2.0 | 31.99 | 12.26 | 7,601,686 | 595.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.85/shufflenetv2_w2-1226-f66f6987.tf.npz.log)) |
| ShuffleNetV2b x0.5 | 39.80 | 17.84 | 1,366,792 | 43.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.158/shufflenetv2b_wd2-1784-fd5df5a3.tf.npz.log)) |
| ShuffleNetV2b x1.0 | 30.40 | 11.04 | 2,279,760 | 150.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.161/shufflenetv2b_w1-1104-6df32bad.tf.npz.log)) |
| ShuffleNetV2c x0.5 | 39.93 | 18.11 | 1,366,792 | 43.31M | From [tensorpack/tensorpack] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.158/shufflenetv2c_wd2-1811-8da982e0.tf.npz.log)) |
| ShuffleNetV2c x1.0 | 30.77 | 11.39 | 2,279,760 | 150.62M | From [tensorpack/tensorpack] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.158/shufflenetv2c_w1-1139-5117ee49.tf.npz.log)) |
| 108-MENet-8x1 (g=3) | 43.67 | 20.32 | 654,516 | 42.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.89/menet108_8x1_g3-2032-4e9e89e1.tf.npz.log)) |
| 128-MENet-8x1 (g=4) | 42.04 | 19.15 | 750,796 | 45.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.103/menet128_8x1_g4-1915-148105f4.tf.npz.log)) |
| 160-MENet-8x1 (g=8) | 43.53 | 20.28 | 850,120 | 45.63M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.154/menet160_8x1_g8-2028-7ff635d1.tf.npz.log)) |
| 228-MENet-12x1 (g=3) | 33.85 | 12.92 | 1,806,568 | 152.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.131/menet228_12x1_g3-1292-e594e8bb.tf.npz.log)) |
| 256-MENet-12x1 (g=4) | 32.19 | 12.19 | 1,888,240 | 150.65M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.152/menet256_12x1_g4-1219-25b42dc0.tf.npz.log)) |
| 348-MENet-12x1 (g=3) | 31.19 | 11.41 | 3,368,128 | 312.00M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.87/menet348_12x1_g3-1141-f90f3c12.tf.npz.log)) |
| 352-MENet-12x1 (g=8) | 34.65 | 13.71 | 2,272,872 | 157.35M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.87/menet352_12x1_g8-1371-3621d3c0.tf.npz.log)) |
| 456-MENet-24x1 (g=3) | 29.56 | 10.46 | 5,304,784 | 567.90M | From [clavichord93/MENet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.87/menet456_24x1_g3-1046-6d70fb21.tf.npz.log)) |
| MobileNet x0.25 | 45.78 | 22.21 | 470,072 | 44.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.80/mobilenet_wd4-2221-15ee9820.tf.npz.log)) |
| MobileNet x0.5 | 33.85 | 13.31 | 1,331,592 | 155.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.156/mobilenet_wd2-1331-4c5b66f1.tf.npz.log)) |
| MobileNet x0.75 | 29.82 | 10.49 | 2,585,560 | 333.99M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.130/mobilenet_w3d4-1049-3139bba7.tf.npz.log)) |
| MobileNet x1.0 | 26.45 | 8.67 | 4,231,976 | 579.80M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.155/mobilenet_w1-0867-83beb02e.tf.npz.log)) |
| FD-MobileNet x0.25 | 56.08 | 31.44 | 383,160 | 12.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.81/fdmobilenet_wd4-3144-3febaec9.tf.npz.log)) |
| FD-MobileNet x0.5 | 42.67 | 19.70 | 993,928 | 41.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.83/fdmobilenet_wd2-1970-d778e687.tf.npz.log)) |
| FD-MobileNet x0.75 | 37.95 | 16.02 | 1,833,304 | 86.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.159/fdmobilenet_w3d4-1602-91d5bf30.tf.npz.log)) |
| FD-MobileNet x1.0 | 33.78 | 13.18 | 2,901,288 | 147.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.162/fdmobilenet_w1-1318-da6a9808.tf.npz.log)) |
| MobileNetV2 x0.25 | 48.18 | 24.16 | 1,516,392 | 34.24M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.137/mobilenetv2_wd4-2416-ae7e5137.tf.npz.log)) |
| MobileNetV2 x0.5 | 35.56 | 14.46 | 1,964,736 | 100.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.170/mobilenetv2_wd2-1446-696501bd.tf.npz.log)) |
| MobileNetV2 x0.75 | 30.79 | 11.24 | 2,627,592 | 198.50M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.82/mobilenetv2_w3d4-1124-3531c997.tf.npz.log)) |
| MobileNetV2 x1.0 | 28.53 | 9.90 | 3,504,960 | 329.36M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.82/mobilenetv2_w1-0990-e80f9fe4.tf.npz.log)) |
| IGCV3 x0.25 | 53.39 | 28.35 | 1,534,020 | 41.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.142/igcv3_wd4-2835-b41fb3c7.tf.npz.log)) |
| IGCV3 x0.5 | 39.38 | 17.05 | 1,985,528 | 111.12M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.132/igcv3_wd2-1705-de0b98d9.tf.npz.log)) |
| IGCV3 x1.0 | 28.17 | 9.55 | 3,491,688 | 340.79M | From [homles11/IGCV3] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.126/igcv3_w1-0955-cb263e3a.tf.npz.log)) |
| MnasNet | 31.29 | 11.44 | 4,308,816 | 317.67M | From [zeusees/Mnasnet...Model] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.117/mnasnet-1144-f2b84fc4.tf.npz.log)) |

[dmlc/gluon-cv]: https://github.com/dmlc/gluon-cv
[tornadomeet/ResNet]: https://github.com/tornadomeet/ResNet
[Cadene/pretrained...pytorch]: https://github.com/Cadene/pretrained-models.pytorch
[tensorpack/tensorpack]: https://github.com/tensorpack/tensorpack
[clavichord93/MENet]: https://github.com/clavichord93/MENet
[homles11/IGCV3]: https://github.com/homles11/IGCV3
[zeusees/Mnasnet...Model]: https://github.com/zeusees/Mnasnet-Pretrained-Model
