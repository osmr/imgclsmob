# Computer vision models on TensorFlow 2.x

[![PyPI](https://img.shields.io/pypi/v/tf2cv.svg)](https://pypi.python.org/pypi/tf2cv)
[![Downloads](https://pepy.tech/badge/tf2cv)](https://pepy.tech/project/tf2cv)

This is a collection of image classification, segmentation, detection, and pose estimation models. Many of them are pretrained on
[ImageNet-1K](http://www.image-net.org), [CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html),
[SVHN](http://ufldl.stanford.edu/housenumbers), [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html),
[Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012), [ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K),
[Cityscapes](https://www.cityscapes-dataset.com), and [COCO](http://cocodataset.org) datasets and loaded automatically
during use. All pretrained models require the same ordinary normalization. Scripts for training/evaluating/converting
models are in the [`imgclsmob`](https://github.com/osmr/imgclsmob) repo.

## List of implemented models

- AlexNet (['One weird trick for parallelizing convolutional neural networks'](https://arxiv.org/abs/1404.5997))
- ZFNet (['Visualizing and Understanding Convolutional Networks'](https://arxiv.org/abs/1311.2901))
- VGG/BN-VGG (['Very Deep Convolutional Networks for Large-Scale Image Recognition'](https://arxiv.org/abs/1409.1556))
- BN-Inception (['Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift'](https://arxiv.org/abs/1502.03167))
- ResNet (['Deep Residual Learning for Image Recognition'](https://arxiv.org/abs/1512.03385))
- PreResNet (['Identity Mappings in Deep Residual Networks'](https://arxiv.org/abs/1603.05027))
- ResNeXt (['Aggregated Residual Transformations for Deep Neural Networks'](http://arxiv.org/abs/1611.05431))
- SENet/SE-ResNet/SE-PreResNet/SE-ResNeXt (['Squeeze-and-Excitation Networks'](https://arxiv.org/abs/1709.01507))
- ResNeSt(A) (['ResNeSt: Split-Attention Networks'](https://arxiv.org/abs/2004.08955))
- IBN-ResNet/IBN-ResNeXt/IBN-DenseNet (['Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net'](https://arxiv.org/abs/1807.09441))
- AirNet/AirNeXt (['Attention Inspiring Receptive-Fields Network for Learning Invariant Representations'](https://ieeexplore.ieee.org/document/8510896))
- BAM-ResNet (['BAM: Bottleneck Attention Module'](https://arxiv.org/abs/1807.06514))
- CBAM-ResNet (['CBAM: Convolutional Block Attention Module'](https://arxiv.org/abs/1807.06521))
- SCNet (['Improving Convolutional Networks with Self-Calibrated Convolutions'](http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf))
- RegNet (['Designing Network Design Spaces'](https://arxiv.org/abs/2003.13678))
- PyramidNet (['Deep Pyramidal Residual Networks'](https://arxiv.org/abs/1610.02915))
- DiracNetV2 (['DiracNets: Training Very Deep Neural Networks Without Skip-Connections'](https://arxiv.org/abs/1706.00388))
- DenseNet (['Densely Connected Convolutional Networks'](https://arxiv.org/abs/1608.06993))
- PeleeNet (['Pelee: A Real-Time Object Detection System on Mobile Devices'](https://arxiv.org/abs/1804.06882))
- WRN (['Wide Residual Networks'](https://arxiv.org/abs/1605.07146))
- DRN-C/DRN-D (['Dilated Residual Networks'](https://arxiv.org/abs/1705.09914))
- DPN (['Dual Path Networks'](https://arxiv.org/abs/1707.01629))
- DarkNet Ref/Tiny/19 (['Darknet: Open source neural networks in c'](https://github.com/pjreddie/darknet))
- DarkNet-53 (['YOLOv3: An Incremental Improvement'](https://arxiv.org/abs/1804.02767))
- BagNet (['Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet'](https://openreview.net/pdf?id=SkfMWhAqYQ))
- DLA (['Deep Layer Aggregation'](https://arxiv.org/abs/1707.06484))
- DiCENet (['DiCENet: Dimension-wise Convolutions for Efficient Networks'](https://arxiv.org/abs/1906.03516))
- HRNet (['Deep High-Resolution Representation Learning for Visual Recognition'](https://arxiv.org/abs/1908.07919))
- VoVNet (['An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection'](https://arxiv.org/abs/1904.09730))
- SelecSLS (['XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB Camera'](https://arxiv.org/abs/1907.00837))
- HarDNet (['HarDNet: A Low Memory Traffic Network'](https://arxiv.org/abs/1909.00948))
- SqueezeNet/SqueezeResNet (['SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size'](https://arxiv.org/abs/1602.07360))
- SqueezeNext (['SqueezeNext: Hardware-Aware Neural Network Design'](https://arxiv.org/abs/1803.10615))
- ShuffleNet (['ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices'](https://arxiv.org/abs/1707.01083))
- ShuffleNetV2 (['ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design'](https://arxiv.org/abs/1807.11164))
- MENet (['Merging and Evolution: Improving Convolutional Neural Networks for Mobile Applications'](https://arxiv.org/abs/1803.09127))
- MobileNet (['MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications'](https://arxiv.org/abs/1704.04861))
- FD-MobileNet (['FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy'](https://arxiv.org/abs/1802.03750))
- MobileNetV2 (['MobileNetV2: Inverted Residuals and Linear Bottlenecks'](https://arxiv.org/abs/1801.04381))
- MobileNetV3 (['Searching for MobileNetV3'](https://arxiv.org/abs/1905.02244))
- IGCV3 (['IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks'](https://arxiv.org/abs/1806.00178))
- GhostNet (['GhostNet: More Features from Cheap Operations'](https://arxiv.org/abs/1911.11907))
- MnasNet (['MnasNet: Platform-Aware Neural Architecture Search for Mobile'](https://arxiv.org/abs/1807.11626))
- ProxylessNAS (['ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware'](https://arxiv.org/abs/1812.00332))
- FBNet (['FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search'](https://arxiv.org/abs/1812.03443))
- Xception (['Xception: Deep Learning with Depthwise Separable Convolutions'](https://arxiv.org/abs/1610.02357))
- InceptionV3 (['Rethinking the Inception Architecture for Computer Vision'](https://arxiv.org/abs/1512.00567))
- InceptionV4/InceptionResNetV2 (['Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning'](https://arxiv.org/abs/1602.07261))
- PolyNet (['PolyNet: A Pursuit of Structural Diversity in Very Deep Networks'](https://arxiv.org/abs/1611.05725))
- NASNet (['Learning Transferable Architectures for Scalable Image Recognition'](https://arxiv.org/abs/1707.07012))
- PNASNet (['Progressive Neural Architecture Search'](https://arxiv.org/abs/1712.00559))
- SPNASNet (['Single-Path NAS: Designing Hardware-Efficient ConvNets in less than 4 Hours'](https://arxiv.org/abs/1904.02877))
- EfficientNet (['EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks'](https://arxiv.org/abs/1905.11946))
- MixNet (['MixConv: Mixed Depthwise Convolutional Kernels'](https://arxiv.org/abs/1907.09595))
- FCN-8s (['Fully Convolutional Networks for Semantic Segmentation'](https://arxiv.org/abs/1411.4038))
- PSPNet (['Pyramid Scene Parsing Network'](https://arxiv.org/abs/1612.01105))
- DeepLabv3 (['Rethinking Atrous Convolution for Semantic Image Segmentation'](https://arxiv.org/abs/1706.05587))
- ICNet (['ICNet for Real-Time Semantic Segmentation on High-Resolution Images'](https://arxiv.org/abs/1704.08545))
- Fast-SCNN (['Fast-SCNN: Fast Semantic Segmentation Network'](https://arxiv.org/abs/1902.04502))
- CGNet (['CGNet: A Light-weight Context Guided Network for Semantic Segmentation'](https://arxiv.org/abs/1811.08201))
- DABNet (['DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation'](https://arxiv.org/abs/1907.11357))
- BiSeNet (['BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation'](https://arxiv.org/abs/1808.00897))
- DANet (['Dual Attention Network for Scene Segmentation'](https://arxiv.org/abs/1809.02983))
- FPENet (['Feature Pyramid Encoding Network for Real-time Semantic Segmentation'](https://arxiv.org/abs/1909.08599))
- LEDNet (['LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation'](https://arxiv.org/abs/1905.02423))
- CenterNet (['Objects as Points'](https://arxiv.org/abs/1904.07850))
- LFFD (['LFFD: A Light and Fast Face Detector for Edge Devices'](https://arxiv.org/abs/1904.10633))
- AlphaPose (['RMPE: Regional Multi-person Pose Estimation'](https://arxiv.org/abs/1612.00137))
- SimplePose (['Simple Baselines for Human Pose Estimation and Tracking'](https://arxiv.org/abs/1804.06208))
- Lightweight OpenPose (['Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose'](https://arxiv.org/abs/1811.12004))
- IBPPose (['Simple Pose: Rethinking and Improving a Bottom-up Approach for Multi-Person Pose Estimation'](https://arxiv.org/abs/1911.10529))
- VOCA (['Capture, Learning, and Synthesis of 3D Speaking Styles'](https://arxiv.org/abs/1905.03079))
- Neural Voice Puppetry Audio-to-Expression net (['Neural Voice Puppetry: Audio-driven Facial Reenactment'](https://arxiv.org/abs/1912.05566))
- Jasper/JasperDR (['Jasper: An End-to-End Convolutional Neural Acoustic Model'](https://arxiv.org/abs/1904.03288))
- QuartzNet (['QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel Separable Convolutions'](https://arxiv.org/abs/1910.10261))

## Installation

To use the models in your project, simply install the `tf2cv` package with `tensorflow`:
```
pip install tf2cv tensorflow>=2.0.0
```
To enable/disable different hardware supports, check out TensorFlow installation [instructions](https://www.tensorflow.org).

## Usage

Example of using a pretrained ResNet-18 model (with `channels_first` data format):
```
from tf2cv.model_provider import get_model as tf2cv_get_model
import tensorflow as tf

net = tf2cv_get_model("resnet18", pretrained=True, data_format="channels_last")
x = tf.random.normal((1, 224, 224, 3))
y_net = net(x)
```

## Pretrained models (ImageNet-1K)

Some remarks:
- Top1/Top5 are the standard 1-crop Top-1/Top-5 errors (in percents) on the validation subset of the ImageNet-1K dataset.
- FLOPs/2 is the number of FLOPs divided by two to be similar to the number of MACs.
- Remark `Converted from GL model` means that the model was trained on `MXNet/Gluon` and then converted to TensorFlow.
- ResNet(A) is an average downsampled ResNet intended for use as an feature extractor in some pose estimation networks.
- ResNet(D) is a dilated ResNet intended for use as an feature extractor in some segmentation networks.
- Models with *-suffix use non-standard preprocessing (see the training log).

| Model | Top1 | Top5 | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | ---: | --- |
| AlexNet | 38.06 | 16.09 | 62,378,344 | 1,132.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.481/alexnet-1609-8ae4618e.tf2.h5.log)) |
| AlexNet-b | 39.28 | 17.06 | 61,100,840 | 714.83M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.485/alexnetb-1706-df9cb6fd.tf2.h5.log)) |
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
| BN-VGG-13b | 28.23 | 9.16 | 133,053,736 | 11,342.14M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.488/bn_vgg13b-0916-64ddd3e7.tf2.h5.log)) |
| BN-VGG-16b | 25.80 | 7.76 | 138,365,992 | 15,507.20M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.489/bn_vgg16b-0776-4e07f81c.tf2.h5.log)) |
| BN-VGG-19b | 24.86 | 7.33 | 143,678,248 | 19,672.26M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.490/bn_vgg19b-0733-7a0920e8.tf2.h5.log)) |
| BN-Inception | 26.62 | 8.65 | 11,295,240 | 2,048.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.423/bninception-0865-4cab3cce.tf2.h5.log)) |
| ResNet-10 | 32.56 | 12.56 | 5,418,792 | 894.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.483/resnet10-1256-b113c5e6.tf2.h5.log)) |
| ResNet-12 | 31.63 | 12.01 | 5,492,776 | 1,126.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.485/resnet12-1201-b8f1c73d.tf2.h5.log)) |
| ResNet-14 | 30.38 | 10.91 | 5,788,200 | 1,357.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.491/resnet14-1091-b1d49202.tf2.h5.log)) |
| ResNet-BC-14b | 29.19 | 10.37 | 10,064,936 | 1,479.12M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.481/resnetbc14b-1037-3b92ac6b.tf2.h5.log)) |
| ResNet-16 | 28.54 | 9.77 | 6,968,872 | 1,589.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/resnet16-0977-6f729109.tf2.h5.log)) |
| ResNet-18 x0.25 | 39.30 | 17.45 | 3,937,400 | 270.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet18_wd4-1745-6e800416.tf2.h5.log)) |
| ResNet-18 x0.5 | 33.40 | 12.83 | 5,804,296 | 608.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet18_wd2-1283-85a7caff.tf2.h5.log)) |
| ResNet-18 x0.75 | 29.98 | 10.67 | 8,476,056 | 1,129.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet18_w3d4-1067-c1735b7d.tf2.h5.log)) |
| ResNet-18 | 26.80 | 8.70 | 11,689,512 | 1,820.41M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.478/resnet18-0870-e1d3f22e.tf2.h5.log)) |
| ResNet-26 | 25.97 | 8.24 | 17,960,232 | 2,746.79M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.489/resnet26-0824-0ed69716.tf2.h5.log)) |
| ResNet-BC-26b | 24.80 | 7.57 | 15,995,176 | 2,356.67M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnetbc26b-0757-d70a2cad.tf2.h5.log)) |
| ResNet-34 | 24.50 | 7.44 | 21,797,672 | 3,672.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet34-0744-7f7d70e7.tf2.h5.log)) |
| ResNet-BC-38b | 23.44 | 6.77 | 21,925,416 | 3,234.21M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnetbc38b-0677-75e405a7.tf2.h5.log)) |
| ResNet-50 | 22.09 | 6.04 | 25,557,032 | 3,877.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet50-0604-728800bf.tf2.h5.log)) |
| ResNet-50b | 22.09 | 6.14 | 25,557,032 | 4,110.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet50b-0614-b2a49da6.tf2.h5.log)) |
| ResNet-101 | 20.52 | 5.18 | 44,549,160 | 7,597.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.499/resnet101-0518-64320ac1.tf2.h5.log)) |
| ResNet-101b | 20.25 | 5.11 | 44,549,160 | 7,830.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet101b-0511-e3076227.tf2.h5.log)) |
| ResNet-152 | 19.82 | 4.89 | 60,192,808 | 11,321.85M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.506/resnet152-0489-71c6f9cb.tf2.h5.log)) |
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
| PreResNet-101 | 20.59 | 5.36 | 44,541,608 | 7,595.44M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.504/preresnet101-0536-2a62fe0a.tf2.h5.log)) |
| PreResNet-101b | 20.86 | 5.39 | 44,541,608 | 7,827.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet101b-0539-54d23aff.tf2.h5.log)) |
| PreResNet-152 | 19.17 | 4.46 | 60,185,256 | 11,319.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.510/preresnet152-0446-60b1d097.tf2.h5.log)) |
| PreResNet-152b | 19.86 | 5.00 | 60,185,256 | 11,551.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet152b-0500-119062d9.tf2.h5.log)) |
| PreResNet-200b | 21.07 | 5.63 | 64,666,280 | 15,068.63M | From [tornadomeet/ResNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet200b-0563-2f9c761d.tf2.h5.log)) |
| PreResNet-269b | 20.75 | 5.57 | 102,065,832 | 20,101.11M | From [soeaver/mxnet-model] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet269b-0557-7003b3c4.tf2.h5.log)) |
| ResNeXt-14 (16x4d) | 31.69 | 12.22 | 7,127,336 | 1,045.77M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnext14_16x4d-1222-bff90c1d.tf2.h5.log)) |
| ResNeXt-14 (32x2d) | 32.14 | 12.47 | 7,029,416 | 1,031.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnext14_32x2d-1247-06aa6709.tf2.h5.log)) |
| ResNeXt-14 (32x4d) | 29.94 | 11.15 | 9,411,880 | 1,603.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnext14_32x4d-1115-3acdaec1.tf2.h5.log)) |
| ResNeXt-26 (32x2d) | 26.32 | 8.51 | 9,924,136 | 1,461.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnext26_32x2d-0851-827791cc.tf2.h5.log)) |
| ResNeXt-26 (32x4d) | 23.94 | 7.18 | 15,389,480 | 2,488.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnext26_32x4d-0718-4f05525e.tf2.h5.log)) |
| ResNeXt-50 (32x4d) | 20.82 | 5.47 | 25,028,904 | 4,255.86M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.498/resnext50_32x4d-0547-7f89b9f7.tf2.h5.log)) |
| ResNeXt-101 (32x4d) | 18.53 | 4.20 | 44,177,704 | 8,003.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.510/resnext101_32x4d-0420-0099e8ae.tf2.h5.log)) |
| ResNeXt-101 (64x4d) | 19.31 | 4.84 | 83,455,272 | 15,500.27M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnext101_64x4d-0484-f8cf1580.tf2.h5.log)) |
| SE-ResNet-10 | 31.42 | 11.71 | 5,463,332 | 894.27M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.486/seresnet10-1171-b7907036.tf2.h5.log)) |
| SE-ResNet-18 | 27.97 | 9.21 | 11,778,592 | 1,820.88M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnet18-0921-46c847ab.tf2.h5.log)) |
| SE-ResNet-26 | 25.42 | 8.07 | 18,093,852 | 2,747.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnet26-0807-5178b3b1.tf2.h5.log)) |
| SE-ResNet-BC-26b | 23.39 | 6.84 | 17,395,976 | 2,359.58M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnetbc26b-0684-1460a381.tf2.h5.log)) |
| SE-ResNet-BC-38b | 21.43 | 5.75 | 24,026,616 | 3,238.58M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnetbc38b-0575-18fcfcc1.tf2.h5.log)) |
| SE-ResNet-50 | 21.09 | 5.60 | 28,088,024 | 3,883.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.441/seresnet50-0560-f1b84c8d.tf2.h5.log)) |
| SE-ResNet-50b | 20.58 | 5.33 | 28,088,024 | 4,115.78M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnet50b-0533-256002c3.tf2.h5.log)) |
| SE-ResNet-101 | 19.61 | 4.75 | 49,326,872 | 7,602.76M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.508/seresnet101-0475-935a5b7e.tf2.h5.log)) |
| SE-ResNet-101b | 19.49 | 4.64 | 49,326,872 | 7,839.75M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.460/seresnet101b-0464-a10be1d2.tf2.h5.log)) |
| SE-ResNet-152 | 21.47 | 5.76 | 66,821,848 | 11,328.52M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnet152-0576-8023259a.tf2.h5.log)) |
| SE-PreResNet-10 | 33.62 | 13.09 | 5,461,668 | 894.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/sepreresnet10-1309-af20d06c.tf2.h5.log)) |
| SE-PreResNet-18 | 27.70 | 9.40 | 11,776,928 | 1,821.03M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/sepreresnet18-0940-fe403280.tf2.h5.log)) |
| SE-PreResNet-BC-26b | 22.95 | 6.40 | 17,388,424 | 2,357.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/sepreresnetbc26b-0640-a72bf876.tf2.h5.log)) |
| SE-PreResNet-BC-38b | 21.44 | 5.67 | 24,019,064 | 3,236.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/sepreresnetbc38b-0567-17d10c63.tf2.h5.log)) |
| SE-PreResNet-50b | 20.71 | 5.31 | 28,080,472 | 4,113.27M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.461/sepreresnet50b-0531-0882c0e9.tf2.h5.log)) |
| SE-ResNeXt-50 (32x4d) | 18.73 | 4.34 | 27,559,896 | 4,261.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.505/seresnext50_32x4d-0434-c265c58c.tf2.h5.log)) |
| SE-ResNeXt-101 (32x4d) | 19.01 | 4.59 | 48,955,416 | 8,012.73M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnext101_32x4d-0459-13a9b2fd.tf2.h5.log)) |
| SE-ResNeXt-101 (64x4d) | 18.96 | 4.65 | 88,232,984 | 15,509.54M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnext101_64x4d-0465-ec0a3b13.tf2.h5.log)) |
| SENet-16 | 25.37 | 8.05 | 31,366,168 | 5,081.30M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/senet16-0805-f5f57656.tf2.h5.log)) |
| SENet-28 | 21.68 | 5.90 | 36,453,768 | 5,732.71M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/senet28-0590-667d5687.tf2.h5.log)) |
| SENet-154 | 18.78 | 4.66 | 115,088,984 | 20,745.78M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/senet154-0466-f1b79a9b.tf2.h5.log)) |
| ResNeSt(A)-BC-14 | 22.27 | 6.35 | 10,611,688 | 2,767.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/resnestabc14-0635-fa9e06db.tf2.h5.log)) |
| ResNeSt(A)-18 | 23.42 | 6.90 | 12,763,784 | 2,587.50M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.489/resnesta18-0690-90c54f4b.tf2.h5.log)) |
| ResNeSt(A)-BC-26 | 19.58 | 4.70 | 17,069,448 | 3,646.57M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.495/resnestabc26-0470-05e07501.tf2.h5.log)) |
| ResNeSt(A)-50 | 18.87 | 4.52 | 27,483,240 | 5,403.11M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.465/resnesta50-0452-28ac82bd.tf2.h5.log)) |
| ResNeSt(A)-101 | 17.71 | 4.00 | 48,275,016 | 10,247.88M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.465/resnesta101-0400-bb2a90f5.tf2.h5.log)) |
| ResNeSt(A)-200 | 16.81 | 3.38 | 70,201,544 | 22,857.88M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.465/resnesta200-0338-29a8a745.tf2.h5.log)) |
| ResNeSt(A)-269 | 16.37 | 3.36 | 110,929,480 | 46,012.95M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.465/resnesta269-0336-9a33e31b.tf2.h5.log)) |
| IBN-ResNet-50 | 21.47 | 5.62 | 25,557,032 | 4,110.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.495/ibn_resnet50-0562-515dd253.tf2.h5.log)) |
| IBN-ResNet-101 | 21.86 | 5.84 | 44,549,160 | 7,830.48M | From [XingangPan/IBN-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/ibn_resnet101-0584-2c2c4993.tf2.h5.log)) |
| IBN(b)-ResNet-50 | 23.88 | 6.95 | 25,558,568 | 4,112.89M | From [XingangPan/IBN-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/ibnb_resnet50-0695-7178cc50.tf2.h5.log)) |
| IBN-ResNeXt-101 (32x4d) | 21.41 | 5.64 | 44,177,704 | 8,003.45M | From [XingangPan/IBN-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/ibn_resnext101_32x4d-0564-c149beb5.tf2.h5.log)) |
| IBN-DenseNet-121 | 23.34 | 6.47 | 7,978,856 | 2,872.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/ibn_densenet121-0647-830420b2.tf2.h5.log)) |
| IBN-DenseNet-169 | 22.13 | 6.07 | 14,149,480 | 3,403.89M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.500/ibn_densenet169-0607-74a97a40.tf2.h5.log)) |
| AirNet50-1x64d (r=2) | 22.54 | 6.23 | 27,425,864 | 4,772.11M | From [soeaver/AirNet-PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.423/airnet50_1x64d_r2-0623-6940f0e5.tf2.h5.log)) |
| AirNet50-1x64d (r=16) | 22.89 | 6.50 | 25,714,952 | 4,399.97M | From [soeaver/AirNet-PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.423/airnet50_1x64d_r16-0650-b7bb8662.tf2.h5.log)) |
| AirNeXt50-32x4d (r=2) | 21.47 | 5.72 | 27,604,296 | 5,339.58M | From [soeaver/AirNet-PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.423/airnext50_32x4d_r2-0572-fa8e40ab.tf2.h5.log)) |
| BAM-ResNet-50 | 20.60 | 5.37 | 25,915,099 | 4,196.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.499/bam_resnet50-0537-a9720e15.tf2.h5.log)) |
| CBAM-ResNet-50 | 22.96 | 6.39 | 28,089,624 | 4,116.97M | From [Jongchan/attention-module] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.429/cbam_resnet50-0639-1d0bdb0e.tf2.h5.log)) |
| SCNet-50 | 21.18 | 5.39 | 25,564,584 | 3,951.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/scnet50-0539-de94eb1b.tf2.h5.log)) |
| SCNet-101 | 19.82 | 4.73 | 44,565,416 | 7,204.24M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.507/scnet101-0473-61bd73af.tf2.h5.log)) |
| SCNet(A)-50 | 19.61 | 4.65 | 25,583,816 | 4,715.84M | From [MCG-NKU/SCNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.472/scneta50-0465-c1f8f295.tf2.h5.log)) |
| RegNetX-200MF | 29.94 | 10.37 | 2,684,792 | 203.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.475/regnetx002-1136-a0183973.tf2.h5.log)) |
| RegNetX-400MF | 26.28 | 8.52 | 5,157,512 | 403.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.479/regnetx004-0852-f0707cff.tf2.h5.log)) |
| RegNetX-600MF | 24.69 | 7.59 | 6,196,040 | 608.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.482/regnetx006-0759-2e47a916.tf2.h5.log)) |
| RegNetX-800MF | 24.11 | 7.27 | 7,259,656 | 809.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.482/regnetx008-0727-b19816ad.tf2.h5.log)) |
| RegNetX-1.6GF | 22.13 | 6.13 | 9,190,136 | 1,618.99M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.486/regnetx016-0613-c51845ab.tf2.h5.log)) |
| RegNetX-3.2GF | 21.31 | 5.68 | 15,296,552 | 3,199.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.492/regnetx032-0568-5f628734.tf2.h5.log)) |
| RegNetX-4.0GF | 19.51 | 4.70 | 22,118,248 | 3,986.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.495/regnetx040-0470-a3f54788.tf2.h5.log)) |
| RegNetX-6.4GF | 20.76 | 5.40 | 26,209,256 | 6,491.01M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.473/regnetx064-0540-32169638.tf2.h5.log)) |
| RegNetX-8.0GF | 20.68 | 5.42 | 39,572,648 | 8,017.94M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.473/regnetx080-0542-d0c9bc40.tf2.h5.log)) |
| RegNetX-12GF | 20.29 | 5.23 | 46,106,056 | 12,124.22M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.473/regnetx120-0523-4256f719.tf2.h5.log)) |
| RegNetX-16GF | 19.98 | 5.07 | 54,278,536 | 15,986.64M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.473/regnetx160-0507-f9023af0.tf2.h5.log)) |
| RegNetX-32GF | 19.59 | 4.85 | 107,811,560 | 31,790.24M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.473/regnetx320-0485-c11d938f.tf2.h5.log)) |
| RegNetY-200MF | 28.49 | 9.53 | 3,162,996 | 203.99M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.476/regnety002-0953-8935adba.tf2.h5.log)) |
| RegNetY-400MF | 24.82 | 7.50 | 4,344,144 | 410.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.481/regnety004-0750-65a10212.tf2.h5.log)) |
| RegNetY-600MF | 23.58 | 7.00 | 6,055,160 | 610.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.483/regnety006-0700-af7dca34.tf2.h5.log)) |
| RegNetY-800MF | 22.53 | 6.46 | 6,263,168 | 808.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.483/regnety008-0646-03922980.tf2.h5.log)) |
| RegNetY-1.6GF | 21.25 | 5.69 | 11,202,430 | 1,629.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.486/regnety016-0569-285f4f57.tf2.h5.log)) |
| RegNetY-3.2GF | 18.31 | 4.11 | 19,436,338 | 3,199.15M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.473/regnety032-0411-7fde6bb0.tf2.h5.log)) |
| RegNetY-4.0GF | 19.55 | 4.68 | 20,646,656 | 3,999.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.494/regnety040-0468-5df9e764.tf2.h5.log)) |
| RegNetY-6.4GF | 19.07 | 4.46 | 30,583,252 | 6,388.91M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.512/regnety064-0446-58f09867.tf2.h5.log)) |
| RegNetY-8.0GF | 20.04 | 5.08 | 39,180,068 | 7,996.54M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.473/regnety080-0508-f6b8907d.tf2.h5.log)) |
| RegNetY-12GF | 19.65 | 4.82 | 51,822,544 | 12,132.55M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.473/regnety120-0482-ff6070da.tf2.h5.log)) |
| RegNetY-16GF | 19.66 | 4.97 | 83,590,140 | 15,944.53M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.473/regnety160-0497-239036d5.tf2.h5.log)) |
| RegNetY-32GF | 19.10 | 4.58 | 145,046,770 | 32,317.66M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.473/regnety320-0458-b9ceda18.tf2.h5.log)) |
| PyramidNet-101 (a=360) | 20.42 | 5.20 | 42,455,070 | 8,743.54M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.507/pyramidnet101_a360-0520-a0d72160.tf2.h5.log)) |
| DiracNetV2-18 | 30.59 | 11.13 | 11,511,784 | 1,796.62M | From [szagoruyko/diracnets] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.429/diracnet18v2-1113-4d687b74.tf2.h5.log)) |
| DiracNetV2-34 | 27.92 | 9.50 | 21,616,232 | 3,646.93M | From [szagoruyko/diracnets] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.429/diracnet34v2-0950-161d97fd.tf2.h5.log)) |
| DenseNet-121 | 23.23 | 6.84 | 7,978,856 | 2,872.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/densenet121-0684-e9196a9c.tf2.h5.log)) |
| DenseNet-161 | 21.84 | 5.91 | 28,681,000 | 7,793.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.432/densenet161-0591-78224027.tf2.h5.log)) |
| DenseNet-169 | 22.13 | 6.06 | 14,149,480 | 3,403.89M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/densenet169-0606-f708dc33.tf2.h5.log)) |
| DenseNet-201 | 21.57 | 5.91 | 20,013,928 | 4,347.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.426/densenet201-0591-450c6568.tf2.h5.log)) |
| PeleeNet | 29.39 | 9.82 | 2,802,248 | 514.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.496/peleenet-0982-5f84bad2.tf2.h5.log)) |
| WRN-50-2 | 22.10 | 6.14 | 68,849,128 | 11,405.42M | From [szagoruyko/functional-zoo] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.423/wrn50_2-0614-bea17aa9.tf2.h5.log)) |
| DRN-C-26 | 24.36 | 7.10 | 21,126,584 | 16,993.90M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.508/drnc26-0710-4797ca29.tf2.h5.log)) |
| DRN-C-42 | 23.74 | 6.93 | 31,234,744 | 25,093.75M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.425/drnc42-0693-52dd6028.tf2.h5.log)) |
| DRN-C-58 | 22.36 | 6.26 | 40,542,008 | 32,489.94M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.425/drnc58-0626-e5c7be89.tf2.h5.log)) |
| DRN-D-22 | 24.71 | 7.47 | 16,393,752 | 13,051.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.498/drnd22-0747-99f94425.tf2.h5.log)) |
| DRN-D-38 | 24.52 | 7.37 | 26,501,912 | 21,151.19M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.425/drnd38-0737-a1108275.tf2.h5.log)) |
| DRN-D-54 | 22.07 | 6.26 | 35,809,176 | 28,547.38M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.425/drnd54-0626-cb792485.tf2.h5.log)) |
| DRN-D-105 | 21.31 | 5.83 | 54,801,304 | 43,442.43M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.425/drnd105-0583-80eb9ec2.tf2.h5.log)) |
| DPN-68 | 22.92 | 6.58 | 12,611,602 | 2,351.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/dpn68-0658-5b70b7b8.tf2.h5.log)) |
| DPN-98 | 20.24 | 5.28 | 61,570,728 | 11,716.51M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/dpn98-0528-6883ec37.tf2.h5.log)) |
| DPN-131 | 20.05 | 5.24 | 79,254,504 | 16,076.15M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/dpn131-0524-971af47c.tf2.h5.log)) |
| DarkNet Tiny | 40.34 | 17.45 | 1,042,104 | 500.85M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/darknet_tiny-1745-d30be41a.tf2.h5.log)) |
| DarkNet Ref | 38.10 | 16.71 | 7,319,416 | 367.59M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/darknet_ref-1671-b4991f6b.tf2.h5.log)) |
| DarkNet-53 | 21.26 | 5.54 | 41,609,928 | 7,133.86M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.501/darknet53-0554-118630cc.tf2.h5.log)) |
| BagNet-9 | 59.59 | 35.53 | 15,688,744 | 16,049.19M | From [wielandbrendel/bag...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.424/bagnet9-3553-43eb57dc.tf2.h5.log)) |
| BagNet-17 | 44.75 | 21.54 | 16,213,032 | 15,768.77M | From [wielandbrendel/bag...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.424/bagnet17-2154-8a31e347.tf2.h5.log)) |
| BagNet-33 | 36.42 | 14.97 | 18,310,184 | 16,371.52M | From [wielandbrendel/bag...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.424/bagnet33-1497-ef600c89.tf2.h5.log)) |
| DLA-34 | 24.37 | 7.05 | 15,742,104 | 3,071.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.486/dla34-0705-ade65c16.tf2.h5.log)) |
| DLA-46-C | 33.83 | 12.87 | 1,301,400 | 585.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/dla46c-1287-dfcae3b5.tf2.h5.log)) |
| DLA-X-46-C | 32.90 | 12.29 | 1,068,440 | 546.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/dla46xc-1229-a858beca.tf2.h5.log)) |
| DLA-60 | 21.26 | 5.53 | 22,036,632 | 4,255.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.494/dla60-0553-61a8f4e7.tf2.h5.log)) |
| DLA-X-60 | 20.72 | 5.50 | 17,352,344 | 3,543.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/dla60x-0550-b692a226.tf2.h5.log)) |
| DLA-X-60-C | 30.66 | 10.75 | 1,319,832 | 596.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/dla60xc-1075-a7850f03.tf2.h5.log)) |
| DLA-102 | 20.57 | 5.17 | 33,268,888 | 7,190.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.505/dla102-0517-9bebb44b.tf2.h5.log)) |
| DLA-X-102 | 20.11 | 4.91 | 26,309,272 | 5,884.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.503/dla102x-0491-0a95e90b.tf2.h5.log)) |
| DLA-X2-102 | 21.11 | 5.53 | 41,282,200 | 9,340.61M | From [ucbdrive/dla] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/dla102x2-0553-06c93031.tf2.h5.log)) |
| DLA-169 | 19.61 | 4.81 | 53,389,720 | 11,593.20M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.509/dla169-0481-39a0a5d7.tf2.h5.log)) |
| DiCENet x0.2 | 55.84 | 31.16 | 1,130,704 | 18.76M | From [sacmehta/EdgeNets] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.497/dicenet_wd5-3116-6fbe46e2.tf2.h5.log)) |
| DiCENet x0.5 | 48.32 | 24.29 | 1,214,120 | 30.48M | From [sacmehta/EdgeNets] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.497/dicenet_wd2-2429-fbf5fc52.tf2.h5.log)) |
| DiCENet x0.75 | 39.02 | 17.01 | 1,495,676 | 55.80M | From [sacmehta/EdgeNets] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.497/dicenet_w3d4-1701-bc438808.tf2.h5.log)) |
| DiCENet x1.0 | 36.01 | 14.91 | 1,805,604 | 82.17M | From [sacmehta/EdgeNets] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.497/dicenet_w1-1491-7e0a19a8.tf2.h5.log)) |
| DiCENet x1.25 | 34.29 | 13.52 | 2,162,888 | 111.87M | From [sacmehta/EdgeNets] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.497/dicenet_w5d4-1352-834f1cb5.tf2.h5.log)) |
| DiCENet x1.5 | 32.14 | 12.16 | 2,652,200 | 151.81M | From [sacmehta/EdgeNets] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.497/dicenet_w3d2-1216-8cba581c.tf2.h5.log)) |
| DiCENet x1.75 | 31.76 | 11.84 | 3,264,932 | 201.26M | From [sacmehta/EdgeNets] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.497/dicenet_w7d8-1184-0b337403.tf2.h5.log)) |
| DiCENet x2.0 | 30.49 | 11.13 | 3,979,044 | 257.95M | From [sacmehta/EdgeNets] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.497/dicenet_w2-1113-a597b5bc.tf2.h5.log)) |
| HRNet-W18 Small V1 | 26.23 | 8.71 | 13,187,464 | 1,614.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.492/hrnet_w18_small_v1-0871-6ae644af.tf2.h5.log)) |
| HRNet-W18 Small V2 | 21.69 | 6.02 | 15,597,464 | 2,618.54M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.499/hrnet_w18_small_v2-0602-e9db4e0c.tf2.h5.log)) |
| HRNetV2-W18 | 20.17 | 5.04 | 21,299,004 | 4,322.66M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.508/hrnetv2_w18-0504-5e025edd.tf2.h5.log)) |
| HRNetV2-W30 | 22.31 | 6.06 | 37,712,220 | 8,156.14M | From [HRNet/HRNet...ation] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/hrnetv2_w30-0606-4883e345.tf2.h5.log)) |
| HRNetV2-W32 | 22.32 | 6.07 | 41,232,680 | 8,973.31M | From [HRNet/HRNet...ation] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/hrnetv2_w32-0607-ef949840.tf2.h5.log)) |
| HRNetV2-W40 | 21.71 | 5.73 | 57,557,160 | 12,751.34M | From [HRNet/HRNet...ation] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/hrnetv2_w40-0573-29cece1c.tf2.h5.log)) |
| HRNetV2-W44 | 21.74 | 5.95 | 67,064,984 | 14,945.95M | From [HRNet/HRNet...ation] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/hrnetv2_w44-0595-a4e4781c.tf2.h5.log)) |
| HRNetV2-W48 | 21.42 | 5.81 | 77,469,864 | 17,344.29M | From [HRNet/HRNet...ation] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/hrnetv2_w48-0581-3af4ed57.tf2.h5.log)) |
| HRNetV2-W64 | 21.10 | 5.53 | 128,059,944 | 28,974.95M | From [HRNet/HRNet...ation] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/hrnetv2_w64-0553-aede8def.tf2.h5.log)) |
| VoVNet-39 | 23.75 | 6.94 | 22,600,296 | 7,086.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/vovnet39-0694-ae8d6df0.tf2.h5.log)) |
| VoVNet-57 | 22.42 | 6.23 | 36,640,296 | 8,943.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.505/vovnet57-0623-16133ef5.tf2.h5.log)) |
| SelecSLS-42b | 21.79 | 5.98 | 32,458,248 | 2,980.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/selecsls42b-0598-6003cd2d.tf2.h5.log)) |
| SelecSLS-60 | 20.17 | 5.13 | 30,670,768 | 3,591.78M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.496/selecsls60-0513-1f2a07e4.tf2.h5.log)) |
| SelecSLS-60b | 20.61 | 5.38 | 32,774,064 | 3,629.14M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.495/selecsls60b-0538-5e4cdf65.tf2.h5.log)) |
| HarDNet-39DS | 26.49 | 8.71 | 3,488,228 | 437.52M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.485/hardnet39ds-0871-0bd9fa5e.tf2.h5.log)) |
| HarDNet-68DS | 24.24 | 7.41 | 4,180,602 | 788.86M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.487/hardnet68ds-0741-371ee29a.tf2.h5.log)) |
| HarDNet-68 | 24.10 | 7.12 | 17,565,348 | 4,256.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.491/hardnet68-0712-bbfe6e11.tf2.h5.log)) |
| HarDNet-85 | 21.84 | 5.69 | 36,670,212 | 9,088.58M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.495/hardnet85-0569-28a9588e.tf2.h5.log)) |
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
| MobileNetb x0.25 | 45.22 | 21.69 | 467,592 | 42.88M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.481/mobilenetb_wd4-2169-4aba9700.tf2.h5.log)) |
| MobileNetb x0.5 | 32.90 | 12.69 | 1,326,632 | 153.00M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.480/mobilenetb_wd2-1269-4ebf1936.tf2.h5.log)) |
| MobileNetb x0.75 | 29.08 | 10.18 | 2,578,120 | 330.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.481/mobilenetb_w3d4-1018-2c5ff66f.tf2.h5.log)) |
| MobileNetb x1.0 | 25.04 | 7.89 | 4,222,056 | 574.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.489/mobilenetb_w1-0789-fdd5af09.tf2.h5.log)) |
| FD-MobileNet x0.25 | 55.42 | 30.62 | 383,160 | 12.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/fdmobilenet_wd4-3062-36aa16df.tf2.h5.log)) |
| FD-MobileNet x0.5 | 42.66 | 19.77 | 993,928 | 41.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/fdmobilenet_wd2-1977-34541b84.tf2.h5.log)) |
| FD-MobileNet x0.75 | 37.97 | 15.97 | 1,833,304 | 86.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/fdmobilenet_w3d4-1597-0123c031.tf2.h5.log)) |
| FD-MobileNet x1.0 | 33.90 | 13.12 | 2,901,288 | 147.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/fdmobilenet_w1-1312-fa99fb8d.tf2.h5.log)) |
| MobileNetV2 x0.25 | 48.10 | 24.13 | 1,516,392 | 34.24M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/mobilenetv2_wd4-2413-c3705f55.tf2.h5.log)) |
| MobileNetV2 x0.5 | 35.62 | 14.46 | 1,964,736 | 100.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/mobilenetv2_wd2-1446-b0c9a98b.tf2.h5.log)) |
| MobileNetV2 x0.75 | 29.75 | 10.44 | 2,627,592 | 198.50M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/mobilenetv2_w3d4-1044-e122c73e.tf2.h5.log)) |
| MobileNetV2 x1.0 | 26.80 | 8.63 | 3,504,960 | 329.36M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/mobilenetv2_w1-0863-b32cede3.tf2.h5.log)) |
| MobileNetV2b x0.25 | 46.77 | 23.41 | 1,516,312 | 33.18M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.483/mobilenetv2b_wd4-2341-059d9244.tf2.h5.log)) |
| MobileNetV2b x0.5 | 34.26 | 13.75 | 1,964,448 | 96.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.486/mobilenetv2b_wd2-1375-55eb7d49.tf2.h5.log)) |
| MobileNetV2b x0.75 | 30.14 | 10.66 | 2,626,968 | 190.52M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.483/mobilenetv2b_w3d4-1066-bab6a262.tf2.h5.log)) |
| MobileNetV2b x1.0 | 27.16 | 8.91 | 3,503,872 | 315.51M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.483/mobilenetv2b_w1-0891-eabc2c72.tf2.h5.log)) |
| MobileNetV3 L/224/1.0 | 24.36 | 7.32 | 5,481,752 | 226.80M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.491/mobilenetv3_large_w1-0732-2aaed9cc.tf2.h5.log)) |
| IGCV3 x0.25 | 53.38 | 28.28 | 1,534,020 | 41.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/igcv3_wd4-2828-309359dc.tf2.h5.log)) |
| IGCV3 x0.5 | 39.36 | 17.01 | 1,985,528 | 111.12M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/igcv3_wd2-1701-b952333a.tf2.h5.log)) |
| IGCV3 x0.75 | 30.74 | 11.00 | 2,638,084 | 210.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/igcv3_w3d4-1100-00294c7b.tf2.h5.log)) |
| IGCV3 x1.0 | 27.70 | 8.99 | 3,491,688 | 340.79M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/igcv3_w1-0899-a0cb775d.tf2.h5.log)) |
| MnasNet-B1 | 24.70 | 7.22 | 4,383,312 | 326.30M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/mnasnet_b1-0722-61d97108.tf2.h5.log)) |
| MnasNet-A1 | 24.08 | 7.05 | 3,887,038 | 326.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.486/mnasnet_a1-0705-0ea3bd76.tf2.h5.log)) |
| ProxylessNAS CPU | 24.77 | 7.51 | 4,361,648 | 459.96M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.424/proxylessnas_cpu-0751-47e14316.tf2.h5.log)) |
| ProxylessNAS GPU | 24.65 | 7.26 | 7,119,848 | 476.08M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.424/proxylessnas_gpu-0726-d536cb3e.tf2.h5.log)) |
| ProxylessNAS Mobile | 25.29 | 7.83 | 4,080,512 | 332.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.424/proxylessnas_mobile-0783-da8cdb80.tf2.h5.log)) |
| ProxylessNAS Mob-14 | 22.93 | 6.53 | 6,857,568 | 597.10M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.424/proxylessnas_mobile14-0653-478b58cd.tf2.h5.log)) |
| FBNet-Cb | 24.82 | 7.65 | 5,572,200 | 399.26M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.486/fbnet_cb-0765-1f5ffd7c.tf2.h5.log)) |
| Xception | 21.14 | 5.58 | 22,855,952 | 8,403.63M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.423/xception-0558-b95b5051.tf2.h5.log)) |
| InceptionV3 | 21.11 | 5.63 | 23,834,568 | 5,743.06M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/inceptionv3-0563-b0094c1c.tf2.h5.log)) |
| InceptionV4 | 20.78 | 5.41 | 42,679,816 | 12,304.93M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/inceptionv4-0541-c1fa5642.tf2.h5.log)) |
| InceptionResNetV2 | 20.00 | 4.95 | 55,843,464 | 13,188.64M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/inceptionresnetv2-0495-3e2cc545.tf2.h5.log)) |
| PolyNet | 19.09 | 4.51 | 95,366,600 | 34,821.34M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/polynet-0451-e752c86b.tf2.h5.log)) |
| NASNet-A 4@1056 | 25.67 | 8.15 | 5,289,978 | 584.90M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.495/nasnet_4a1056-0815-5b38d08a.tf2.h5.log)) |
| NASNet-A 6@4032 | 18.24 | 4.27 | 88,753,150 | 23,976.44M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/nasnet_6a4032-0427-1f0d2198.tf2.h5.log)) |
| PNASNet-5-Large | 18.02 | 4.27 | 86,057,668 | 25,140.77M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/pnasnet5large-0427-90e804af.tf2.h5.log)) |
| SPNASNet | 25.06 | 7.77 | 4,421,616 | 346.73M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.490/spnasnet-0777-774167df.tf2.h5.log)) |
| EfficientNet-B0 | 24.49 | 7.25 | 5,288,548 | 413.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/efficientnet_b0-0725-fc13925b.tf2.h5.log)) |
| EfficientNet-B1 | 22.93 | 6.30 | 7,794,184 | 730.44M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/efficientnet_b1-0630-82e0c512.tf2.h5.log)) |
| EfficientNet-B0b | 23.05 | 6.68 | 5,288,548 | 413.13M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.429/efficientnet_b0b-0668-77127244.tf2.h5.log)) |
| EfficientNet-B1b | 21.17 | 5.77 | 7,794,184 | 730.44M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.429/efficientnet_b1b-0577-b294ee16.tf2.h5.log)) |
| EfficientNet-B2b | 20.22 | 5.30 | 9,109,994 | 1,049.29M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.429/efficientnet_b2b-0530-55bcdc5d.tf2.h5.log)) |
| EfficientNet-B3b | 19.14 | 4.69 | 12,233,232 | 1,923.98M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.429/efficientnet_b3b-0469-b8210e1a.tf2.h5.log)) |
| EfficientNet-B4b | 17.52 | 3.99 | 19,341,616 | 4,597.56M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.429/efficientnet_b4b-0399-5e35e9c5.tf2.h5.log)) |
| EfficientNet-B5b | 16.43 | 3.43 | 30,389,784 | 10,674.67M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.429/efficientnet_b5b-0343-0ed0c69d.tf2.h5.log)) |
| EfficientNet-B6b | 15.96 | 3.12 | 43,040,704 | 19,761.35M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.429/efficientnet_b6b-0312-faf63104.tf2.h5.log)) |
| EfficientNet-B7b | 15.85 | 3.15 | 66,347,960 | 38,949.07M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.429/efficientnet_b7b-0315-4024912e.tf2.h5.log)) |
| EfficientNet-B0c* | 22.62 | 6.46 | 5,288,548 | 414.31M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b0c-0646-2bd0e2af.tf2.h5.log)) |
| EfficientNet-B1c* | 20.98 | 5.82 | 7,794,184 | 732.54M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b1c-0582-a760b325.tf2.h5.log)) |
| EfficientNet-B2c* | 20.21 | 5.33 | 9,109,994 | 1,051.98M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b2c-0533-ea6ca9cf.tf2.h5.log)) |
| EfficientNet-B3c* | 18.80 | 4.64 | 12,233,232 | 1,928.55M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b3c-0464-1c8fced8.tf2.h5.log)) |
| EfficientNet-B4c* | 17.29 | 3.90 | 19,341,616 | 4,607.46M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b4c-0390-dc4379ea.tf2.h5.log)) |
| EfficientNet-B5c* | 15.87 | 3.10 | 30,389,784 | 10,695.20M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b5c-0310-80258ef7.tf2.h5.log)) |
| EfficientNet-B6c* | 15.29 | 2.86 | 43,040,704 | 19,796.24M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b6c-0286-285f830a.tf2.h5.log)) |
| EfficientNet-B7c* | 14.96 | 2.76 | 66,347,960 | 39,010.98M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b7c-0276-1ffad4ec.tf2.h5.log)) |
| EfficientNet-B8c* | 14.64 | 2.70 | 87,413,142 | 64,541.66M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b8c-0270-aa691b94.tf2.h5.log)) |
| EfficientNet-Edge-Small-b* | 22.66 | 6.42 | 5,438,392 | 2,378.12M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.434/efficientnet_edge_small_b-0642-1c03bb73.tf2.h5.log)) |
| EfficientNet-Edge-Medium-b* | 21.38 | 5.65 | 6,899,496 | 3,700.12M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.434/efficientnet_edge_medium_b-0565-73153b18.tf2.h5.log)) |
| EfficientNet-Edge-Large-b* | 19.86 | 4.96 | 10,589,712 | 9,747.66M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.434/efficientnet_edge_large_b-0496-d72edce1.tf2.h5.log)) |
| MixNet-S | 23.86 | 7.07 | 4,134,606 | 260.26M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/mixnet_s-0707-f8ada6d8.tf2.h5.log)) |
| MixNet-M | 22.39 | 6.32 | 5,014,382 | 366.05M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/mixnet_m-0632-6c91c967.tf2.h5.log)) |
| MixNet-L | 21.44 | 5.56 | 7,329,252 | 590.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.500/mixnet_l-0556-1b72f9aa.tf2.h5.log)) |
| ResNet(A)-10 | 30.89 | 11.61 | 5,438,024 | 1,135.86M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.484/resneta10-1161-208ecb25.tf2.h5.log)) |
| ResNet(A)-BC-14 | 27.72 | 9.60 | 10,084,168 | 1,721.57M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.477/resnetabc14b-0960-96153ace.tf2.h5.log)) |
| ResNet(A)-18 | 25.40 | 8.04 | 11,708,744 | 2,062.24M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.486/resneta18-0804-aa3ba975.tf2.h5.log)) |
| ResNet(A)-50b | 20.79 | 5.38 | 25,576,264 | 4,352.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.492/resneta50b-0538-54936268.tf2.h5.log)) |
| ResNet(A)-101b | 19.62 | 4.88 | 44,568,392 | 8,072.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.511/resneta101b-0488-39d81b95.tf2.h5.log)) |
| ResNet(A)-152b | 19.42 | 4.65 | 60,212,040 | 11,796.83M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.452/resneta152b-0465-a54b896f.tf2.h5.log)) |
| ResNet(D)-50b | 20.80 | 5.49 | 25,680,808 | 20,497.60M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.447/resnetd50b-0549-1c84294f.tf2.h5.log)) |
| ResNet(D)-101b | 19.51 | 4.59 | 44,672,936 | 35,392.65M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.447/resnetd101b-0459-7cce7f13.tf2.h5.log)) |
| ResNet(D)-152b | 19.37 | 4.68 | 60,316,584 | 47,662.18M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.447/resnetd152b-0468-4673f64c.tf2.h5.log)) |

### CIFAR-10

| Model | Error, % | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | --- |
| ResNet-20 | 5.97 | 272,474 | 41.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet20_cifar10-0597-451230e9.tf2.h5.log)) |
| ResNet-56 | 4.52 | 855,770 | 127.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet56_cifar10-0452-a39ad94a.tf2.h5.log)) |
| ResNet-110 | 3.69 | 1,730,714 | 255.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet110_cifar10-0369-c625643a.tf2.h5.log)) |
| ResNet-164(BN) | 3.68 | 1,704,154 | 255.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet164bn_cifar10-0368-cf08cca7.tf2.h5.log)) |
| ResNet-272(BN) | 3.33 | 2,816,986 | 420.61M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet272bn_cifar10-0333-c8b0a926.tf2.h5.log)) |
| ResNet-542(BN) | 3.43 | 5,599,066 | 833.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet542bn_cifar10-0343-c31829d4.tf2.h5.log)) |
| ResNet-1001 | 3.28 | 10,328,602 | 1,536.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet1001_cifar10-0328-552ab287.tf2.h5.log)) |
| ResNet-1202 | 3.53 | 19,424,026 | 2,857.17M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet1202_cifar10-0353-3559a943.tf2.h5.log)) |
| PreResNet-20 | 6.51 | 272,282 | 41.27M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet20_cifar10-0651-d3e7771e.tf2.h5.log)) |
| PreResNet-56 | 4.49 | 855,578 | 127.03M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet56_cifar10-0449-b4bfdaa8.tf2.h5.log)) |
| PreResNet-110 | 3.86 | 1,730,522 | 255.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet110_cifar10-0386-287a4b0c.tf2.h5.log)) |
| PreResNet-164(BN) | 3.64 | 1,703,258 | 255.08M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet164bn_cifar10-0364-29a459fa.tf2.h5.log)) |
| PreResNet-272(BN) | 3.25 | 2,816,090 | 420.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet272bn_cifar10-0325-5bacdc95.tf2.h5.log)) |
| PreResNet-542(BN) | 3.14 | 5,598,170 | 833.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet542bn_cifar10-0314-d8324d47.tf2.h5.log)) |
| PreResNet-1001 | 2.65 | 10,327,706 | 1,536.18M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet1001_cifar10-0265-978844c1.tf2.h5.log)) |
| PreResNet-1202 | 3.39 | 19,423,834 | 2,857.14M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet1202_cifar10-0339-ab04c456.tf2.h5.log)) |
| ResNeXt-20 (1x64d) | 4.33 | 3,446,602 | 538.36M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_1x64d_cifar10-0433-e0ab8667.tf2.h5.log)) |
| ResNeXt-20 (2x32d) | 4.53 | 2,672,458 | 425.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_2x32d_cifar10-0453-7aa966dd.tf2.h5.log)) |
| ResNeXt-20 (4x16d) | 4.70 | 2,285,386 | 368.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_4x16d_cifar10-0470-333e834d.tf2.h5.log)) |
| ResNeXt-20 (8x8d) | 4.66 | 2,091,850 | 340.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_8x8d_cifar10-0466-1dbd9f5e.tf2.h5.log)) |
| ResNeXt-20 (16x4d) | 4.04 | 1,995,082 | 326.10M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_16x4d_cifar10-0404-c6719935.tf2.h5.log)) |
| ResNeXt-20 (32x2d) | 4.61 | 1,946,698 | 319.03M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_32x2d_cifar10-0461-b05d3491.tf2.h5.log)) |
| ResNeXt-20 (64x1d) | 4.93 | 1,922,506 | 315.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_64x1d_cifar10-0493-a13300ce.tf2.h5.log)) |
| ResNeXt-20 (2x64d) | 4.03 | 6,198,602 | 987.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_2x64d_cifar10-0403-367377ed.tf2.h5.log)) |
| ResNeXt-20 (4x32d) | 3.73 | 4,650,314 | 761.57M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_4x32d_cifar10-0373-e4aa1b0d.tf2.h5.log)) |
| ResNeXt-20 (8x16d) | 4.04 | 3,876,170 | 648.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_8x16d_cifar10-0404-5329db5f.tf2.h5.log)) |
| ResNeXt-20 (16x8d) | 3.94 | 3,489,098 | 591.77M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_16x8d_cifar10-0394-cf7c675c.tf2.h5.log)) |
| ResNeXt-20 (32x4d) | 4.20 | 3,295,562 | 563.47M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_32x4d_cifar10-0420-6011e9e9.tf2.h5.log)) |
| ResNeXt-20 (64x2d) | 4.38 | 3,198,794 | 549.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_64x2d_cifar10-0438-3846d7a7.tf2.h5.log)) |
| ResNeXt-56 (1x64d) | 2.87 | 9,317,194 | 1,399.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_1x64d_cifar10-0287-5da5fe18.tf2.h5.log)) |
| ResNeXt-56 (2x32d) | 3.01 | 6,994,762 | 1,059.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_2x32d_cifar10-0301-54d6f2df.tf2.h5.log)) |
| ResNeXt-56 (4x16d) | 3.11 | 5,833,546 | 889.91M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_4x16d_cifar10-0311-766ab89f.tf2.h5.log)) |
| ResNeXt-56 (8x8d) | 3.07 | 5,252,938 | 805.01M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_8x8d_cifar10-0307-685eab39.tf2.h5.log)) |
| ResNeXt-56 (16x4d) | 3.12 | 4,962,634 | 762.56M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_16x4d_cifar10-0312-930e5d5b.tf2.h5.log)) |
| ResNeXt-56 (32x2d) | 3.14 | 4,817,482 | 741.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_32x2d_cifar10-0314-9e387e2e.tf2.h5.log)) |
| ResNeXt-56 (64x1d) | 3.41 | 4,744,906 | 730.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_64x1d_cifar10-0341-bc746947.tf2.h5.log)) |
| ResNeXt-29 (32x4d) | 3.15 | 4,775,754 | 780.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_64x2d_cifar10-0438-3846d7a7.tf2.h5.log)) |
| ResNeXt-29 (16x64d) | 2.41 | 68,155,210 | 10,709.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext29_16x64d_cifar10-0241-712e4744.tf2.h5.log)) |
| ResNeXt-272 (1x64d) | 2.55 | 44,540,746 | 6,565.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext272_1x64d_cifar10-0255-6efe448a.tf2.h5.log)) |
| ResNeXt-272 (2x32d) | 2.74 | 32,928,586 | 4,867.11M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext272_2x32d_cifar10-0274-4e35f994.tf2.h5.log)) |
| SE-ResNet-20 | 6.01 | 274,847 | 41.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet20_cifar10-0601-2f392e4a.tf2.h5.log)) |
| SE-ResNet-56 | 4.13 | 862,889 | 127.19M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet56_cifar10-0413-0224e930.tf2.h5.log)) |
| SE-ResNet-110 | 3.63 | 1,744,952 | 255.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet110_cifar10-0363-4c28f93f.tf2.h5.log)) |
| SE-ResNet-164(BN) | 3.39 | 1,906,258 | 256.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet164bn_cifar10-0339-64d05154.tf2.h5.log)) |
| SE-ResNet-272(BN) | 3.39 | 3,153,826 | 422.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet272bn_cifar10-0339-baa561b6.tf2.h5.log)) |
| SE-ResNet-542(BN) | 3.47 | 6,272,746 | 838.01M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet542bn_cifar10-0347-e95ebdb9.tf2.h5.log)) |
| SE-PreResNet-20 | 6.18 | 274,559 | 41.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet20_cifar10-0618-22217b32.tf2.h5.log)) |
| SE-PreResNet-56 | 4.51 | 862,601 | 127.20M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet56_cifar10-0451-32637db5.tf2.h5.log)) |
| SE-PreResNet-110 | 4.54 | 1,744,664 | 255.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet110_cifar10-0454-e317c569.tf2.h5.log)) |
| SE-PreResNet-164(BN) | 3.73 | 1,904,882 | 256.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet164bn_cifar10-0373-253c0430.tf2.h5.log)) |
| SE-PreResNet-272(BN) | 3.39 | 3,152,450 | 422.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet272bn_cifar10-0339-1ca0bed3.tf2.h5.log)) |
| SE-PreResNet-542(BN) | 3.09 | 6,271,370 | 837.78M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet542bn_cifar10-0309-7764e8bd.tf2.h5.log)) |
| PyramidNet-110 (a=48) | 3.72 | 1,772,706 | 408.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet110_a48_cifar10-0372-3b6ab160.tf2.h5.log)) |
| PyramidNet-110 (a=84) | 2.98 | 3,904,446 | 778.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet110_a84_cifar10-0298-bf303f34.tf2.h5.log)) |
| PyramidNet-110 (a=270) | 2.51 | 28,485,477 | 4,730.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet110_a270_cifar10-0251-983d9983.tf2.h5.log)) |
| PyramidNet-164 (a=270, BN) | 2.42 | 27,216,021 | 4,608.81M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet164_a270_bn_cifar10-0242-aa879193.tf2.h5.log)) |
| PyramidNet-200 (a=240, BN) | 2.44 | 26,752,702 | 4,563.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet200_a240_bn_cifar10-0244-c269bf7d.tf2.h5.log)) |
| PyramidNet-236 (a=220, BN) | 2.47 | 26,969,046 | 4,631.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet236_a220_bn_cifar10-0247-26aac5d0.tf2.h5.log)) |
| PyramidNet-272 (a=200, BN) | 2.39 | 26,210,842 | 4,541.36M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet272_a200_bn_cifar10-0239-b57f64f1.tf2.h5.log)) |
| DenseNet-40 (k=12) | 5.61 | 599,050 | 210.80M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet40_k12_cifar10-0561-e6e20ebf.tf2.h5.log)) |
| DenseNet-BC-40 (k=12) | 6.43 | 176,122 | 74.89M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet40_k12_bc_cifar10-0643-58950791.tf2.h5.log)) |
| DenseNet-BC-40 (k=24) | 4.52 | 690,346 | 293.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet40_k24_bc_cifar10-0452-61a7fe9c.tf2.h5.log)) |
| DenseNet-BC-40 (k=36) | 4.04 | 1,542,682 | 654.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet40_k36_bc_cifar10-0404-ce27624f.tf2.h5.log)) |
| DenseNet-100 (k=12) | 3.66 | 4,068,490 | 1,353.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet100_k12_cifar10-fc483c0b.tf2.h5.log)) |
| DenseNet-100 (k=24) | 3.13 | 16,114,138 | 5,354.19M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet100_k24_cifar10-0313-7f9ee9b3.tf2.h5.log)) |
| DenseNet-BC-100 (k=12) | 4.16 | 769,162 | 298.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet100_k12_bc_cifar10-0416-66beb8fc.tf2.h5.log)) |
| DenseNet-BC-190 (k=40) | 2.52 | 25,624,430 | 9,400.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet190_k40_bc_cifar10-0252-9cc5cfcb.tf2.h5.log)) |
| DenseNet-BC-250 (k=24) | 2.67 | 15,324,406 | 5,519.54M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet250_k24_bc_cifar10-0267-3217a1b3.tf2.h5.log)) |

### CIFAR-100

| Model | Error, % | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | --- |
| ResNet-20 | 29.64 | 278,324 | 41.30M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet20_cifar100-2964-5fa28f78.tf2.h5.log)) |
| ResNet-56 | 24.88 | 861,620 | 127.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet56_cifar100-2488-8e413ab9.tf2.h5.log)) |
| ResNet-110 | 22.80 | 1,736,564 | 255.71M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet110_cifar100-2280-c248211b.tf2.h5.log)) |
| ResNet-164(BN) | 20.44 | 1,727,284 | 255.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet164bn_cifar100-2044-1ba34790.tf2.h5.log)) |
| ResNet-272(BN) | 20.07 | 2,840,116 | 420.63M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet272bn_cifar100-2007-5357e0df.tf2.h5.log)) |
| ResNet-542(BN) | 19.32 | 5,622,196 | 833.89M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet542bn_cifar100-1932-2db913a6.tf2.h5.log)) |
| ResNet-1001 | 19.79 | 10,351,732 | 1,536.43M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet1001_cifar100-1979-75c8acac.tf2.h5.log)) |
| ResNet-1202 | 21.56 | 19,429,876 | 2,857.17M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet1202_cifar100-2156-28fcf786.tf2.h5.log)) |
| PreResNet-20 | 30.22 | 278,132 | 41.28M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet20_cifar100-3022-447255f8.tf2.h5.log)) |
| PreResNet-56 | 25.05 | 861,428 | 127.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet56_cifar100-2505-180fc208.tf2.h5.log)) |
| PreResNet-110 | 22.67 | 1,736,372 | 255.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet110_cifar100-2267-ab677c09.tf2.h5.log)) |
| PreResNet-164(BN) | 20.18 | 1,726,388 | 255.10M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet164bn_cifar100-2018-c7649701.tf2.h5.log)) |
| PreResNet-272(BN) | 19.63 | 2,839,220 | 420.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet272bn_cifar100-1963-22e09198.tf2.h5.log)) |
| PreResNet-542(BN) | 18.71 | 5,621,300 | 833.66M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet542bn_cifar100-1871-703875c6.tf2.h5.log)) |
| PreResNet-1001 | 18.41 | 10,350,836 | 1,536.20M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet1001_cifar100-1841-7481e79c.tf2.h5.log)) |
| ResNeXt-20 (1x64d) | 21.97 | 3,538,852 | 538.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_1x64d_cifar100-2197-413945af.tf2.h5.log)) |
| ResNeXt-20 (2x32d) | 22.55 | 2,764,708 | 425.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_2x32d_cifar100-2255-bf34e56a.tf2.h5.log)) |
| ResNeXt-20 (4x16d) | 23.04 | 2,377,636 | 368.65M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_4x16d_cifar100-2304-fa8d4e06.tf2.h5.log)) |
| ResNeXt-20 (8x8d) | 22.82 | 2,184,100 | 340.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_8x8d_cifar100-2282-51922108.tf2.h5.log)) |
| ResNeXt-20 (16x4d) | 22.82 | 2,087,332 | 326.19M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_16x4d_cifar100-2282-e800aabb.tf2.h5.log)) |
| ResNeXt-20 (32x2d) | 21.73 | 2,038,948 | 319.12M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_32x2d_cifar100-2322-2def8cc2.tf2.h5.log)) |
| ResNeXt-20 (64x1d) | 23.53 | 2,014,756 | 315.58M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_64x1d_cifar100-2353-91695baa.tf2.h5.log)) |
| ResNeXt-20 (2x64d) | 20.60 | 6,290,852 | 988.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_2x64d_cifar100-2060-6eef33bc.tf2.h5.log)) |
| ResNeXt-20 (4x32d) | 21.31 | 4,742,564 | 761.66M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_4x32d_cifar100-2131-edabd5da.tf2.h5.log)) |
| ResNeXt-20 (8x16d) | 21.72 | 3,968,420 | 648.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_8x16d_cifar100-2172-3665fda7.tf2.h5.log)) |
| ResNeXt-20 (16x8d) | 21.73 | 3,581,348 | 591.86M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_16x8d_cifar100-2173-0a330298.tf2.h5.log)) |
| ResNeXt-20 (32x4d) | 22.13 | 3,387,812 | 563.56M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_32x4d_cifar100-2213-9508c15d.tf2.h5.log)) |
| ResNeXt-20 (64x2d) | 22.35 | 3,291,044 | 549.41M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_64x2d_cifar100-2235-e4a559cc.tf2.h5.log)) |
| ResNeXt-56 (1x64d) | 18.25 | 9,409,444 | 1,399.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_1x64d_cifar100-1825-72700951.tf2.h5.log)) |
| ResNeXt-56 (2x32d) | 17.86 | 7,087,012 | 1,059.81M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_2x32d_cifar100-1786-6639c30d.tf2.h5.log)) |
| ResNeXt-56 (4x16d) | 18.09 | 5,925,796 | 890.01M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_4x16d_cifar100-1809-61b41c3b.tf2.h5.log)) |
| ResNeXt-56 (8x8d) | 18.06 | 5,345,188 | 805.10M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_8x8d_cifar100-1806-f3f80382.tf2.h5.log)) |
| ResNeXt-56 (16x4d) | 18.24 | 5,054,884 | 762.65M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_16x4d_cifar100-1824-667ba183.tf2.h5.log)) |
| ResNeXt-56 (32x2d) | 18.60 | 4,909,732 | 741.43M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_32x2d_cifar100-1860-7a236896.tf2.h5.log)) |
| ResNeXt-56 (64x1d) | 18.16 | 4,837,156 | 730.81M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_64x1d_cifar100-1816-06c6c7a0.tf2.h5.log)) |
| ResNeXt-29 (32x4d) | 19.50 | 4,868,004 | 780.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext29_32x4d_cifar100-1950-e9979139.tf2.h5.log)) |
| ResNeXt-29 (16x64d) | 16.93 | 68,247,460 | 10,709.43M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext29_16x64d_cifar100-1693-2df09272.tf2.h5.log)) |
| ResNeXt-272 (1x64d) | 19.11 | 44,632,996 | 6,565.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext272_1x64d_cifar100-1911-e9275c94.tf2.h5.log)) |
| ResNeXt-272 (2x32d) | 18.34 | 33,020,836 | 4,867.20M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext272_2x32d_cifar100-1834-274ef607.tf2.h5.log)) |
| SE-ResNet-20 | 28.54 | 280,697 | 41.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet20_cifar100-2854-598b5858.tf2.h5.log)) |
| SE-ResNet-56 | 22.94 | 868,739 | 127.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet56_cifar100-2294-9c86ec99.tf2.h5.log)) |
| SE-ResNet-110 | 20.86 | 1,750,802 | 255.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet110_cifar100-2086-6435b022.tf2.h5.log)) |
| SE-ResNet-164(BN) | 19.95 | 1,929,388 | 256.57M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet164bn_cifar100-1995-121a777a.tf2.h5.log)) |
| SE-ResNet-272(BN) | 19.07 | 3,176,956 | 422.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet272bn_cifar100-1907-a29e50de.tf2.h5.log)) |
| SE-ResNet-542(BN) | 18.87 | 6,295,876 | 838.03M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet542bn_cifar100-1887-ddc4d5c8.tf2.h5.log)) |
| SE-PreResNet-20 | 28.31 | 280,409 | 41.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet20_cifar100-2831-e8dab8b8.tf2.h5.log)) |
| SE-PreResNet-56 | 23.05 | 868,451 | 127.21M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet56_cifar100-2305-aea4d90b.tf2.h5.log)) |
| SE-PreResNet-110 | 22.61 | 1,750,514 | 255.99M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet110_cifar100-2261-19a8d4a1.tf2.h5.log)) |
| SE-PreResNet-164(BN) | 20.05 | 1,928,012 | 256.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet164bn_cifar100-2005-9c3ed250.tf2.h5.log)) |
| SE-PreResNet-272(BN) | 19.13 | 3,175,580 | 422.47M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet272bn_cifar100-1913-eb75217f.tf2.h5.log)) |
| SE-PreResNet-542(BN) | 19.45 | 6,294,500 | 837.80M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet542bn_cifar100-1945-969d2bf0.tf2.h5.log)) |
| PyramidNet-110 (a=48) | 20.95 | 1,778,556 | 408.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet110_a48_cifar100-2095-3490690a.tf2.h5.log)) |
| PyramidNet-110 (a=84) | 18.87 | 3,913,536 | 778.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet110_a84_cifar100-1887-85789d68.tf2.h5.log)) |
| PyramidNet-110 (a=270) | 17.10 | 28,511,307 | 4,730.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet110_a270_cifar100-1710-cc58021f.tf2.h5.log)) |
| PyramidNet-164 (a=270, BN) | 16.70 | 27,319,071 | 4,608.91M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet164_a270_bn_cifar100-1670-25ddf056.tf2.h5.log)) |
| PyramidNet-200 (a=240, BN) | 16.09 | 26,844,952 | 4,563.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet200_a240_bn_cifar100-1609-d2b16822.tf2.h5.log)) |
| PyramidNet-236 (a=220, BN) | 16.34 | 27,054,096 | 4,631.41M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet236_a220_bn_cifar100-1634-37d5b197.tf2.h5.log)) |
| PyramidNet-272 (a=200, BN) | 16.19 | 26,288,692 | 4,541.43M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet272_a200_bn_cifar100-1619-5c233384.tf2.h5.log)) |
| DenseNet-40 (k=12) | 24.90 | 622,360 | 210.82M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet40_k12_cifar100-2490-ef38ff65.tf2.h5.log)) |
| DenseNet-BC-40 (k=12) | 28.41 | 188,092 | 74.90M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet40_k12_bc_cifar100-2841-c7fbb0f4.tf2.h5.log)) |
| DenseNet-BC-40 (k=24) | 22.67 | 714,196 | 293.11M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet40_k24_bc_cifar100-2267-b3878e82.tf2.h5.log)) |
| DenseNet-BC-40 (k=36) | 20.50 | 1,578,412 | 654.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet40_k36_bc_cifar100-2050-045ae83a.tf2.h5.log)) |
| DenseNet-100 (k=12) | 19.65 | 4,129,600 | 1,353.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet100_k12_cifar100-1965-4f0083d6.tf2.h5.log)) |
| DenseNet-100 (k=24) | 18.08 | 16,236,268 | 5,354.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet100_k24_cifar100-1808-b0842c59.tf2.h5.log)) |
| DenseNet-BC-100 (k=12) | 21.19 | 800,032 | 298.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet100_k12_bc_cifar100-2119-c1b857d5.tf2.h5.log)) |
| DenseNet-BC-250 (k=24) | 17.39 | 15,480,556 | 5,519.69M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet250_k24_bc_cifar100-1739-02d967b5.tf2.h5.log)) |

### SVHN

| Model | Error, % | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | --- |
| ResNet-20 | 3.43 | 272,474 | 41.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet20_svhn-0343-3480eec0.tf2.h5.log)) |
| ResNet-56 | 2.75 | 855,770 | 127.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet56_svhn-0275-5acc5537.tf2.h5.log)) |
| ResNet-110 | 2.45 | 1,730,714 | 255.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet110_svhn-0245-a07e849f.tf2.h5.log)) |
| ResNet-164(BN) | 2.42 | 1,704,154 | 255.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet164bn_svhn-0242-1bfa8083.tf2.h5.log)) |
| ResNet-272(BN) | 2.43 | 2,816,986 | 420.61M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet272bn_svhn-0243-e2a8e355.tf2.h5.log)) |
| ResNet-542(BN) | 2.34 | 5,599,066 | 833.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet542bn_svhn-0234-0d6759e7.tf2.h5.log)) |
| ResNet-1001 | 2.41 | 10,328,602 | 1,536.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet1001_svhn-0241-c9a01550.tf2.h5.log)) |
| PreResNet-20 | 3.22 | 272,282 | 41.27M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet20_svhn-0322-6dcae612.tf2.h5.log)) |
| PreResNet-56 | 2.80 | 855,578 | 127.03M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet56_svhn-0280-6e074c73.tf2.h5.log)) |
| PreResNet-110 | 2.79 | 1,730,522 | 255.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet110_svhn-0279-226a0b34.tf2.h5.log)) |
| PreResNet-164(BN) | 2.58 | 1,703,258 | 255.08M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet164bn_svhn-0258-2307c36f.tf2.h5.log)) |
| PreResNet-272(BN) | 2.34 | 2,816,090 | 420.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet272bn_svhn-0234-3451d5fb.tf2.h5.log)) |
| PreResNet-542(BN) | 2.36 | 5,598,170 | 833.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet542bn_svhn-0236-5ca07592.tf2.h5.log)) |
| ResNeXt-20 (1x64d) | 2.98 | 3,446,602 | 538.36M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_1x64d_svhn-0298-105736c8.tf2.h5.log)) |
| ResNeXt-20 (2x32d) | 2.96 | 2,672,458 | 425.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_2x32d_svhn-0296-b61e1395.tf2.h5.log)) |
| ResNeXt-20 (4x16d) | 3.17 | 2,285,386 | 368.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_4x16d_svhn-0317-cab6d9fd.tf2.h5.log)) |
| ResNeXt-20 (8x8d) | 3.18 | 2,091,850 | 340.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_8x8d_svhn-0318-6ef55252.tf2.h5.log)) |
| ResNeXt-20 (16x4d) | 3.21 | 1,995,082 | 326.10M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_16x4d_svhn-0321-77a670a8.tf2.h5.log)) |
| ResNeXt-20 (32x2d) | 3.27 | 1,946,698 | 319.03M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_32x2d_svhn-0327-0c099194.tf2.h5.log)) |
| ResNeXt-20 (64x1d) | 3.42 | 1,922,506 | 315.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_64x1d_svhn-0342-a3bad459.tf2.h5.log)) |
| ResNeXt-20 (2x64d) | 2.83 | 6,198,602 | 987.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_2x64d_svhn-0283-dedfbac2.tf2.h5.log)) |
| ResNeXt-20 (4x32d) | 2.98 | 4,650,314 | 761.57M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_4x32d_svhn-0298-82b75cbb.tf2.h5.log)) |
| ResNeXt-20 (8x16d) | 3.01 | 3,876,170 | 648.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_8x16d_svhn-0301-d1a547e4.tf2.h5.log)) |
| ResNeXt-20 (16x8d) | 2.93 | 3,489,098 | 591.77M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_16x8d_svhn-0293-4ebac276.tf2.h5.log)) |
| ResNeXt-20 (32x4d) | 3.09 | 3,295,562 | 563.47M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_32x4d_svhn-0309-c8a843e1.tf2.h5.log)) |
| ResNeXt-20 (64x2d) | 3.14 | 3,198,794 | 549.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_64x2d_svhn-0314-c755e25d.tf2.h5.log)) |
| ResNeXt-56 (1x64d) | 2.42 | 9,317,194 | 1,399.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_1x64d_svhn-0242-dd7ac31e.tf2.h5.log)) |
| ResNeXt-56 (2x32d) | 2.46 | 6,994,762 | 1,059.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_2x32d_svhn-0246-61524d8a.tf2.h5.log)) |
| ResNeXt-56 (4x16d) | 2.44 | 5,833,546 | 889.91M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_4x16d_svhn-0244-b7ab2469.tf2.h5.log)) |
| ResNeXt-56 (8x8d) | 2.47 | 5,252,938 | 805.01M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_8x8d_svhn-0247-85692d77.tf2.h5.log)) |
| ResNeXt-56 (16x4d) | 2.56 | 4,962,634 | 762.56M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_16x4d_svhn-0256-86f327a9.tf2.h5.log)) |
| ResNeXt-56 (32x2d) | 2.53 | 4,817,482 | 741.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_32x2d_svhn-0253-b93a0535.tf2.h5.log)) |
| ResNeXt-56 (64x1d) | 2.55 | 4,744,906 | 730.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_64x1d_svhn-0255-9e9e3cc2.tf2.h5.log)) |
| ResNeXt-29 (32x4d) | 2.80 | 4,775,754 | 780.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext29_32x4d_svhn-0280-de6cba99.tf2.h5.log)) |
| ResNeXt-29 (16x64d) | 2.68 | 68,155,210 | 10,709.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext29_16x64d_svhn-0268-c929fada.tf2.h5.log)) |
| ResNeXt-272 (1x64d) | 2.34 | 44,540,746 | 6,565.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext272_1x64d_svhn-0234-4d348e9e.tf2.h5.log)) |
| ResNeXt-272 (2x32d) | 2.44 | 32,928,586 | 4,867.11M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext272_2x32d_svhn-0244-f7923965.tf2.h5.log)) |
| SE-ResNet-20 | 3.23 | 274,847 | 41.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet20_svhn-0323-ef43ce80.tf2.h5.log)) |
| SE-ResNet-56 | 2.64 | 862,889 | 127.19M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet56_svhn-0264-a8fcc570.tf2.h5.log)) |
| SE-ResNet-110 | 2.35 | 1,744,952 | 255.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet110_svhn-0235-57751ac7.tf2.h5.log)) |
| SE-ResNet-164(BN) | 2.45 | 1,906,258 | 256.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet164bn_svhn-0245-a19e2e88.tf2.h5.log)) |
| SE-ResNet-272(BN) | 2.38 | 3,153,826 | 422.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet272bn_svhn-0238-918ee0de.tf2.h5.log)) |
| SE-ResNet-542(BN) | 2.26 | 6,272,746 | 838.01M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet542bn_svhn-0226-5ec784aa.tf2.h5.log)) |
| SE-PreResNet-20 | 3.24 | 274,559 | 41.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet20_svhn-0324-e7dbcc96.tf2.h5.log)) |
| SE-PreResNet-56 | 2.71 | 862,601 | 127.20M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet56_svhn-0271-ea024196.tf2.h5.log)) |
| SE-PreResNet-110 | 2.59 | 1,744,664 | 255.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet110_svhn-0259-6291c548.tf2.h5.log)) |
| SE-PreResNet-164(BN) | 2.56 | 1,904,882 | 256.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet164bn_svhn-0256-c8952322.tf2.h5.log)) |
| SE-PreResNet-272(BN) | 2.49 | 3,152,450 | 422.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet272bn_svhn-0249-0a778e9d.tf2.h5.log)) |
| SE-PreResNet-542(BN) | 2.47 | 6,271,370 | 837.78M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet542bn_svhn-0247-8e242736.tf2.h5.log)) |
| PyramidNet-110 (a=48) | 2.47 | 1,772,706 | 408.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet110_a48_svhn-0247-15827390.tf2.h5.log)) |
| PyramidNet-110 (a=84) | 2.43 | 3,904,446 | 778.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet110_a84_svhn-0243-aacb5f88.tf2.h5.log)) |
| PyramidNet-110 (a=270) | 2.38 | 28,485,477 | 4,730.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet110_a270_svhn-0238-b8742320.tf2.h5.log)) |
| PyramidNet-164 (a=270, BN) | 2.34 | 27,216,021 | 4,608.81M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet164_a270_bn_svhn-0234-94bb4029.tf2.h5.log)) |
| PyramidNet-200 (a=240, BN) | 2.32 | 26,752,702 | 4,563.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet200_a240_bn_svhn-0232-77f2380c.tf2.h5.log)) |
| PyramidNet-236 (a=220, BN) | 2.35 | 26,969,046 | 4,631.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet236_a220_bn_svhn-0235-6a9a8b0a.tf2.h5.log)) |
| PyramidNet-272 (a=200, BN) | 2.40 | 26,210,842 | 4,541.36M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet272_a200_bn_svhn-0240-0a389e2f.tf2.h5.log)) |
| DenseNet-40 (k=12) | 3.05 | 599,050 | 210.80M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet40_k12_svhn-0305-7d5860ae.tf2.h5.log)) |
| DenseNet-BC-40 (k=12) | 3.20 | 176,122 | 74.89M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet40_k12_bc_svhn-0320-77fd3ddf.tf2.h5.log)) |
| DenseNet-BC-40 (k=24) | 2.90 | 690,346 | 293.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet40_k24_bc_svhn-0290-b8a231f7.tf2.h5.log)) |
| DenseNet-BC-40 (k=36) | 2.60 | 1,542,682 | 654.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet40_k36_bc_svhn-0260-a176dcf1.tf2.h5.log)) |
| DenseNet-100 (k=12) | 2.60 | 4,068,490 | 1,353.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet100_k12_svhn-0260-e810c380.tf2.h5.log)) |

### CUB-200-2011

| Model | Error, % | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | --- |
| ResNet-10 | 27.58 | 5,008,392 | 893.63M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.446/resnet10_cub-2758-1a6846b3.tf2.h5.log)) |
| ResNet-12 | 26.68 | 5,082,376 | 1,125.84M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.446/resnet12_cub-2668-03c80736.tf2.h5.log)) |
| ResNet-14 | 24.35 | 5,377,800 | 1,357.53M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.446/resnet14_cub-2435-24b0bfeb.tf2.h5.log)) |
| ResNet-16 | 23.28 | 6,558,472 | 1,588.93M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.446/resnet16_cub-2328-81cc8192.tf2.h5.log)) |
| ResNet-18 | 23.35 | 11,279,112 | 1,820.00M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.446/resnet18_cub-2335-198bdc26.tf2.h5.log)) |
| ResNet-26 | 22.64 | 17,549,832 | 2,746.38M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.446/resnet26_cub-2264-54596784.tf2.h5.log)) |
| SE-ResNet-10 | 27.49 | 5,052,932 | 893.67M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.446/seresnet10_cub-2749-484fc166.tf2.h5.log)) |
| SE-ResNet-12 | 26.11 | 5,127,496 | 1,125.88M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.446/seresnet12_cub-2611-0e5b4e23.tf2.h5.log)) |
| SE-ResNet-14 | 23.75 | 5,425,104 | 1,357.58M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.446/seresnet14_cub-2375-56c26872.tf2.h5.log)) |
| SE-ResNet-16 | 23.21 | 6,614,240 | 1,588.99M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.446/seresnet16_cub-2321-ed3ead79.tf2.h5.log)) |
| SE-ResNet-18 | 23.09 | 11,368,192 | 1,820.10M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.446/seresnet18_cub-2309-f699f05f.tf2.h5.log)) |
| SE-ResNet-26 | 22.58 | 17,683,452 | 2,746.52M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.446/seresnet26_cub-2258-c02ba474.tf2.h5.log)) |
| MobileNet x1.0 | 23.46 | 3,411,976 | 578.98M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.446/mobilenet_w1_cub-2346-b8f24c14.tf2.h5.log)) |
| ProxylessNAS Mobile | 22.02 | 3,055,712 | 331.44M | Training ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.446/proxylessnas_mobile_cub-2202-73ceed5a.tf2.h5.log)) |

### Pascal VOC20102

| Model | Extractor | Pix.Acc.,% | mIoU,% | Params | FLOPs/2 | Remarks |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PSPNet | ResNet(D)-101b | 96.28 | 75.99 | 65,708,501 | 230,771.01M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.448/pspnet_resnetd101b_voc-7599-fbe47bfc.tf2.h5.log)) |
| DeepLabv3 | ResNet(D)-101b | 96.32 | 75.60 | 58,754,773 | 47,625.34M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.448/deeplabv3_resnetd101b_voc-7560-e261b6fd.tf2.h5.log)) |
| DeepLabv3 | ResNet(D)-152b | 96.95 | 77.91 | 74,398,421 | 59,894.87M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.448/deeplabv3_resnetd152b_voc-7791-72038cab.tf2.h5.log)) |
| FCN-8s(d) | ResNet(D)-101b | 97.53 | 80.39 | 52,072,917 | 196,562.96M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.448/fcn8sd_resnetd101b_voc-8039-e140349c.tf2.h5.log)) |

### ADE20K

| Model | Extractor | Pix.Acc.,% | mIoU,% | Params | FLOPs/2 | Remarks |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PSPNet | ResNet(D)-50b | 68.46 | 27.12 | 46,782,550 | 162,595.14M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.450/pspnet_resnetd50b_ade20k-2712-f4fadf0b.tf2.h5.log)) |
| PSPNet | ResNet(D)-101b | 74.76 | 32.59 | 65,774,678 | 231,008.79M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.450/pspnet_resnetd101b_ade20k-3259-ac8569f4.tf2.h5.log)) |
| DeepLabv3 | ResNet(D)-50b | 74.34 | 31.72 | 39,795,798 | 32,756.18M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.450/deeplabv3_resnetd50b_ade20k-3172-2ba069a7.tf2.h5.log)) |
| DeepLabv3 | ResNet(D)-101b | 77.50 | 34.88 | 58,787,926 | 47,651.23M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.450/deeplabv3_resnetd101b_ade20k-3488-08c90933.tf2.h5.log)) |
| FCN-8s(d) | ResNet(D)-50b | 76.70 | 33.10 | 33,146,966 | 128,387.08M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.450/fcn8sd_resnetd50b_ade20k-3310-d440f859.tf2.h5.log)) |
| FCN-8s(d) | ResNet(D)-101b | 78.72 | 35.50 | 52,139,094 | 196,800.73M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.450/fcn8sd_resnetd101b_ade20k-3550-970d968a.tf2.h5.log)) |

### Cityscapes

| Model | Extractor | Pix.Acc.,% | mIoU,% | Params | FLOPs/2 | Remarks |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PSPNet | ResNet(D)-101b | 92.80 | 57.60 | 65,707,475 | 230,767.33M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.449/pspnet_resnetd101b_cityscapes-5760-6dc20af6.tf2.h5.log)) |
| ICNet | ResNet(D)-50b | 95.37 | 60.60 | 47,489,184 | 14,253.43M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.457/icnet_resnetd50b_cityscapes-6060-1e53e1d1.tf2.h5.log)) |
| Fast-SCNN | - | 94.98 | 65.05 | 1,138,051 | 3493.33M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.474/fastscnn_cityscapes-6505-ccc39c9b.tf2.h5.log)) |
| DANet | ResNet(D)-50b | 95.96 | 68.06 | 47,586,427 | 180,397.43M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.468/danet_resnetd50b_cityscapes-6806-c79f5f22.tf2.h5.log)) |
| DANet | ResNet(D)-101b | 96.01 | 67.90 | 66,578,555 | 248,811.08M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.468/danet_resnetd101b_cityscapes-6790-ebd5eef6.tf2.h5.log)) |

### COCO Semantic Segmentation

| Model | Extractor | Pix.Acc.,% | mIoU,% | Params | FLOPs/2 | Remarks |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PSPNet | ResNet(D)-101b | 88.91 | 54.38 | 65,708,501 | 230,771.01M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.451/pspnet_resnetd101b_coco-5438-b64ff2dc.tf2.h5.log)) |
| DeepLabv3 | ResNet(D)-101b | 89.98 | 58.65 | 58,754,773 | 47,625.34M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.451/deeplabv3_resnetd101b_coco-5865-39525a13.tf2.h5.log)) |
| DeepLabv3 | ResNet(D)-152b | 90.40 | 60.67 | 74,398,421 | 275,087.91M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.451/deeplabv3_resnetd152b_coco-6067-f4dabc62.tf2.h5.log)) |
| FCN-8s(d) | ResNet(D)-101b | 91.36 | 59.68 | 52,072,917 | 196,562.96M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.451/fcn8sd_resnetd101b_coco-5968-69c001b3.tf2.h5.log)) |

### CelebAMask-HQ

| Model | Extractor | Params | FLOPs/2 | Remarks |
| --- | --- | ---: | ---: | --- |
| BiSeNet | ResNet-18 | 13,300,416 | - | From [zllrunning/face...Torch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.462/bisenet_resnet18_celebamaskhq-0000-e8799341.log.log)) |

### COCO Keypoints Detection

| Model | Extractor | OKS AP, % | Params | FLOPs/2 | Remarks |
| --- | --- | ---: | ---: | ---: | --- |
| AlphaPose | Fast-SE-ResNet-101b | 74.15/91.59/80.68 | 59,569,873 | 9,553.89M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.454/alphapose_fastseresnet101b_coco-7415-d1f0464a.tf2.h5.log)) |
| SimplePose | ResNet-18 | 66.31/89.20/73.41 | 15,376,721 | 1,799.25M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.455/simplepose_resnet18_coco-6631-4d907c70.tf2.h5.log)) |
| SimplePose | ResNet-50b | 71.02/91.23/78.57 | 33,999,697 | 4,041.06M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.455/simplepose_resnet50b_coco-7102-74506b66.tf2.h5.log)) |
| SimplePose | ResNet-101b | 72.44/92.18/79.76 | 52,991,825 | 7,685.04M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.455/simplepose_resnet101b_coco-7244-6f9e08d6.tf2.h5.log)) |
| SimplePose | ResNet-152b | 72.53/92.14/79.61 | 68,635,473 | 11,332.86M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.455/simplepose_resnet152b_coco-7253-c018fb87.tf2.h5.log)) |
| SimplePose | ResNet(A)-50b | 71.70/91.31/78.66 | 34,018,929 | 4,278.56M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.455/simplepose_resneta50b_coco-7170-c9ddc1c9.tf2.h5.log)) |
| SimplePose | ResNet(A)-101b | 72.97/92.24/80.81 | 53,011,057 | 7,922.54M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.455/simplepose_resneta101b_coco-7297-6db62b71.tf2.h5.log)) |
| SimplePose | ResNet(A)-152b | 73.44/92.27/80.72 | 68,654,705 | 11,570.36M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.455/simplepose_resneta152b_coco-7344-f65954b9.tf2.h5.log)) |
| SimplePose(Mobile) | ResNet-18 | 66.25/89.17/74.32 | 12,858,208 | 1,960.96M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.456/simplepose_mobile_resnet18_coco-6625-8f3e5cc4.tf2.h5.log)) |
| SimplePose(Mobile) | ResNet-50b | 71.10/91.28/78.67 | 25,582,944 | 4,221.30M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.456/simplepose_mobile_resnet50b_coco-7110-e8f61fda.tf2.h5.log)) |
| SimplePose(Mobile) | 1.0 MobileNet-224 | 64.10/88.06/71.23 | 5,019,744 | 751.36M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.456/simplepose_mobile_mobilenet_w1_coco-6410-27c918b9.tf2.h5.log)) |
| SimplePose(Mobile) | 1.0 MobileNetV2b-224 | 63.74/88.12/71.06 | 4,102,176 | 495.95M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.456/simplepose_mobile_mobilenetv2b_w1_coco-6374-4bcc3462.tf2.h5.log)) |
| SimplePose(Mobile) | MobileNetV3 Small 224/1.0 | 54.34/83.67/59.35 | 2,625,088 | 236.51M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.456/simplepose_mobile_mobilenetv3_small_w1_coco-5434-1cfee871.tf2.h5.log)) |
| SimplePose(Mobile) | MobileNetV3 Large 224/1.0 | 63.67/88.91/70.82 | 4,768,336 | 403.97M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.456/simplepose_mobile_mobilenetv3_large_w1_coco-6367-8c8583fb.tf2.h5.log)) |
| Lightweight OpenPose 2D | MobileNet | 39.99/65.95/40.70 | 4,091,698 | 8,948.96M | From [Daniil-Osokin/lighw...ch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.458/lwopenpose2d_mobilenet_cmupan_coco-3999-626b66cb.tf2.h5.log)) |
| Lightweight OpenPose 3D | MobileNet | 39.99/65.95/40.70 | 5,085,983 | 11,049.43M | From [Daniil-Osokin/li...3d...ch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.458/lwopenpose3d_mobilenet_cmupan_coco-3999-df9b1c5f.tf2.h5.log)) |
| IBPPose | - | 64.87/83.62/70.13 | 95,827,784 | 57,195.91M | From [jialee93/Improved...Parts] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.459/ibppose_coco-6487-79500f3d.tf2.h5.log)) |

[dmlc/gluon-cv]: https://github.com/dmlc/gluon-cv
[tornadomeet/ResNet]: https://github.com/tornadomeet/ResNet
[Cadene/pretrained...pytorch]: https://github.com/Cadene/pretrained-models.pytorch
[tensorpack/tensorpack]: https://github.com/tensorpack/tensorpack
[clavichord93/MENet]: https://github.com/clavichord93/MENet
[zeusees/Mnasnet...Model]: https://github.com/zeusees/Mnasnet-Pretrained-Model
[soeaver/mxnet-model]: https://github.com/soeaver/mxnet-model
[rwightman/pyt...models]: https://github.com/rwightman/pytorch-image-models
[soeaver/AirNet-PyTorch]: https://github.com/soeaver/AirNet-PyTorch
[dyhan0920/Pyramid...PyTorch]: https://github.com/dyhan0920/PyramidNet-PyTorch
[szagoruyko/diracnets]: https://github.com/szagoruyko/diracnets
[szagoruyko/functional-zoo]: https://github.com/szagoruyko/functional-zoo
[Jongchan/attention-module]: https://github.com/Jongchan/attention-module
[wielandbrendel/bag...models]: https://github.com/wielandbrendel/bag-of-local-features-models
[fyu/drn]: https://github.com/fyu/drn
[ucbdrive/dla]: https://github.com/ucbdrive/dla
[sacmehta/EdgeNets]: https://github.com/sacmehta/EdgeNets
[XingangPan/IBN-Net]: https://github.com/XingangPan/IBN-Net
[HRNet/HRNet...ation]: https://github.com/HRNet/HRNet-Image-Classification
[stigma0617/VoVNet.pytorch]: https://github.com/stigma0617/VoVNet.pytorch
[PingoLH/Pytorch-HarDNet]: https://github.com/PingoLH/Pytorch-HarDNet
[Daniil-Osokin/lighw...ch]: https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
[Daniil-Osokin/li...3d...ch]: https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch
[jialee93/Improved...Parts]: https://github.com/jialee93/Improved-Body-Parts
[zllrunning/face...Torch]: https://github.com/zllrunning/face-parsing.PyTorch
[MCG-NKU/SCNet]: https://github.com/MCG-NKU/SCNet