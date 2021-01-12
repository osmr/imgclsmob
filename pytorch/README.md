# Computer vision models on PyTorch

[![PyPI](https://img.shields.io/pypi/v/pytorchcv.svg)](https://pypi.python.org/pypi/pytorchcv)
[![Downloads](https://pepy.tech/badge/pytorchcv)](https://pepy.tech/project/pytorchcv)

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
- ResAttNet (['Residual Attention Network for Image Classification'](https://arxiv.org/abs/1704.06904))
- SKNet (['Selective Kernel Networks'](https://arxiv.org/abs/1903.06586))
- SCNet (['Improving Convolutional Networks with Self-Calibrated Convolutions'](http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf))
- RegNet (['Designing Network Design Spaces'](https://arxiv.org/abs/2003.13678))
- DIA-ResNet (['DIANet: Dense-and-Implicit Attention Network'](https://arxiv.org/abs/1905.10671))
- PyramidNet (['Deep Pyramidal Residual Networks'](https://arxiv.org/abs/1610.02915))
- DiracNetV2 (['DiracNets: Training Very Deep Neural Networks Without Skip-Connections'](https://arxiv.org/abs/1706.00388))
- ShaResNet (['ShaResNet: reducing residual network parameter number by sharing weights'](https://arxiv.org/abs/1702.08782))
- DenseNet (['Densely Connected Convolutional Networks'](https://arxiv.org/abs/1608.06993))
- CondenseNet (['CondenseNet: An Efficient DenseNet using Learned Group Convolutions'](https://arxiv.org/abs/1711.09224))
- SparseNet (['Sparsely Aggregated Convolutional Networks'](https://arxiv.org/abs/1801.05895))
- PeleeNet (['Pelee: A Real-Time Object Detection System on Mobile Devices'](https://arxiv.org/abs/1804.06882))
- Oct-ResNet (['Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution'](https://arxiv.org/abs/1904.05049))
- WRN (['Wide Residual Networks'](https://arxiv.org/abs/1605.07146))
- WRN-1bit (['Training wide residual networks for deployment using a single bit for each weight'](https://arxiv.org/abs/1802.08530))
- DRN-C/DRN-D (['Dilated Residual Networks'](https://arxiv.org/abs/1705.09914))
- DPN (['Dual Path Networks'](https://arxiv.org/abs/1707.01629))
- DarkNet Ref/Tiny/19 (['Darknet: Open source neural networks in c'](https://github.com/pjreddie/darknet))
- DarkNet-53 (['YOLOv3: An Incremental Improvement'](https://arxiv.org/abs/1804.02767))
- ChannelNet (['ChannelNets: Compact and Efficient Convolutional Neural Networks via Channel-Wise Convolutions'](https://arxiv.org/abs/1809.01330))
- iSQRT-COV-ResNet (['Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix Square Root Normalization'](https://arxiv.org/abs/1712.01034))
- RevNet (['The Reversible Residual Network: Backpropagation Without Storing Activations'](https://arxiv.org/abs/1707.04585))
- i-RevNet (['i-RevNet: Deep Invertible Networks'](https://arxiv.org/abs/1802.07088))
- BagNet (['Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet'](https://openreview.net/pdf?id=SkfMWhAqYQ))
- DLA (['Deep Layer Aggregation'](https://arxiv.org/abs/1707.06484))
- MSDNet (['Multi-Scale Dense Networks for Resource Efficient Image Classification'](https://arxiv.org/abs/1703.09844))
- FishNet (['FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction'](http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction.pdf))
- ESPNetv2 (['ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network'](https://arxiv.org/abs/1811.11431))
- HRNet (['Deep High-Resolution Representation Learning for Visual Recognition'](https://arxiv.org/abs/1908.07919))
- VoVNet (['An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection'](https://arxiv.org/abs/1904.09730))
- SelecSLS (['XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB Camera'](https://arxiv.org/abs/1907.00837))
- HarDNet (['HarDNet: A Low Memory Traffic Network'](https://arxiv.org/abs/1909.00948))
- X-DenseNet (['Deep Expander Networks: Efficient Deep Networks from Graph Theory'](https://arxiv.org/abs/1711.08757))
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
- DARTS (['DARTS: Differentiable Architecture Search'](https://arxiv.org/abs/1806.09055))
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
- NIN (['Network In Network'](https://arxiv.org/abs/1312.4400))
- RoR-3 (['Residual Networks of Residual Networks: Multilevel Residual Networks'](https://arxiv.org/abs/1608.02908))
- RiR (['Resnet in Resnet: Generalizing Residual Architectures'](https://arxiv.org/abs/1603.08029))
- ResDrop-ResNet (['Deep Networks with Stochastic Depth'](https://arxiv.org/abs/1603.09382))
- Shake-Shake-ResNet (['Shake-Shake regularization'](https://arxiv.org/abs/1705.07485))
- ShakeDrop-ResNet (['ShakeDrop Regularization for Deep Residual Learning'](https://arxiv.org/abs/1802.02375))
- FractalNet (['FractalNet: Ultra-Deep Neural Networks without Residuals'](https://arxiv.org/abs/1605.07648))
- NTS-Net (['Learning to Navigate for Fine-grained Classification'](https://arxiv.org/abs/1809.00287))
- PSPNet (['Pyramid Scene Parsing Network'](https://arxiv.org/abs/1612.01105))
- DeepLabv3 (['Rethinking Atrous Convolution for Semantic Image Segmentation'](https://arxiv.org/abs/1706.05587))
- FCN-8s (['Fully Convolutional Networks for Semantic Segmentation'](https://arxiv.org/abs/1411.4038))
- ICNet (['ICNet for Real-Time Semantic Segmentation on High-Resolution Images'](https://arxiv.org/abs/1704.08545))
- Fast-SCNN (['Fast-SCNN: Fast Semantic Segmentation Network'](https://arxiv.org/abs/1902.04502))
- SINet (['SINet: Extreme Lightweight Portrait Segmentation Networks with Spatial Squeeze Modules and Information Blocking Decoder'](https://arxiv.org/abs/1911.09099))
- BiSeNet (['BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation'](https://arxiv.org/abs/1808.00897))
- DANet (['Dual Attention Network for Scene Segmentation'](https://arxiv.org/abs/1809.02983))
- CenterNet (['Objects as Points'](https://arxiv.org/abs/1904.07850))
- LFFD (['LFFD: A Light and Fast Face Detector for Edge Devices'](https://arxiv.org/abs/1904.10633))
- AlphaPose (['RMPE: Regional Multi-person Pose Estimation'](https://arxiv.org/abs/1612.00137))
- SimplePose (['Simple Baselines for Human Pose Estimation and Tracking'](https://arxiv.org/abs/1804.06208))
- Lightweight OpenPose (['Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose'](https://arxiv.org/abs/1811.12004))
- IBPPose (['Simple Pose: Rethinking and Improving a Bottom-up Approach for Multi-Person Pose Estimation'](https://arxiv.org/abs/1911.10529))
- VOCA (['Capture, Learning, and Synthesis of 3D Speaking Styles'](https://arxiv.org/abs/1905.03079))
- Neural Voice Puppetry Audio-to-Expression net (['Neural Voice Puppetry: Audio-driven Facial Reenactment'](https://arxiv.org/abs/1912.05566))

## Installation

To use the models in your project, simply install the `pytorchcv` package with `torch` (>=0.4.1 is recommended):
```
pip install pytorchcv torch>=0.4.0
```
To enable/disable different hardware supports such as GPUs, check out PyTorch installation [instructions](https://pytorch.org).

## Usage

Example of using a pretrained ResNet-18 model:
```
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from torch.autograd import Variable

net = ptcv_get_model("resnet18", pretrained=True)
x = Variable(torch.randn(1, 3, 224, 224))
y = net(x)
```

## Pretrained models

### ImageNet-1K

Some remarks:
- Top1/Top5 are the standard 1-crop Top-1/Top-5 errors (in percents) on the validation subset of the ImageNet-1K dataset.
- FLOPs/2 is the number of FLOPs divided by two to be similar to the number of MACs.
- Remark `Converted from GL model` means that the model was trained on `MXNet/Gluon` and then converted to PyTorch.
- You may notice that quality estimations are quite different from ones for the corresponding models in other frameworks. This
is due to the fact that the quality is estimated on the standard TorchVision stack of image transformations. Using
OpenCV `Resize` transformation instead of PIL one quality evaluation results will be similar to ones for the Gluon models.
- ResNet(A) is an average downsampled ResNet intended for use as an feature extractor in some pose estimation networks.
- ResNet(D) is a dilated ResNet intended for use as an feature extractor in some segmentation networks.
- Models with *-suffix use non-standard preprocessing (see the training log).

| Model | Top1 | Top5 | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | ---: | --- |
| AlexNet | 38.50 | 16.64 | 62,378,344 | 1,132.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.481/alexnet-1664-2768cdb3.pth.log)) |
| AlexNet-b | 39.74 | 17.47 | 61,100,840 | 714.83M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.485/alexnetb-1747-ac887bf7.pth.log)) |
| ZFNet | 39.79 | 17.27 | 62,357,608 | 1,170.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.395/zfnet-1727-d010ddca.pth.log)) |
| ZFNet-b | 36.37 | 14.90 | 107,627,624 | 2,479.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.400/zfnetb-1490-f6bec24e.pth.log)) |
| VGG-11 | 29.90 | 10.36 | 132,863,336 | 7,615.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.381/vgg11-1036-71e85f6e.pth.log)) |
| VGG-13 | 28.76 | 9.75 | 133,047,848 | 11,317.65M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.388/vgg13-0975-2b2c8770.pth.log)) |
| VGG-16 | 26.98 | 8.65 | 138,357,544 | 15,480.10M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.401/vgg16-0865-5ca155da.pth.log)) |
| VGG-19 | 25.74 | 7.90 | 143,667,240 | 19,642.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.420/vgg19-0790-9bd923a8.pth.log)) |
| BN-VGG-11 | 29.01 | 9.61 | 132,866,088 | 7,630.21M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.339/bn_vgg11-0961-10f01fba.pth.log)) |
| BN-VGG-13 | 27.83 | 9.13 | 133,050,792 | 11,341.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.353/bn_vgg13-0913-b1acd715.pth.log)) |
| BN-VGG-16 | 25.72 | 7.79 | 138,361,768 | 15,506.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.359/bn_vgg16-0779-0f570b92.pth.log)) |
| BN-VGG-19 | 24.13 | 7.12 | 143,672,744 | 19,671.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.360/bn_vgg19-0712-3f286cbd.pth.log)) |
| BN-VGG-11b | 29.56 | 9.96 | 132,868,840 | 7,630.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.407/bn_vgg11b-0996-ef747edc.pth.log)) |
| BN-VGG-13b | 28.47 | 9.24 | 133,053,736 | 11,342.14M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.488/bn_vgg13b-0924-5f313c53.pth.log)) |
| BN-VGG-16b | 25.97 | 7.95 | 138,365,992 | 15,507.20M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.489/bn_vgg16b-0795-bfff365a.pth.log)) |
| BN-VGG-19b | 25.13 | 7.46 | 143,678,248 | 19,672.26M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.490/bn_vgg19b-0746-f523b4e4.pth.log)) |
| BN-Inception | 25.37 | 7.74 | 11,295,240 | 2,048.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.405/bninception-0774-d79ba5f5.pth.log)) |
| ResNet-10 | 32.76 | 12.93 | 5,418,792 | 894.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.483/resnet10-1293-cedc302c.pth.log)) |
| ResNet-12 | 32.10 | 12.23 | 5,492,776 | 1,126.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.485/resnet12-1223-84a43cf6.pth.log)) |
| ResNet-14 | 30.51 | 11.09 | 5,788,200 | 1,357.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.491/resnet14-1109-b3132cbf.pth.log)) |
| ResNet-BC-14b | 29.43 | 10.74 | 10,064,936 | 1,479.12M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.481/resnetbc14b-1074-14b1fd95.pth.log)) |
| ResNet-16 | 28.74 | 10.09 | 6,968,872 | 1,589.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/resnet16-1009-4352d6a9.pth.log)) |
| ResNet-18 x0.25 | 39.62 | 17.85 | 3,937,400 | 270.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.262/resnet18_wd4-1785-fe79b31f.pth.log)) |
| ResNet-18 x0.5 | 33.80 | 13.27 | 5,804,296 | 608.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.263/resnet18_wd2-1327-6654f50a.pth.log)) |
| ResNet-18 x0.75 | 30.40 | 11.06 | 8,476,056 | 1,129.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.266/resnet18_w3d4-1106-3636648b.pth.log)) |
| ResNet-18 | 26.94 | 8.96 | 11,689,512 | 1,820.41M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.478/resnet18-0896-77a56f15.pth.log)) |
| ResNet-26 | 26.16 | 8.49 | 17,960,232 | 2,746.79M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.489/resnet26-0849-4bfbc640.pth.log)) |
| ResNet-BC-26b | 25.09 | 7.97 | 15,995,176 | 2,356.67M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.313/resnetbc26b-0797-7af52a73.pth.log)) |
| ResNet-34 | 24.84 | 7.80 | 21,797,672 | 3,672.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.291/resnet34-0780-3f775482.pth.log)) |
| ResNet-BC-38b | 23.69 | 7.00 | 21,925,416 | 3,234.21M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.328/resnetbc38b-0700-3fbac61d.pth.log)) |
| ResNet-50 | 22.28 | 6.33 | 25,557,032 | 3,877.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.329/resnet50-0633-b00d1c8e.pth.log)) |
| ResNet-50b | 22.39 | 6.38 | 25,557,032 | 4,110.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.308/resnet50b-0638-8a5473ef.pth.log)) |
| ResNet-101 | 21.90 | 6.22 | 44,549,160 | 7,597.95M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.1/resnet101-0622-ab0cf005.pth.log)) |
| ResNet-101b | 20.59 | 5.30 | 44,549,160 | 7,830.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.357/resnet101b-0530-f059ba3c.pth.log)) |
| ResNet-152 | 21.01 | 5.50 | 60,192,808 | 11,321.85M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.144/resnet152-0550-800b2cb1.pth.log)) |
| ResNet-152b | 19.92 | 4.99 | 60,192,808 | 11,554.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.378/resnet152b-0499-667ea926.pth.log)) |
| PreResNet-10 | 35.11 | 14.21 | 5,417,128 | 894.19M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.249/preresnet10-1421-b3973cd4.pth.log)) |
| PreResNet-12 | 33.86 | 13.48 | 5,491,112 | 1,126.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.257/preresnet12-1348-563066fa.pth.log)) |
| PreResNet-14 | 32.64 | 12.39 | 5,786,536 | 1,358.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.260/preresnet14-1239-4be725fd.pth.log)) |
| PreResNet-BC-14b | 31.29 | 11.81 | 10,057,384 | 1,476.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.315/preresnetbc14b-1181-a68d31c3.pth.log)) |
| PreResNet-16 | 30.53 | 11.08 | 6,967,208 | 1,589.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.261/preresnet16-1108-06d8c87e.pth.log)) |
| PreResNet-18 x0.25 | 40.06 | 18.11 | 3,935,960 | 270.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.272/preresnet18_wd4-1811-41135c15.pth.log)) |
| PreResNet-18 x0.5 | 34.00 | 13.40 | 5,802,440 | 608.73M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.273/preresnet18_wd2-1340-c1fe4e31.pth.log)) |
| PreResNet-18 x0.75 | 30.23 | 11.05 | 8,473,784 | 1,129.51M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.274/preresnet18_w3d4-1105-ed2f9ca4.pth.log)) |
| PreResNet-18 | 28.43 | 9.72 | 11,687,848 | 1,820.56M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.140/preresnet18-0972-5651bc2d.pth.log)) |
| PreResNet-26 | 26.33 | 8.51 | 17,958,568 | 2,746.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.316/preresnet26-0851-99e7d6cc.pth.log)) |
| PreResNet-BC-26b | 25.48 | 8.03 | 15,987,624 | 2,354.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.325/preresnetbc26b-0803-d7283bdd.pth.log)) |
| PreResNet-34 | 24.89 | 7.74 | 21,796,008 | 3,672.83M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.300/preresnet34-0774-fd5bd1e8.pth.log)) |
| PreResNet-BC-38b | 22.92 | 6.57 | 21,917,864 | 3,231.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.348/preresnetbc38b-0657-9e523bb9.pth.log)) |
| PreResNet-50 | 22.40 | 6.47 | 25,549,480 | 3,875.44M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.330/preresnet50-0647-222ca73b.pth.log)) |
| PreResNet-50b | 22.51 | 6.55 | 25,549,480 | 4,107.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.307/preresnet50b-0655-8b60378e.pth.log)) |
| PreResNet-101 | 21.74 | 5.91 | 44,541,608 | 7,595.44M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.2/preresnet101-0591-4bacff79.pth.log)) |
| PreResNet-101b | 21.04 | 5.56 | 44,541,608 | 7,827.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.351/preresnet101b-0556-76bfe6d0.pth.log)) |
| PreResNet-152 | 20.94 | 5.55 | 60,185,256 | 11,319.34M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.14/preresnet152-0555-c842a030.pth.log)) |
| PreResNet-152b | 20.14 | 5.16 | 60,185,256 | 11,551.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.386/preresnet152b-0516-f3805f4b.pth.log)) |
| PreResNet-200b | 21.33 | 5.88 | 64,666,280 | 15,068.63M | From [tornadomeet/ResNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.45/preresnet200b-0588-f7104ff3.pth.log)) |
| PreResNet-269b | 20.92 | 5.81 | 102,065,832 | 20,101.11M | From [soeaver/mxnet-model] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.239/preresnet269b-0581-1a7878bb.pth.log)) |
| ResNeXt-14 (16x4d) | 31.94 | 12.48 | 7,127,336 | 1,045.77M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.370/resnext14_16x4d-1248-35ffac2a.pth.log)) |
| ResNeXt-14 (32x2d) | 32.58 | 12.81 | 7,029,416 | 1,031.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.371/resnext14_32x2d-1281-14521186.pth.log)) |
| ResNeXt-14 (32x4d) | 30.32 | 11.46 | 9,411,880 | 1,603.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.327/resnext14_32x4d-1146-89aa6793.pth.log)) |
| ResNeXt-26 (32x2d) | 26.63 | 8.87 | 9,924,136 | 1,461.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.373/resnext26_32x2d-0887-c3bd1307.pth.log)) |
| ResNeXt-26 (32x4d) | 24.14 | 7.46 | 15,389,480 | 2,488.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.332/resnext26_32x4d-0746-1011ac35.pth.log)) |
| ResNeXt-50 (32x4d) | 20.78 | 5.58 | 25,028,904 | 4,255.86M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.417/resnext50_32x4d-0558-b629a522.pth.log)) |
| ResNeXt-101 (32x4d) | 19.98 | 5.23 | 44,177,704 | 8,003.45M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.417/resnext101_32x4d-0523-279a3189.pth.log)) |
| ResNeXt-101 (64x4d) | 19.58 | 5.09 | 83,455,272 | 15,500.27M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.417/resnext101_64x4d-0509-2af0b822.pth.log)) |
| SE-ResNet-10 | 31.76 | 12.02 | 5,463,332 | 894.27M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.486/seresnet10-1202-8dace12e.pth.log)) |
| SE-ResNet-18 | 28.18 | 9.61 | 11,778,592 | 1,820.88M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.355/seresnet18-0961-022123a5.pth.log)) |
| SE-ResNet-26 | 25.67 | 8.24 | 18,093,852 | 2,747.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.363/seresnet26-0824-64fc8759.pth.log)) |
| SE-ResNet-BC-26b | 23.59 | 7.03 | 17,395,976 | 2,359.58M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.366/seresnetbc26b-0703-b98d9d6a.pth.log)) |
| SE-ResNet-BC-38b | 21.60 | 5.95 | 24,026,616 | 3,238.58M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.374/seresnetbc38b-0595-03671c05.pth.log)) |
| SE-ResNet-50 | 21.22 | 5.75 | 28,088,024 | 3,883.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.441/seresnet50-0575-004bfde4.pth.log)) |
| SE-ResNet-50b | 20.79 | 5.39 | 28,088,024 | 4,115.78M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.387/seresnet50b-0539-459e6871.pth.log)) |
| SE-ResNet-101 | 21.88 | 5.89 | 49,326,872 | 7,602.76M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.11/seresnet101-0589-5e6e831b.pth.log)) |
| SE-ResNet-101b | 19.70 | 4.87 | 49,326,872 | 7,839.75M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.460/seresnet101b-0487-b83a20fd.pth.log)) |
| SE-ResNet-152 | 21.48 | 5.76 | 66,821,848 | 11,328.52M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.11/seresnet152-0576-814cf72e.pth.log)) |
| SE-PreResNet-10 | 34.03 | 13.38 | 5,461,668 | 894.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.377/sepreresnet10-1338-935ed560.pth.log)) |
| SE-PreResNet-18 | 28.09 | 9.63 | 11,776,928 | 1,821.03M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.380/sepreresnet18-0963-c065cd9e.pth.log)) |
| SE-PreResNet-BC-26b | 23.22 | 6.60 | 17,388,424 | 2,357.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.399/sepreresnetbc26b-0660-f750b2f5.pth.log)) |
| SE-PreResNet-BC-38b | 21.60 | 5.78 | 24,019,064 | 3,236.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.409/sepreresnetbc38b-0578-12827fcd.pth.log)) |
| SE-PreResNet-50b | 20.85 | 5.49 | 28,080,472 | 4,113.27M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.461/sepreresnet50b-0549-4628a07d.pth.log)) |
| SE-ResNeXt-50 (32x4d) | 20.29 | 5.21 | 27,559,896 | 4,261.16M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.418/seresnext50_32x4d-0521-b0ce2520.pth.log)) |
| SE-ResNeXt-101 (32x4d) | 19.22 | 4.80 | 48,955,416 | 8,012.73M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.418/seresnext101_32x4d-0480-4f6479f0.pth.log)) |
| SE-ResNeXt-101 (64x4d) | 19.28 | 4.76 | 88,232,984 | 15,509.54M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.418/seresnext101_64x4d-0476-da806109.pth.log)) |
| SENet-16 | 25.65 | 8.20 | 31,366,168 | 5,081.30M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.341/senet16-0820-373aeafd.pth.log)) |
| SENet-28 | 21.94 | 5.98 | 36,453,768 | 5,732.71M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.356/senet28-0598-27165b63.pth.log)) |
| SENet-154 | 18.62 | 4.61 | 115,088,984 | 20,745.78M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.13/senet154-0461-6512228c.pth.log)) |
| ResNeSt(A)-BC-14 | 22.46 | 6.47 | 10,611,688 | 2,767.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/resnestabc14-0647-0c3d9e34.pth.log)) |
| ResNeSt(A)-18 | 23.68 | 7.07 | 12,763,784 | 2,587.50M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.489/resnesta18-0707-efca5a69.pth.log)) |
| ResNeSt(A)-BC-26 | 21.52 | 5.71 | 17,069,448 | 3,646.57M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.465/resnestabc26-0571-d6a8a7ae.pth.log)) |
| ResNeSt(A)-50 | 19.04 | 4.62 | 27,483,240 | 5,403.11M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.465/resnesta50-0462-c98fe615.pth.log)) |
| ResNeSt(A)-101 | 17.83 | 4.03 | 48,275,016 | 10,247.88M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.465/resnesta101-0403-61e14773.pth.log)) |
| ResNeSt(A)-200 | 16.87 | 3.39 | 70,201,544 | 22,857.88M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.465/resnesta200-0339-6dc30087.pth.log)) |
| ResNeSt(A)-269 | 16.47 | 3.38 | 110,929,480 | 46,012.95M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.465/resnesta269-0338-6a555ce8.pth.log)) |
| IBN-ResNet-50 | 22.76 | 6.41 | 25,557,032 | 4,110.48M | From [XingangPan/IBN-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.127/ibn_resnet50-0641-e48a1fe5.pth.log)) |
| IBN-ResNet-101 | 21.29 | 5.61 | 44,549,160 | 7,830.48M | From [XingangPan/IBN-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.127/ibn_resnet101-0561-5279c78a.pth.log)) |
| IBN(b)-ResNet-50 | 23.64 | 6.86 | 25,558,568 | 4,112.89M | From [XingangPan/IBN-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.127/ibnb_resnet50-0686-e138995e.pth.log)) |
| IBN-ResNeXt-101 (32x4d) | 20.88 | 5.42 | 44,177,704 | 8,003.45M | From [XingangPan/IBN-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.127/ibn_resnext101_32x4d-0542-b5233c66.pth.log)) |
| IBN-DenseNet-121 | 23.58 | 6.73 | 7,978,856 | 2,872.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/ibn_densenet121-0673-0ea2c535.pth.log)) |
| IBN-DenseNet-169 | 23.25 | 6.51 | 14,149,480 | 3,403.89M | From [XingangPan/IBN-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.127/ibn_densenet169-0651-96dd755e.pth.log)) |
| AirNet50-1x64d (r=2) | 21.84 | 5.90 | 27,425,864 | 4,772.11M | From [soeaver/AirNet-PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.120/airnet50_1x64d_r2-0590-3ec42212.pth.log)) |
| AirNet50-1x64d (r=16) | 22.11 | 6.19 | 25,714,952 | 4,399.97M | From [soeaver/AirNet-PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.120/airnet50_1x64d_r16-0619-090179e7.pth.log)) |
| AirNeXt50-32x4d (r=2) | 20.87 | 5.51 | 27,604,296 | 5,339.58M | From [soeaver/AirNet-PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.120/airnext50_32x4d_r2-0551-c68156e5.pth.log)) |
| BAM-ResNet-50 | 23.14 | 6.58 | 25,915,099 | 4,196.09M | From [Jongchan/attention-module] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.124/bam_resnet50-0658-96a37c82.pth.log)) |
| CBAM-ResNet-50 | 22.38 | 6.05 | 28,089,624 | 4,116.97M | From [Jongchan/attention-module] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.125/cbam_resnet50-0605-a1172fe6.pth.log)) |
| SCNet-50 | 21.22 | 5.47 | 25,564,584 | 3,951.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/scnet50-0547-18741240.pth.log)) |
| SCNet-101 | 21.06 | 5.75 | 44,565,416 | 7,204.24M | From [MCG-NKU/SCNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.472/scnet101-0575-40cd4d4c.pth.log)) |
| SCNet(A)-50 | 19.53 | 4.68 | 25,583,816 | 4,715.84M | From [MCG-NKU/SCNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.472/scneta50-0468-eb3c25d6.pth.log)) |
| RegNetX-200MF | 30.20 | 10.66 | 2,684,792 | 203.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.475/regnetx002-1066-e389d6ce.pth.log)) |
| RegNetX-400MF | 26.44 | 8.66 | 5,157,512 | 403.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.479/regnetx004-0866-9584cc0b.pth.log)) |
| RegNetX-600MF | 24.96 | 7.91 | 6,196,040 | 608.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.482/regnetx006-0791-30ca597a.pth.log)) |
| RegNetX-800MF | 24.21 | 7.40 | 7,259,656 | 809.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.482/regnetx008-0740-157abf5e.pth.log)) |
| RegNetX-1.6GF | 22.44 | 6.37 | 9,190,136 | 1,618.99M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.486/regnetx016-0637-6de8a97b.pth.log)) |
| RegNetX-3.2GF | 21.48 | 5.92 | 15,296,552 | 3,199.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.492/regnetx032-0592-75dc82ab.pth.log)) |
| RegNetX-4.0GF | 21.61 | 5.86 | 22,118,248 | 3,986.29M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.473/regnetx040-0586-54660c9c.pth.log)) |
| RegNetX-6.4GF | 21.06 | 5.57 | 26,209,256 | 6,491.01M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.473/regnetx064-0557-e28df79c.pth.log)) |
| RegNetX-8.0GF | 21.00 | 5.51 | 39,572,648 | 8,017.94M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.473/regnetx080-0551-e8d5baaa.pth.log)) |
| RegNetX-12GF | 20.55 | 5.38 | 46,106,056 | 12,124.22M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.473/regnetx120-0538-5eb7ad44.pth.log)) |
| RegNetX-16GF | 20.07 | 5.17 | 54,278,536 | 15,986.64M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.473/regnetx160-0517-27653d34.pth.log)) |
| RegNetX-32GF | 19.65 | 4.94 | 107,811,560 | 31,790.24M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.473/regnetx320-0494-54a1c651.pth.log)) |
| RegNetY-200MF | 28.66 | 9.80 | 3,162,996 | 203.99M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.476/regnety002-0980-57f04168.pth.log)) |
| RegNetY-400MF | 25.02 | 7.69 | 4,344,144 | 410.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.481/regnety004-0769-8c36573f.pth.log)) |
| RegNetY-600MF | 23.79 | 7.12 | 6,055,160 | 610.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.483/regnety006-0712-d6401a37.pth.log)) |
| RegNetY-800MF | 22.67 | 6.60 | 6,263,168 | 808.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.483/regnety008-0660-ed298c23.pth.log)) |
| RegNetY-1.6GF | 21.41 | 5.81 | 11,202,430 | 1,629.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.486/regnety016-0581-b45eccd6.pth.log)) |
| RegNetY-3.2GF | 18.04 | 4.04 | 19,436,338 | 3,199.15M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.473/regnety032-0404-cb331486.pth.log)) |
| RegNetY-4.0GF | 20.84 | 5.41 | 20,646,656 | 3,999.16M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.473/regnety040-0541-238ef52b.pth.log)) |
| RegNetY-6.4GF | 20.23 | 5.23 | 30,583,252 | 6,388.91M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.473/regnety064-0523-494ac81b.pth.log)) |
| RegNetY-8.0GF | 20.18 | 5.13 | 39,180,068 | 7,996.54M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.473/regnety080-0513-c69743cd.pth.log)) |
| RegNetY-12GF | 19.68 | 4.92 | 51,822,544 | 12,132.55M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.473/regnety120-0492-ba4fb43d.pth.log)) |
| RegNetY-16GF | 19.76 | 5.03 | 83,590,140 | 15,944.53M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.473/regnety160-0503-2c0ad1f9.pth.log)) |
| RegNetY-32GF | 19.32 | 4.74 | 145,046,770 | 32,317.66M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.473/regnety320-0474-643155eb.pth.log)) |
| PyramidNet-101 (a=360) | 21.98 | 6.20 | 42,455,070 | 8,743.54M | From [dyhan0920/Pyramid...PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.104/pyramidnet101_a360-0620-3a24427b.pth.log)) |
| DiracNetV2-18 | 31.47 | 11.70 | 11,511,784 | 1,796.62M | From [szagoruyko/diracnets] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.111/diracnet18v2-1170-e0673770.pth.log)) |
| DiracNetV2-34 | 28.75 | 9.93 | 21,616,232 | 3,646.93M | From [szagoruyko/diracnets] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.111/diracnet34v2-0993-a6a661c0.pth.log)) |
| DenseNet-121 | 23.48 | 7.04 | 7,978,856 | 2,872.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.314/densenet121-0704-cf90d139.pth.log)) |
| DenseNet-161 | 21.91 | 6.06 | 28,681,000 | 7,793.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.432/densenet161-0606-da489277.pth.log)) |
| DenseNet-169 | 22.42 | 6.29 | 14,149,480 | 3,403.89M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.406/densenet169-0629-44974a17.pth.log)) |
| DenseNet-201 | 21.78 | 6.12 | 20,013,928 | 4,347.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.426/densenet201-0612-6adc8625.pth.log)) |
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
| DPN-68 | 23.24 | 6.79 | 12,611,602 | 2,351.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.310/dpn68-0679-a33c98c7.pth.log)) |
| DPN-98 | 20.81 | 5.53 | 61,570,728 | 11,716.51M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.17/dpn98-0553-52c55969.pth.log)) |
| DPN-131 | 20.54 | 5.48 | 79,254,504 | 16,076.15M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.17/dpn131-0548-0c53e5b3.pth.log)) |
| DarkNet Tiny | 40.74 | 17.84 | 1,042,104 | 500.85M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.69/darknet_tiny-1784-4561e1ad.pth.log)) |
| DarkNet Ref | 38.58 | 17.18 | 7,319,416 | 367.59M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.64/darknet_ref-1718-034595b4.pth.log)) |
| DarkNet-53 | 21.75 | 5.64 | 41,609,928 | 7,133.86M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.150/darknet53-0564-b36bef6b.pth.log)) |
| i-RevNet-301 | 25.98 | 8.41 | 125,120,356 | 14,453.87M | From [jhjacobsen/pytorch-i-revnet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.251/irevnet301-0841-95dc8d94.pth.log)) |
| BagNet-9 | 53.61 | 29.61 | 15,688,744 | 16,049.19M | From [wielandbrendel/bag...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.255/bagnet9-2961-cab11792.pth.log)) |
| BagNet-17 | 41.20 | 18.84 | 16,213,032 | 15,768.77M | From [wielandbrendel/bag...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.255/bagnet17-1884-6b2a100f.pth.log)) |
| BagNet-33 | 33.34 | 13.01 | 18,310,184 | 16,371.52M | From [wielandbrendel/bag...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.255/bagnet33-1301-4f17b6e8.pth.log)) |
| DLA-34 | 24.59 | 7.24 | 15,742,104 | 3,071.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.486/dla34-0724-649c67e6.pth.log)) |
| DLA-46-C | 34.28 | 13.23 | 1,301,400 | 585.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.282/dla46c-1323-efcd3636.pth.log)) |
| DLA-X-46-C | 33.26 | 12.69 | 1,068,440 | 546.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.293/dla46xc-1269-00d3754a.pth.log)) |
| DLA-60 | 22.98 | 6.69 | 22,036,632 | 4,255.49M | From [ucbdrive/dla] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.202/dla60-0669-b2cd6e51.pth.log)) |
| DLA-X-60 | 21.07 | 5.75 | 17,352,344 | 3,543.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/dla60x-0575-fae6dc6d.pth.log)) |
| DLA-X-60-C | 30.98 | 10.91 | 1,319,832 | 596.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.289/dla60xc-1091-0f6381f3.pth.log)) |
| DLA-102 | 21.97 | 6.05 | 33,268,888 | 7,190.95M | From [ucbdrive/dla] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.202/dla102-0605-11df1322.pth.log)) |
| DLA-X-102 | 21.49 | 5.77 | 26,309,272 | 5,884.94M | From [ucbdrive/dla] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.202/dla102x-0577-58331655.pth.log)) |
| DLA-X2-102 | 20.55 | 5.36 | 41,282,200 | 9,340.61M | From [ucbdrive/dla] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.202/dla102x2-0536-07936111.pth.log)) |
| DLA-169 | 21.29 | 5.66 | 53,389,720 | 11,593.20M | From [ucbdrive/dla] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.202/dla169-0566-ae0c6a82.pth.log)) |
| FishNet-150 | 21.97 | 6.04 | 24,959,400 | 6,435.05M | From [kevin-ssy/FishNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.168/fishnet150-0604-f5af4873.pth.log)) |
| ESPNetv2 x0.5 | 42.32 | 20.15 | 1,241,332 | 35.36M | From [sacmehta/ESPNetv2] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.238/espnetv2_wd2-2015-d234781f.pth.log)) |
| ESPNetv2 x1.0 | 33.92 | 13.45 | 1,670,072 | 98.09M | From [sacmehta/ESPNetv2] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.238/espnetv2_w1-1345-550d5422.pth.log)) |
| ESPNetv2 x1.25 | 32.06 | 12.18 | 1,965,440 | 138.18M | From [sacmehta/ESPNetv2] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.238/espnetv2_w5d4-1218-85d97b2b.pth.log)) |
| ESPNetv2 x1.5 | 30.83 | 11.29 | 2,314,856 | 185.77M | From [sacmehta/ESPNetv2] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.238/espnetv2_w3d2-1129-3bbb49ad.pth.log)) |
| ESPNetv2 x2.0 | 27.94 | 9.61 | 3,498,136 | 306.93M | From [sacmehta/ESPNetv2] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.238/espnetv2_w2-0961-13ba0f72.pth.log)) |
| HRNet-W18 Small V1 | 26.48 | 9.01 | 13,187,464 | 1,615.00M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.492/hrnet_w18_small_v1-0901-30023064.pth.log)) |
| HRNet-W18 Small V2 | 24.87 | 7.58 | 15,597,464 | 2,618.84M | From [HRNet/HRNet...ation] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.421/hrnet_w18_small_v2-0758-27f85f31.pth.log)) |
| HRNetV2-W18 | 23.24 | 6.56 | 21,299,004 | 4,323.07M | From [HRNet/HRNet...ation] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.421/hrnetv2_w18-0656-78b1f85b.pth.log)) |
| HRNetV2-W30 | 21.80 | 5.78 | 37,712,220 | 8,156.82M | From [HRNet/HRNet...ation] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.421/hrnetv2_w30-0578-839e57eb.pth.log)) |
| HRNetV2-W32 | 21.55 | 5.81 | 41,232,680 | 8,974.04M | From [HRNet/HRNet...ation] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.421/hrnetv2_w32-0581-bef9ada0.pth.log)) |
| HRNetV2-W40 | 21.07 | 5.53 | 57,557,160 | 12,752.26M | From [HRNet/HRNet...ation] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.421/hrnetv2_w40-0553-e4b5a38a.pth.log)) |
| HRNetV2-W44 | 21.11 | 5.63 | 67,064,984 | 14,946.96M | From [HRNet/HRNet...ation] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.421/hrnetv2_w44-0563-9321bfd8.pth.log)) |
| HRNetV2-W48 | 20.69 | 5.48 | 77,469,864 | 17,345.39M | From [HRNet/HRNet...ation] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.421/hrnetv2_w48-0548-40f98610.pth.log)) |
| HRNetV2-W64 | 20.53 | 5.35 | 128,059,944 | 28,976.42M | From [HRNet/HRNet...ation] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.421/hrnetv2_w64-0535-5961efd0.pth.log)) |
| VoVNet-39 | 21.71 | 5.64 | 22,600,296 | 7,086.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/vovnet39-0564-63bfa613.pth.log)) |
| VoVNet-57 | 22.27 | 6.28 | 36,640,296 | 8,943.09M | From [stigma0617/VoVNet.pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.431/vovnet57-0628-99f8a0c8.pth.log)) |
| SelecSLS-42b | 21.93 | 6.11 | 32,458,248 | 2,980.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/selecsls42b-0611-acff1e8b.pth.log)) |
| SelecSLS-60 | 22.10 | 6.12 | 30,670,768 | 3,591.78M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.430/selecsls60-0612-5261403f.pth.log)) |
| SelecSLS-60b | 21.62 | 5.84 | 32,774,064 | 3,629.14M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.430/selecsls60b-0584-470ace6b.pth.log)) |
| HarDNet-39DS | 26.83 | 8.81 | 3,488,228 | 437.52M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.485/hardnet39ds-0881-ea47fc93.pth.log)) |
| HarDNet-68DS | 24.55 | 7.56 | 4,180,602 | 788.86M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.487/hardnet68ds-0756-e0da0750.pth.log)) |
| HarDNet-68 | 23.51 | 6.99 | 17,565,348 | 4,256.32M | From [PingoLH/Pytorch-HarDNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.435/hardnet68-0699-2e207f79.pth.log)) |
| HarDNet-85 | 21.96 | 6.11 | 36,670,212 | 9,088.58M | From [PingoLH/Pytorch-HarDNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.435/hardnet85-0611-ae85d8af.pth.log)) |
| SqueezeNet v1.0 | 39.29 | 17.66 | 1,248,424 | 823.67M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.128/squeezenet_v1_0-1766-afdbcf1a.pth.log)) |
| SqueezeNet v1.1 | 39.31 | 17.72 | 1,235,496 | 352.02M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.88/squeezenet_v1_1-1772-25b77bc3.pth.log)) |
| SqueezeResNet v1.0 | 39.77 | 18.09 | 1,248,424 | 823.67M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.178/squeezeresnet_v1_0-1809-25bfc02e.pth.log)) |
| SqueezeResNet v1.1 | 40.09 | 18.21 | 1,235,496 | 352.02M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.70/squeezeresnet_v1_1-1821-c27ed88f.pth.log)) |
| 1.0-SqNxt-23 | 42.51 | 19.06 | 724,056 | 287.28M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.171/sqnxt23_w1-1906-97b74e0c.pth.log)) |
| 1.0-SqNxt-23v5 | 40.77 | 17.85 | 921,816 | 285.82M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.172/sqnxt23v5_w1-1785-2fe3ad67.pth.log)) |
| 1.5-SqNxt-23 | 34.89 | 13.50 | 1,511,824 | 552.39M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.210/sqnxt23_w3d2-1350-c2f21bce.pth.log)) |
| 1.5-SqNxt-23v5 | 33.81 | 13.01 | 1,953,616 | 550.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.212/sqnxt23v5_w3d2-1301-c244844b.pth.log)) |
| 2.0-SqNxt-23 | 30.62 | 11.00 | 2,583,752 | 898.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.240/sqnxt23_w2-1100-b9bb7302.pth.log)) |
| 2.0-SqNxt-23v5 | 29.63 | 10.66 | 3,366,344 | 897.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.216/sqnxt23v5_w2-1066-229b0d3d.pth.log)) |
| ShuffleNet x0.25 (g=1) | 62.44 | 37.29 | 209,746 | 12.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.134/shufflenet_g1_wd4-3729-47dbd0f2.pth.log)) |
| ShuffleNet x0.25 (g=3) | 61.74 | 36.53 | 305,902 | 13.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.135/shufflenet_g3_wd4-3653-6abdd65e.pth.log)) |
| ShuffleNet x0.5 (g=1) | 46.59 | 22.61 | 534,484 | 41.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.174/shufflenet_g1_wd2-2261-dae4bdad.pth.log)) |
| ShuffleNet x0.5 (g=3) | 44.16 | 20.80 | 718,324 | 41.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.167/shufflenet_g3_wd2-2080-ccaacfc8.pth.log)) |
| ShuffleNet x0.75 (g=1) | 39.58 | 17.11 | 975,214 | 86.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.218/shufflenet_g1_w3d4-1711-161cd24a.pth.log)) |
| ShuffleNet x0.75 (g=3) | 38.20 | 16.50 | 1,238,266 | 85.82M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.219/shufflenet_g3_w3d4-1650-3f3b0aef.pth.log)) |
| ShuffleNet x1.0 (g=1) | 34.93 | 13.89 | 1,531,936 | 148.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.223/shufflenet_g1_w1-1389-4cfb65a3.pth.log)) |
| ShuffleNet x1.0 (g=2) | 34.25 | 13.63 | 1,733,848 | 147.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.241/shufflenet_g2_w1-1363-07256203.pth.log)) |
| ShuffleNet x1.0 (g=3) | 34.39 | 13.48 | 1,865,728 | 145.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.244/shufflenet_g3_w1-1348-ce54f64e.pth.log)) |
| ShuffleNet x1.0 (g=4) | 34.19 | 13.35 | 1,968,344 | 143.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.245/shufflenet_g4_w1-1335-e2415f82.pth.log)) |
| ShuffleNet x1.0 (g=8) | 34.06 | 13.42 | 2,434,768 | 150.76M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.250/shufflenet_g8_w1-1342-9a979b36.pth.log)) |
| ShuffleNetV2 x0.5 | 40.99 | 18.65 | 1,366,792 | 43.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.90/shufflenetv2_wd2-1865-9c22238b.pth.log)) |
| ShuffleNetV2 x1.0 | 31.44 | 11.63 | 2,278,604 | 149.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.133/shufflenetv2_w1-1163-c71dfb7a.pth.log)) |
| ShuffleNetV2 x1.5 | 27.47 | 9.42 | 4,406,098 | 320.77M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.288/shufflenetv2_w3d2-0942-26a92304.pth.log)) |
| ShuffleNetV2 x2.0 | 25.94 | 8.45 | 7,601,686 | 595.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.301/shufflenetv2_w2-0845-337255f6.pth.log)) |
| ShuffleNetV2b x0.5 | 40.29 | 18.22 | 1,366,792 | 43.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.157/shufflenetv2b_wd2-1822-01d18d6f.pth.log)) |
| ShuffleNetV2b x1.0 | 30.62 | 11.25 | 2,279,760 | 150.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.161/shufflenetv2b_w1-1125-6a5d3dc4.pth.log)) |
| ShuffleNetV2b x1.5 | 27.31 | 9.11 | 4,410,194 | 323.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.203/shufflenetv2b_w3d2-0911-f2106fee.pth.log)) |
| ShuffleNetV2b x2.0 | 25.58 | 8.34 | 7,611,290 | 603.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.242/shufflenetv2b_w2-0834-cb36b92c.pth.log)) |
| 108-MENet-8x1 (g=3) | 43.94 | 20.76 | 654,516 | 42.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.89/menet108_8x1_g3-2076-6acc82e4.pth.log)) |
| 128-MENet-8x1 (g=4) | 42.43 | 19.59 | 750,796 | 45.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.103/menet128_8x1_g4-1959-48fa80fc.pth.log)) |
| 160-MENet-8x1 (g=8) | 43.84 | 20.84 | 850,120 | 45.63M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.154/menet160_8x1_g8-2084-0f4fce43.pth.log)) |
| 228-MENet-12x1 (g=3) | 34.11 | 13.16 | 1,806,568 | 152.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.131/menet228_12x1_g3-1316-5b670c42.pth.log)) |
| 256-MENet-12x1 (g=4) | 32.65 | 12.52 | 1,888,240 | 150.65M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.152/menet256_12x1_g4-1252-14c6c86d.pth.log)) |
| 348-MENet-12x1 (g=3) | 28.24 | 9.58 | 3,368,128 | 312.00M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.173/menet348_12x1_g3-0958-ad50f635.pth.log)) |
| 352-MENet-12x1 (g=8) | 31.56 | 12.00 | 2,272,872 | 157.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.198/menet352_12x1_g8-1200-4ee200c5.pth.log)) |
| 456-MENet-24x1 (g=3) | 25.32 | 7.99 | 5,304,784 | 567.90M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.237/menet456_24x1_g3-0799-826c0022.pth.log)) |
| MobileNet x0.25 | 46.26 | 22.49 | 470,072 | 44.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.62/mobilenet_wd4-2249-1ad5e8fe.pth.log)) |
| MobileNet x0.5 | 34.15 | 13.55 | 1,331,592 | 155.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.156/mobilenet_wd2-1355-41a21242.pth.log)) |
| MobileNet x0.75 | 30.14 | 10.76 | 2,585,560 | 333.99M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.130/mobilenet_w3d4-1076-d801bcae.pth.log)) |
| MobileNet x1.0 | 26.61 | 8.95 | 4,231,976 | 579.80M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.155/mobilenet_w1-0895-7e1d739f.pth.log)) |
| MobileNetb x0.25 | 45.51 | 22.01 | 467,592 | 42.88M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.481/mobilenetb_wd4-2201-428da928.pth.log)) |
| MobileNetb x0.5 | 33.42 | 13.10 | 1,326,632 | 153.00M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.480/mobilenetb_wd2-1310-d1549ead.pth.log)) |
| MobileNetb x0.75 | 29.55 | 10.37 | 2,578,120 | 330.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.481/mobilenetb_w3d4-1037-8d732bc9.pth.log)) |
| MobileNetb x1.0 | 25.45 | 8.16 | 4,222,056 | 574.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.489/mobilenetb_w1-0816-107275a1.pth.log)) |
| FD-MobileNet x0.25 | 55.86 | 30.98 | 383,160 | 12.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.177/fdmobilenet_wd4-3098-2b22b709.pth.log)) |
| FD-MobileNet x0.5 | 43.13 | 20.15 | 993,928 | 41.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.83/fdmobilenet_wd2-2015-414dbeed.pth.log)) |
| FD-MobileNet x0.75 | 38.42 | 16.41 | 1,833,304 | 86.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.159/fdmobilenet_w3d4-1641-5561d58a.pth.log)) |
| FD-MobileNet x1.0 | 34.23 | 13.38 | 2,901,288 | 147.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.162/fdmobilenet_w1-1338-9d026c04.pth.log)) |
| MobileNetV2 x0.25 | 48.34 | 24.51 | 1,516,392 | 34.24M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.137/mobilenetv2_wd4-2451-05e1e3a2.pth.log)) |
| MobileNetV2 x0.5 | 35.98 | 14.93 | 1,964,736 | 100.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.170/mobilenetv2_wd2-1493-b82d79f6.pth.log)) |
| MobileNetV2 x0.75 | 30.17 | 10.82 | 2,627,592 | 198.50M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.230/mobilenetv2_w3d4-1082-8656de5a.pth.log)) |
| MobileNetV2 x1.0 | 26.97 | 8.87 | 3,504,960 | 329.36M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.213/mobilenetv2_w1-0887-13a021bc.pth.log)) |
| MobileNetV2b x0.25 | 47.05 | 23.68 | 1,516,312 | 33.18M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.483/mobilenetv2b_wd4-2368-399f95e6.pth.log)) |
| MobileNetV2b x0.5 | 34.77 | 14.08 | 1,964,448 | 96.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.486/mobilenetv2b_wd2-1408-f820ea85.pth.log)) |
| MobileNetV2b x0.75 | 30.74 | 11.05 | 2,626,968 | 190.52M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.483/mobilenetv2b_w3d4-1105-0924efc9.pth.log)) |
| MobileNetV2b x1.0 | 27.70 | 9.12 | 3,503,872 | 315.51M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.483/mobilenetv2b_w1-0912-2bcab1d0.pth.log)) |
| MobileNetV3 L/224/1.0 | 24.55 | 7.44 | 5,481,752 | 227.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.491/mobilenetv3_large_w1-0744-b59cae6d.pth.log)) |
| IGCV3 x0.25 | 53.70 | 28.71 | 1,534,020 | 41.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.142/igcv3_wd4-2871-c9f28301.pth.log)) |
| IGCV3 x0.5 | 39.75 | 17.32 | 1,985,528 | 111.12M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.132/igcv3_wd2-1732-8c504f44.pth.log)) |
| IGCV3 x0.75 | 31.05 | 11.40 | 2,638,084 | 210.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.207/igcv3_w3d4-1140-63f43cf8.pth.log)) |
| IGCV3 x1.0 | 27.91 | 9.20 | 3,491,688 | 340.79M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.243/igcv3_w1-0920-12385791.pth.log)) |
| MnasNet-B1 | 24.96 | 7.40 | 4,383,312 | 326.30M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/mnasnet_b1-0740-7025b43c.pth.log)) |
| MnasNet-A1 | 24.39 | 7.20 | 3,887,038 | 326.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.486/mnasnet_a1-0720-e155916c.pth.log)) |
| DARTS | 25.31 | 7.75 | 4,718,752 | 539.86M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.485/darts-0775-fc3171c5.pth.log)) |
| ProxylessNAS CPU | 24.71 | 7.61 | 4,361,648 | 459.96M | From [MIT-HAN-LAB/ProxylessNAS] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.304/proxylessnas_cpu-0761-fe9572b1.pth.log)) |
| ProxylessNAS GPU | 24.79 | 7.45 | 7,119,848 | 476.08M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.333/proxylessnas_gpu-0745-acca5941.pth.log)) |
| ProxylessNAS Mobile | 25.41 | 7.80 | 4,080,512 | 332.46M | From [MIT-HAN-LAB/ProxylessNAS] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.304/proxylessnas_mobile-0780-639a90c2.pth.log)) |
| ProxylessNAS Mob-14 | 23.29 | 6.62 | 6,857,568 | 597.10M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.331/proxylessnas_mobile14-0662-0c0ad983.pth.log)) |
| FBNet-Cb | 24.89 | 7.62 | 5,572,200 | 399.26M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.415/fbnet_cb-0762-2edb61f8.pth.log)) |
| Xception | 20.97 | 5.49 | 22,855,952 | 8,403.63M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.115/xception-0549-e4f0232c.pth.log)) |
| InceptionV3 | 21.12 | 5.65 | 23,834,568 | 5,743.06M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.92/inceptionv3-0565-cf406180.pth.log)) |
| InceptionV4 | 20.64 | 5.29 | 42,679,816 | 12,304.93M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.105/inceptionv4-0529-5cb7b4e4.pth.log)) |
| InceptionResNetV2 | 19.93 | 4.90 | 55,843,464 | 13,188.64M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.107/inceptionresnetv2-0490-1d1b4d18.pth.log)) |
| PolyNet | 19.10 | 4.52 | 95,366,600 | 34,821.34M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.96/polynet-0452-6a1b295d.pth.log)) |
| NASNet-A 4@1056 | 25.68 | 8.16 | 5,289,978 | 584.90M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.97/nasnet_4a1056-0816-d21bbaf5.pth.log)) |
| NASNet-A 6@4032 | 18.14 | 4.21 | 88,753,150 | 23,976.44M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.101/nasnet_6a4032-0421-f354d28f.pth.log)) |
| PNASNet-5-Large | 17.88 | 4.28 | 86,057,668 | 25,140.77M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.114/pnasnet5large-0428-65de46eb.pth.log)) |
| SPNASNet | 25.41 | 7.98 | 4,421,616 | 346.73M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.490/spnasnet-0798-a25ca157.pth.log)) |
| EfficientNet-B0 | 24.77 | 7.52 | 5,288,548 | 414.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.364/efficientnet_b0-0752-0e386130.pth.log)) |
| EfficientNet-B1 | 23.08 | 6.38 | 7,794,184 | 732.54M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.376/efficientnet_b1-0638-ac77bcd7.pth.log)) |
| EfficientNet-B0b | 23.88 | 7.02 | 5,288,548 | 414.31M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.403/efficientnet_b0b-0702-ecf61b9b.pth.log)) |
| EfficientNet-B1b | 21.60 | 5.94 | 7,794,184 | 732.54M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.403/efficientnet_b1b-0594-614e8166.pth.log)) |
| EfficientNet-B2b | 20.31 | 5.27 | 9,109,994 | 1,051.98M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.403/efficientnet_b2b-0527-531f10e6.pth.log)) |
| EfficientNet-B3b | 18.83 | 4.45 | 12,233,232 | 1,928.55M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.403/efficientnet_b3b-0445-3c5fbba8.pth.log)) |
| EfficientNet-B4b | 17.45 | 3.89 | 19,341,616 | 4,607.46M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.403/efficientnet_b4b-0389-6305bfe6.pth.log)) |
| EfficientNet-B5b | 16.56 | 3.37 | 30,389,784 | 10,695.20M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.403/efficientnet_b5b-0337-e1c2ffcf.pth.log)) |
| EfficientNet-B6b | 16.29 | 3.23 | 43,040,704 | 19,796.24M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.403/efficientnet_b6b-0323-e5c1d7c3.pth.log)) |
| EfficientNet-B7b | 15.94 | 3.22 | 66,347,960 | 39,010.98M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.403/efficientnet_b7b-0322-b9c5965a.pth.log)) |
| EfficientNet-B0c* | 22.92 | 6.75 | 5,288,548 | 414.31M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b0c-0675-21778c6e.pth.log)) |
| EfficientNet-B1c* | 20.73 | 5.69 | 7,794,184 | 732.54M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b1c-0569-239ed6a4.pth.log)) |
| EfficientNet-B2c* | 19.85 | 5.03 | 9,109,994 | 1,051.98M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b2c-0503-be48d3d7.pth.log)) |
| EfficientNet-B3c* | 18.26 | 4.42 | 12,233,232 | 1,928.55M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b3c-0442-ea7080ab.pth.log)) |
| EfficientNet-B4c* | 16.82 | 3.69 | 19,341,616 | 4,607.46M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b4c-0369-5954cc05.pth.log)) |
| EfficientNet-B5c* | 15.91 | 3.10 | 30,389,784 | 10,695.20M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b5c-0310-589fefc6.pth.log)) |
| EfficientNet-B6c* | 15.47 | 2.96 | 43,040,704 | 19,796.24M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b6c-0296-546e61da.pth.log)) |
| EfficientNet-B7c* | 15.13 | 2.88 | 66,347,960 | 39,010.98M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b7c-0288-13d683f2.pth.log)) |
| EfficientNet-B8c* | 14.85 | 2.76 | 87,413,142 | 64,541.66M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b8c-0276-a9973d66.pth.log)) |
| EfficientNet-Edge-Small-b* | 22.74 | 6.40 | 5,438,392 | 2,378.12M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.434/efficientnet_edge_small_b-0640-e27c3444.pth.log)) |
| EfficientNet-Edge-Medium-b* | 21.18 | 5.63 | 6,899,496 | 3,700.12M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.434/efficientnet_edge_medium_b-0563-99fa34c7.pth.log)) |
| EfficientNet-Edge-Large-b* | 19.66 | 4.91 | 10,589,712 | 9,747.66M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.434/efficientnet_edge_large_b-0491-d502326f.pth.log)) |
| MixNet-S | 23.92 | 7.17 | 4,134,606 | 260.76M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/mixnet_s-0717-ab2c4e37.pth.log)) |
| MixNet-M | 22.45 | 6.47 | 5,014,382 | 366.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.493/mixnet_m-0647-4d90d345.pth.log)) |
| MixNet-L | 21.12 | 5.82 | 7,329,252 | 591.34M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.414/mixnet_l-0582-6cf2c975.pth.log)) |
| ResNet(A)-10 | 31.29 | 11.90 | 5,438,024 | 1,135.86M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.484/resneta10-1190-a066e5e0.pth.log)) |
| ResNet(A)-BC-14 | 28.06 | 9.90 | 10,084,168 | 1,721.57M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.477/resnetabc14b-0990-bad51cb0.pth.log)) |
| ResNet(A)-18 | 25.54 | 8.31 | 11,708,744 | 2,062.24M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.486/resneta18-0831-e9f206f4.pth.log)) |
| ResNet(A)-50b | 21.03 | 5.56 | 25,576,264 | 4,352.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.492/resneta50b-0556-7cedbb3b.pth.log)) |
| ResNet(A)-101b | 19.78 | 5.03 | 44,568,392 | 8,072.93M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.452/resneta101b-0503-80d27539.pth.log)) |
| ResNet(A)-152b | 19.62 | 4.82 | 60,212,040 | 11,796.83M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.452/resneta152b-0482-9b55f86f.pth.log)) |
| ResNet(D)-50b | 21.04 | 5.65 | 25,680,808 | 20,497.60M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.296/resnetd50b-0565-ec03d815.pth.log)) |
| ResNet(D)-101b | 19.59 | 4.73 | 44,672,936 | 35,392.65M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.296/resnetd101b-0473-f851c920.pth.log)) |
| ResNet(D)-152b | 19.42 | 4.82 | 60,316,584 | 47,662.18M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.296/resnetd152b-0482-112e216d.pth.log)) |

### CIFAR-10

| Model | Error, % | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | --- |
| NIN | 7.43 | 966,986 | 222.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.175/nin_cifar10-0743-795b0824.pth.log)) |
| ResNet-20 | 5.97 | 272,474 | 41.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.163/resnet20_cifar10-0597-9b0024ac.pth.log)) |
| ResNet-56 | 4.52 | 855,770 | 127.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.163/resnet56_cifar10-0452-628c42a2.pth.log)) |
| ResNet-110 | 3.69 | 1,730,714 | 255.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.163/resnet110_cifar10-0369-4d6ca1fc.pth.log)) |
| ResNet-164(BN) | 3.68 | 1,704,154 | 255.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.179/resnet164bn_cifar10-0368-74ae9f4b.pth.log)) |
| ResNet-272(BN) | 3.33 | 2,816,986 | 420.61M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.368/resnet272bn_cifar10-0333-84f28e0c.pth.log)) |
| ResNet-542(BN) | 3.43 | 5,599,066 | 833.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.369/resnet542bn_cifar10-0343-0fd36dd1.pth.log)) |
| ResNet-1001 | 3.28 | 10,328,602 | 1,536.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.201/resnet1001_cifar10-0328-77a179e2.pth.log)) |
| ResNet-1202 | 3.53 | 19,424,026 | 2,857.17M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.214/resnet1202_cifar10-0353-1d5a2129.pth.log)) |
| PreResNet-20 | 6.51 | 272,282 | 41.27M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.164/preresnet20_cifar10-0651-76cec68d.pth.log)) |
| PreResNet-56 | 4.49 | 855,578 | 127.03M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.164/preresnet56_cifar10-0449-e9124fcf.pth.log)) |
| PreResNet-110 | 3.86 | 1,730,522 | 255.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.164/preresnet110_cifar10-0386-cc08946a.pth.log)) |
| PreResNet-164(BN) | 3.64 | 1,703,258 | 255.08M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.196/preresnet164bn_cifar10-0364-429012d4.pth.log)) |
| PreResNet-272(BN) | 3.25 | 2,816,090 | 420.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.389/preresnet272bn_cifar10-0325-1a6a016e.pth.log)) |
| PreResNet-542(BN) | 3.14 | 5,598,170 | 833.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.391/preresnet542bn_cifar10-0314-66fd6f20.pth.log)) |
| PreResNet-1001 | 2.65 | 10,327,706 | 1,536.18M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.209/preresnet1001_cifar10-0265-9fedfe5f.pth.log)) |
| PreResNet-1202 | 3.39 | 19,423,834 | 2,857.14M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.246/preresnet1202_cifar10-0339-6fc686b0.pth.log)) |
| ResNeXt-29 (32x4d) | 3.15 | 4,775,754 | 780.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.169/resnext29_32x4d_cifar10-0315-30413525.pth.log)) |
| ResNeXt-29 (16x64d) | 2.41 | 68,155,210 | 10,709.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.176/resnext29_16x64d_cifar10-0241-4133d3d0.pth.log)) |
| ResNeXt-272 (1x64d) | 2.55 | 44,540,746 | 6,565.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.372/resnext272_1x64d_cifar10-0255-070ccc35.pth.log)) |
| ResNeXt-272 (2x32d) | 2.74 | 32,928,586 | 4,867.11M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.375/resnext272_2x32d_cifar10-0274-d2ace03c.pth.log)) |
| SE-ResNet-20 | 6.01 | 274,847 | 41.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.362/seresnet20_cifar10-0601-935d8943.pth.log)) |
| SE-ResNet-56 | 4.13 | 862,889 | 127.19M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.362/seresnet56_cifar10-0413-b61c1439.pth.log)) |
| SE-ResNet-110 | 3.63 | 1,744,952 | 255.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.362/seresnet110_cifar10-0363-1ddec230.pth.log)) |
| SE-ResNet-164(BN) | 3.39 | 1,906,258 | 256.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.362/seresnet164bn_cifar10-0339-1085dab6.pth.log)) |
| SE-ResNet-272(BN) | 3.39 | 3,153,826 | 422.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.390/seresnet272bn_cifar10-0339-812db518.pth.log)) |
| SE-ResNet-542(BN) | 3.47 | 6,272,746 | 838.01M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.385/seresnet542bn_cifar10-0347-d1542214.pth.log)) |
| SE-PreResNet-20 | 6.18 | 274,559 | 41.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet20_cifar10-0618-eabb3fce.pth.log)) |
| SE-PreResNet-56 | 4.51 | 862,601 | 127.20M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet56_cifar10-0451-fc23e153.pth.log)) |
| SE-PreResNet-110 | 4.54 | 1,744,664 | 255.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet110_cifar10-0454-418daea9.pth.log)) |
| SE-PreResNet-164(BN) | 3.73 | 1,904,882 | 256.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet164bn_cifar10-0373-ff353a29.pth.log)) |
| SE-PreResNet-272(BN) | 3.39 | 3,152,450 | 422.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet272bn_cifar10-0339-606d0964.pth.log)) |
| SE-PreResNet-542(BN) | 3.08 | 6,271,370 | 837.78M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.382/sepreresnet542bn_cifar10-0308-652bc884.pth.log)) |
| PyramidNet-110 (a=48) | 3.72 | 1,772,706 | 408.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.184/pyramidnet110_a48_cifar10-0372-eb185645.pth.log)) |
| PyramidNet-110 (a=84) | 2.98 | 3,904,446 | 778.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.185/pyramidnet110_a84_cifar10-0298-7b835a3c.pth.log)) |
| PyramidNet-110 (a=270) | 2.51 | 28,485,477 | 4,730.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.194/pyramidnet110_a270_cifar10-0251-31bdd9d5.pth.log)) |
| PyramidNet-164 (a=270, BN) | 2.42 | 27,216,021 | 4,608.81M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.264/pyramidnet164_a270_bn_cifar10-0242-daa2a402.pth.log)) |
| PyramidNet-200 (a=240, BN) | 2.44 | 26,752,702 | 4,563.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.268/pyramidnet200_a240_bn_cifar10-0244-44433afd.pth.log)) |
| PyramidNet-236 (a=220, BN) | 2.47 | 26,969,046 | 4,631.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.285/pyramidnet236_a220_bn_cifar10-0247-daa91d74.pth.log)) |
| PyramidNet-272 (a=200, BN) | 2.39 | 26,210,842 | 4,541.36M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.284/pyramidnet272_a200_bn_cifar10-0239-586b1ecd.pth.log)) |
| DenseNet-40 (k=12) | 5.61 | 599,050 | 210.80M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.193/densenet40_k12_cifar10-0561-8b8e8194.pth.log)) |
| DenseNet-BC-40 (k=12) | 6.43 | 176,122 | 74.89M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.231/densenet40_k12_bc_cifar10-0643-6dc86a2e.pth.log)) |
| DenseNet-BC-40 (k=24) | 4.52 | 690,346 | 293.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.220/densenet40_k24_bc_cifar10-0452-669c5255.pth.log)) |
| DenseNet-BC-40 (k=36) | 4.04 | 1,542,682 | 654.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.224/densenet40_k36_bc_cifar10-0404-b1a4cc7e.pth.log)) |
| DenseNet-100 (k=12) | 3.66 | 4,068,490 | 1,353.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.205/densenet100_k12_cifar10-0366-26089c6e.pth.log)) |
| DenseNet-100 (k=24) | 3.13 | 16,114,138 | 5,354.19M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.252/densenet100_k24_cifar10-0313-397f0e39.pth.log)) |
| DenseNet-BC-100 (k=12) | 4.16 | 769,162 | 298.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.189/densenet100_k12_bc_cifar10-0416-b9232829.pth.log)) |
| DenseNet-BC-190 (k=40) | 2.52 | 25,624,430 | 9,400.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.286/densenet190_k40_bc_cifar10-0252-2896fa08.pth.log)) |
| DenseNet-BC-250 (k=24) | 2.67 | 15,324,406 | 5,519.54M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.290/densenet250_k24_bc_cifar10-0267-f8f9d305.pth.log)) |
| X-DenseNet-BC-40-2 (k=24) | 5.31 | 690,346 | 293.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.226/xdensenet40_2_k24_bc_cifar10-0531-b91a9dc3.pth.log)) |
| X-DenseNet-BC-40-2 (k=36) | 4.37 | 1,542,682 | 654.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.233/xdensenet40_2_k36_bc_cifar10-0437-ed264a20.pth.log)) |
| WRN-16-10 | 2.93 | 17,116,634 | 2,414.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.166/wrn16_10_cifar10-0293-ce810d8a.pth.log)) |
| WRN-28-10 | 2.39 | 36,479,194 | 5,246.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.166/wrn28_10_cifar10-0239-fe97dcd6.pth.log)) |
| WRN-40-8 | 2.37 | 35,748,314 | 5,176.90M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.166/wrn40_8_cifar10-0237-8dc84ec7.pth.log)) |
| WRN-20-10-1bit | 3.26 | 26,737,140 | 4,019.14M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.302/wrn20_10_1bit_cifar10-0326-e6140f8a.pth.log)) |
| WRN-20-10-32bit | 3.14 | 26,737,140 | 4,019.14M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.302/wrn20_10_32bit_cifar10-0314-a18146e8.pth.log)) |
| RoR-3-56 | 5.43 | 762,746 | 113.43M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.228/ror3_56_cifar10-0543-44f0f47d.pth.log)) |
| RoR-3-110 | 4.35 | 1,637,690 | 242.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.235/ror3_110_cifar10-0435-fb2a2b04.pth.log)) |
| RoR-3-164 | 3.93 | 2,512,634 | 370.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.294/ror3_164_cifar10-0393-de7b6dc6.pth.log)) |
| RiR | 3.28 | 9,492,980 | 1,281.08M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.292/rir_cifar10-0328-414c3e60.pth.log)) |
| Shake-Shake-ResNet-20-2x16d | 5.15 | 541,082 | 81.78M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.215/shakeshakeresnet20_2x16d_cifar10-0515-ef71ec0d.pth.log)) |
| Shake-Shake-ResNet-26-2x32d | 3.17 | 2,923,162 | 428.89M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.217/shakeshakeresnet26_2x32d_cifar10-0317-ecd1f833.pth.log)) |
| DIA-ResNet-20 | 6.22 | 286,866 | 41.54M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.340/diaresnet20_cifar10-0622-5e1a02bf.pth.log)) |
| DIA-ResNet-56 | 5.05 | 870,162 | 129.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.340/diaresnet56_cifar10-0505-8ac86804.pth.log)) |
| DIA-ResNet-110 | 4.10 | 1,745,106 | 264.71M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.340/diaresnet110_cifar10-0410-0c00a7da.pth.log)) |
| DIA-ResNet-164(BN) | 3.50 | 1,923,002 | 343.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.340/diaresnet164bn_cifar10-0350-d31f2ebc.pth.log)) |
| DIA-PreResNet-20 | 6.42 | 286,674 | 41.52M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.343/diapreresnet20_cifar10-0642-14a1eb85.pth.log)) |
| DIA-PreResNet-56 | 4.83 | 869,970 | 129.28M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.343/diapreresnet56_cifar10-0483-41cae958.pth.log)) |
| DIA-PreResNet-110 | 4.25 | 1,744,914 | 264.69M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.343/diapreresnet110_cifar10-0425-56385016.pth.log)) |
| DIA-PreResNet-164(BN) | 3.56 | 1,922,106 | 343.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.343/diapreresnet164bn_cifar10-0356-6ec898c8.pth.log)) |

### CIFAR-100

| Model | Error, % | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | --- |
| NIN | 28.39 | 984,356 | 224.08M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.183/nin_cifar100-2839-627a11c0.pth.log)) |
| ResNet-20 | 29.64 | 278,324 | 41.30M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.180/resnet20_cifar100-2964-a5322afe.pth.log)) |
| ResNet-56 | 24.88 | 861,620 | 127.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.181/resnet56_cifar100-2488-d65f53b1.pth.log)) |
| ResNet-110 | 22.80 | 1,736,564 | 255.71M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.190/resnet110_cifar100-2280-d8d397a7.pth.log)) |
| ResNet-164(BN) | 20.44 | 1,727,284 | 255.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.182/resnet164bn_cifar100-2044-8fa07b72.pth.log)) |
| ResNet-272(BN) | 20.07 | 2,840,116 | 420.63M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.368/resnet272bn_cifar100-2007-a80d2b3c.pth.log)) |
| ResNet-542(BN) | 19.32 | 5,622,196 | 833.89M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.369/resnet542bn_cifar100-1932-a631d3ce.pth.log)) |
| ResNet-1001 | 19.79 | 10,351,732 | 1,536.43M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.254/resnet1001_cifar100-1979-2728b558.pth.log)) |
| ResNet-1202 | 21.56 | 19,429,876 | 2,857.17M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.410/resnet1202_cifar100-2156-86ecd091.pth.log)) |
| PreResNet-20 | 30.22 | 278,132 | 41.28M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.187/preresnet20_cifar100-3022-3dbfa6a2.pth.log)) |
| PreResNet-56 | 25.05 | 861,428 | 127.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.188/preresnet56_cifar100-2505-ca90a2be.pth.log)) |
| PreResNet-110 | 22.67 | 1,736,372 | 255.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.191/preresnet110_cifar100-2267-3954e915.pth.log)) |
| PreResNet-164(BN) | 20.18 | 1,726,388 | 255.10M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.192/preresnet164bn_cifar100-2018-a8e67ca6.pth.log)) |
| PreResNet-272(BN) | 19.63 | 2,839,220 | 420.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.389/preresnet272bn_cifar100-1963-6fe0d2e2.pth.log)) |
| PreResNet-542(BN) | 18.71 | 5,621,300 | 833.66M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.391/preresnet542bn_cifar100-1871-07f1fb25.pth.log)) |
| PreResNet-1001 | 18.41 | 10,350,836 | 1,536.20M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.283/preresnet1001_cifar100-1841-88f14ed9.pth.log)) |
| ResNeXt-29 (32x4d) | 19.50 | 4,868,004 | 780.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.200/resnext29_32x4d_cifar100-1950-13ba13d9.pth.log)) |
| ResNeXt-29 (16x64d) | 16.93 | 68,247,460 | 10,709.43M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.322/resnext29_16x64d_cifar100-1693-05e9a7f1.pth.log)) |
| ResNeXt-272 (1x64d) | 19.11 | 44,632,996 | 6,565.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.372/resnext272_1x64d_cifar100-1911-114eb0f8.pth.log)) |
| ResNeXt-272 (2x32d) | 18.34 | 33,020,836 | 4,867.20M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.375/resnext272_2x32d_cifar100-1834-0b30c470.pth.log)) |
| SE-ResNet-20 | 28.54 | 280,697 | 41.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.362/seresnet20_cifar100-2854-8c7abf66.pth.log)) |
| SE-ResNet-56 | 22.94 | 868,739 | 127.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.362/seresnet56_cifar100-2294-7fa54f45.pth.log)) |
| SE-ResNet-110 | 20.86 | 1,750,802 | 255.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.362/seresnet110_cifar100-2086-a82c3093.pth.log)) |
| SE-ResNet-164(BN) | 19.95 | 1,929,388 | 256.57M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.362/seresnet164bn_cifar100-1995-97dd4ab6.pth.log)) |
| SE-ResNet-272(BN) | 19.07 | 3,176,956 | 422.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.390/seresnet272bn_cifar100-1907-179e1c38.pth.log)) |
| SE-ResNet-542(BN) | 18.87 | 6,295,876 | 838.03M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.385/seresnet542bn_cifar100-1887-9c4e7623.pth.log)) |
| SE-PreResNet-20 | 28.31 | 280,409 | 41.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet20_cifar100-2831-fe7558e0.pth.log)) |
| SE-PreResNet-56 | 23.05 | 868,451 | 127.21M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet56_cifar100-2305-c4bdc5d7.pth.log)) |
| SE-PreResNet-110 | 22.61 | 1,750,514 | 255.99M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet110_cifar100-2261-ed7d3c3e.pth.log)) |
| SE-PreResNet-164(BN) | 20.05 | 1,928,012 | 256.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet164bn_cifar100-2005-df1163c4.pth.log)) |
| SE-PreResNet-272(BN) | 19.13 | 3,175,580 | 422.47M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet272bn_cifar100-1913-cb715113.pth.log)) |
| SE-PreResNet-542(BN) | 19.45 | 6,294,500 | 837.80M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.382/sepreresnet542bn_cifar100-1945-9180f863.pth.log)) |
| PyramidNet-110 (a=48) | 20.95 | 1,778,556 | 408.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.186/pyramidnet110_a48_cifar100-2095-95da1a20.pth.log)) |
| PyramidNet-110 (a=84) | 18.87 | 3,913,536 | 778.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.199/pyramidnet110_a84_cifar100-1887-ff711084.pth.log)) |
| PyramidNet-110 (a=270) | 17.10 | 28,511,307 | 4,730.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.319/pyramidnet110_a270_cifar100-1710-7417dd99.pth.log)) |
| PyramidNet-164 (a=270, BN) | 16.70 | 27,319,071 | 4,608.91M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.312/pyramidnet164_a270_bn_cifar100-1670-54d99c83.pth.log)) |
| PyramidNet-200 (a=240, BN) | 16.09 | 26,844,952 | 4,563.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.317/pyramidnet200_a240_bn_cifar100-1609-0729db37.npz.log)) |
| PyramidNet-236 (a=220, BN) | 16.34 | 27,054,096 | 4,631.41M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.312/pyramidnet236_a220_bn_cifar100-1634-a45816eb.pth.log)) |
| PyramidNet-272 (a=200, BN) | 16.19 | 26,288,692 | 4,541.43M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.312/pyramidnet272_a200_bn_cifar100-1619-98bc2f48.pth.log)) |
| DenseNet-40 (k=12) | 24.90 | 622,360 | 210.82M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.195/densenet40_k12_cifar100-2490-d182c224.pth.log)) |
| DenseNet-BC-40 (k=12) | 28.41 | 188,092 | 74.90M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.232/densenet40_k12_bc_cifar100-2841-1e9db765.pth.log)) |
| DenseNet-BC-40 (k=24) | 22.67 | 714,196 | 293.11M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.221/densenet40_k24_bc_cifar100-2267-411719c0.pth.log)) |
| DenseNet-BC-40 (k=36) | 20.50 | 1,578,412 | 654.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.225/densenet40_k36_bc_cifar100-2050-cde836fa.pth.log)) |
| DenseNet-100 (k=12) | 19.64 | 4,129,600 | 1,353.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.206/densenet100_k12_cifar100-1964-5e10cd83.pth.log)) |
| DenseNet-100 (k=24) | 18.08 | 16,236,268 | 5,354.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.318/densenet100_k24_cifar100-1808-1c0a8067.pth.log)) |
| DenseNet-BC-100 (k=12) | 21.19 | 800,032 | 298.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.208/densenet100_k12_bc_cifar100-2119-05a6f027.pth.log)) |
| DenseNet-BC-250 (k=24) | 17.39 | 15,480,556 | 5,519.69M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.303/densenet250_k24_bc_cifar100-1739-09ac3e7d.pth.log)) |
| X-DenseNet-BC-40-2 (k=24) | 23.96 | 714,196 | 293.11M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.227/xdensenet40_2_k24_bc_cifar100-2396-0ce8f78a.pth.log)) |
| X-DenseNet-BC-40-2 (k=36) | 21.65 | 1,578,412 | 654.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.234/xdensenet40_2_k36_bc_cifar100-2165-6f68f83d.pth.log)) |
| WRN-16-10 | 18.95 | 17,174,324 | 2,414.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.204/wrn16_10_cifar100-1895-bef9809c.pth.log)) |
| WRN-28-10 | 17.88 | 36,536,884 | 5,247.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.320/wrn28_10_cifar100-1788-8c3fe818.pth.log)) |
| WRN-40-8 | 18.03 | 35,794,484 | 5,176.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.321/wrn40_8_cifar100-1803-0d18bfbf.pth.log)) |
| WRN-20-10-1bit | 19.04 | 26,794,920 | 4,022.81M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.302/wrn20_10_1bit_cifar100-1904-149860c8.pth.log)) |
| WRN-20-10-32bit | 18.12 | 26,794,920 | 4,022.81M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.302/wrn20_10_32bit_cifar100-1812-70d8972c.pth.log)) |
| RoR-3-56 | 25.49 | 768,596 | 113.43M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.229/ror3_56_cifar100-2549-34be6719.pth.log)) |
| RoR-3-110 | 23.64 | 1,643,540 | 242.08M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.236/ror3_110_cifar100-2364-d599e3a9.pth.log)) |
| RoR-3-164 | 22.34 | 2,518,484 | 370.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.294/ror3_164_cifar100-2234-d37483fc.pth.log)) |
| RiR | 19.23 | 9,527,720 | 1,283.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.292/rir_cifar100-1923-de8ec24a.pth.log)) |
| Shake-Shake-ResNet-20-2x16d | 29.22 | 546,932 | 81.79M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.247/shakeshakeresnet20_2x16d_cifar100-2922-4d07f142.pth.log)) |
| Shake-Shake-ResNet-26-2x32d | 18.80 | 2,934,772 | 428.90M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.222/shakeshakeresnet26_2x32d_cifar100-1880-b47e371f.pth.log)) |
| DIA-ResNet-20 | 27.71 | 292,716 | 41.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.342/diaresnet20_cifar100-2771-28aa1a18.pth.log)) |
| DIA-ResNet-56 | 24.35 | 876,012 | 129.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.342/diaresnet56_cifar100-2435-19085975.pth.log)) |
| DIA-ResNet-110 | 22.11 | 1,750,956 | 264.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.342/diaresnet110_cifar100-2211-7096ddb3.pth.log)) |
| DIA-ResNet-164(BN) | 19.53 | 1,946,132 | 343.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.342/diaresnet164bn_cifar100-1953-b1c474d2.pth.log)) |
| DIA-PreResNet-20 | 28.37 | 292,524 | 41.53M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.343/diapreresnet20_cifar100-2837-f7675c09.pth.log)) |
| DIA-PreResNet-56 | 25.05 | 875,820 | 129.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.343/diapreresnet56_cifar100-2505-5d357985.pth.log)) |
| DIA-PreResNet-110 | 22.69 | 1,750,764 | 264.69M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.343/diapreresnet110_cifar100-2269-c993cc29.pth.log)) |
| DIA-PreResNet-164(BN) | 19.99 | 1,945,236 | 343.39M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.343/diapreresnet164bn_cifar100-1999-00872f98.pth.log)) |

### SVHN

| Model | Error, % | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | --- |
| NIN | 3.76 | 966,986 | 222.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.270/nin_svhn-0376-1205dc06.pth.log)) |
| ResNet-20 | 3.43 | 272,474 | 41.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.265/resnet20_svhn-0343-8232e6e4.pth.log)) |
| ResNet-56 | 2.75 | 855,770 | 127.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.265/resnet56_svhn-0275-6e08ed92.pth.log)) |
| ResNet-110 | 2.45 | 1,730,714 | 255.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.265/resnet110_svhn-0245-c971f0a3.pth.log)) |
| ResNet-164(BN) | 2.42 | 1,704,154 | 255.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.267/resnet164bn_svhn-0242-54941372.pth.log)) |
| ResNet-272(BN) | 2.43 | 2,816,986 | 420.61M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.368/resnet272bn_svhn-0243-ab1d7da5.pth.log)) |
| ResNet-542(BN) | 2.34 | 5,599,066 | 833.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.369/resnet542bn_svhn-0234-04396c97.pth.log)) |
| ResNet-1001 | 2.41 | 10,328,602 | 1,536.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.408/resnet1001_svhn-0241-9e3d4bb5.pth.log)) |
| PreResNet-20 | 3.22 | 272,282 | 41.27M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.269/preresnet20_svhn-0322-c3c00fed.pth.log)) |
| PreResNet-56 | 2.80 | 855,578 | 127.03M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.269/preresnet56_svhn-0280-b51b4147.pth.log)) |
| PreResNet-110 | 2.79 | 1,730,522 | 255.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.269/preresnet110_svhn-0279-aa49e0a3.pth.log)) |
| PreResNet-164(BN) | 2.58 | 1,703,258 | 255.08M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.269/preresnet164bn_svhn-0258-94d42de4.pth.log)) |
| PreResNet-272(BN) | 2.34 | 2,816,090 | 420.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.389/preresnet272bn_svhn-0234-c04ef5c2.pth.log)) |
| PreResNet-542(BN) | 2.36 | 5,598,170 | 833.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.391/preresnet542bn_svhn-0236-6bdf9236.pth.log)) |
| ResNeXt-29 (32x4d) | 2.80 | 4,775,754 | 780.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.275/resnext29_32x4d_svhn-0280-e85c5217.pth.log)) |
| ResNeXt-29 (16x64d) | 2.68 | 68,155,210 | 10,709.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.358/resnext29_16x64d_svhn-0268-74332b71.pth.log)) |
| ResNeXt-272 (1x64d) | 2.35 | 44,540,746 | 6,565.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.372/resnext272_1x64d_svhn-0235-ab044846.pth.log)) |
| ResNeXt-272 (2x32d) | 2.44 | 32,928,586 | 4,867.11M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.375/resnext272_2x32d_svhn-0244-39b8a336.pth.log)) |
| SE-ResNet-20 | 3.23 | 274,847 | 41.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.362/seresnet20_svhn-0323-d77df31c.pth.log)) |
| SE-ResNet-56 | 2.64 | 862,889 | 127.19M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.362/seresnet56_svhn-0264-93839c76.pth.log)) |
| SE-ResNet-110 | 2.35 | 1,744,952 | 255.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.362/seresnet110_svhn-0235-9572ba73.pth.log)) |
| SE-ResNet-164(BN) | 2.45 | 1,906,258 | 256.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.362/seresnet164bn_svhn-0245-af0a90a5.pth.log)) |
| SE-ResNet-272(BN) | 2.38 | 3,153,826 | 422.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.390/seresnet272bn_svhn-0238-0e16bada.pth.log)) |
| SE-ResNet-542(BN) | 2.26 | 6,272,746 | 838.01M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.385/seresnet542bn_svhn-0226-71a8f298.pth.log)) |
| SE-PreResNet-20 | 3.24 | 274,559 | 41.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet20_svhn-0324-061daa58.pth.log)) |
| SE-PreResNet-56 | 2.71 | 862,601 | 127.20M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet56_svhn-0271-c91e922f.pth.log)) |
| SE-PreResNet-110 | 2.59 | 1,744,664 | 255.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet110_svhn-0259-556909fd.pth.log)) |
| SE-PreResNet-164(BN) | 2.56 | 1,904,882 | 256.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet164bn_svhn-0256-f8dd4e06.pth.log)) |
| SE-PreResNet-272(BN) | 2.49 | 3,152,450 | 422.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.379/sepreresnet272bn_svhn-0249-904d74a2.pth.log)) |
| SE-PreResNet-542(BN) | 2.47 | 6,271,370 | 837.78M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.382/sepreresnet542bn_svhn-0247-318a8325.pth.log)) |
| PyramidNet-110 (a=48) | 2.47 | 1,772,706 | 408.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.281/pyramidnet110_a48_svhn-0247-d48bafbe.pth.log)) |
| PyramidNet-110 (a=84) | 2.43 | 3,904,446 | 778.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.392/pyramidnet110_a84_svhn-0243-971576c6.pth.log)) |
| PyramidNet-110 (a=270) | 2.38 | 28,485,477 | 4,730.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.393/pyramidnet110_a270_svhn-0238-3047a9bb.pth.log)) |
| PyramidNet-164 (a=270, BN) | 2.33 | 27,216,021 | 4,608.81M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.396/pyramidnet164_a270_bn_svhn-0233-42d4c033.pth.log)) |
| PyramidNet-200 (a=240, BN) | 2.32 | 26,752,702 | 4,563.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.397/pyramidnet200_a240_bn_svhn-0232-f9660c25.pth.log)) |
| PyramidNet-236 (a=220, BN) | 2.35 | 26,969,046 | 4,631.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.398/pyramidnet236_a220_bn_svhn-0235-f74fe248.pth.log)) |
| PyramidNet-272 (a=200, BN) | 2.40 | 26,210,842 | 4,541.36M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.404/pyramidnet272_a200_bn_svhn-0240-96f6e740.pth.log)) |
| DenseNet-40 (k=12) | 3.05 | 599,050 | 210.80M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.278/densenet40_k12_svhn-0305-ac0de84a.pth.log)) |
| DenseNet-BC-40 (k=12) | 3.20 | 176,122 | 74.89M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.279/densenet40_k12_bc_svhn-0320-32076052.pth.log)) |
| DenseNet-BC-40 (k=24) | 2.90 | 690,346 | 293.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.280/densenet40_k24_bc_svhn-0290-f4440d3b.pth.log)) |
| DenseNet-BC-40 (k=36) | 2.60 | 1,542,682 | 654.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.311/densenet40_k36_bc_svhn-0260-8c7db0a2.pth.log)) |
| DenseNet-100 (k=12) | 2.60 | 4,068,490 | 1,353.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.311/densenet100_k12_svhn-0260-57fde50e.pth.log)) |
| X-DenseNet-BC-40-2 (k=24) | 2.87 | 690,346 | 293.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.306/xdensenet40_2_k24_bc_svhn-0287-fd9b6def.pth.log)) |
| X-DenseNet-BC-40-2 (k=36) | 2.74 | 1,542,682 | 654.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.306/xdensenet40_2_k36_bc_svhn-0274-540a69f1.pth.log)) |
| WRN-16-10 | 2.78 | 17,116,634 | 2,414.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.271/wrn16_10_svhn-0278-5ab2a4ed.pth.log)) |
| WRN-28-10 | 2.71 | 36,479,194 | 5,246.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.276/wrn28_10_svhn-0271-d62b6bba.pth.log)) |
| WRN-40-8 | 2.54 | 35,748,314 | 5,176.90M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.277/wrn40_8_svhn-0254-dee59602.pth.log)) |
| WRN-20-10-1bit | 2.73 | 26,737,140 | 4,019.14M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.302/wrn20_10_1bit_svhn-0273-ffe96cb7.pth.log)) |
| WRN-20-10-32bit | 2.59 | 26,737,140 | 4,019.14M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.302/wrn20_10_32bit_svhn-0259-ce402a58.pth.log)) |
| RoR-3-56 | 2.69 | 762,746 | 113.43M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.287/ror3_56_svhn-0269-5a9ad66c.pth.log)) |
| RoR-3-110 | 2.57 | 1,637,690 | 242.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.287/ror3_110_svhn-0257-155380ad.pth.log)) |
| RoR-3-164 | 2.73 | 2,512,634 | 370.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.294/ror3_164_svhn-0273-ff0d9af0.pth.log)) |
| RiR | 2.68 | 9,492,980 | 1,281.08M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.292/rir_svhn-0268-12fcbd3b.pth.log)) |
| Shake-Shake-ResNet-20-2x16d | 3.17 | 541,082 | 81.78M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.295/shakeshakeresnet20_2x16d_svhn-0317-a693ec24.pth.log)) |
| Shake-Shake-ResNet-26-2x32d | 2.62 | 2,923,162 | 428.89M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.295/shakeshakeresnet26_2x32d_svhn-0262-c1b8099e.pth.log)) |
| DIA-ResNet-20 | 3.23 | 286,866 | 41.54M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.340/diaresnet20_svhn-0323-b8ee92c9.pth.log)) |
| DIA-ResNet-56 | 2.68 | 870,162 | 129.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.340/diaresnet56_svhn-0268-bd2ec755.pth.log)) |
| DIA-ResNet-110 | 2.47 | 1,745,106 | 264.71M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.340/diaresnet110_svhn-0247-635e42cf.pth.log)) |
| DIA-ResNet-164(BN) | 2.44 | 1,923,002 | 343.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.340/diaresnet164bn_svhn-0244-0b8f6713.pth.log)) |
| DIA-PreResNet-20 | 3.03 | 286,674 | 41.52M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.343/diapreresnet20_svhn-0303-dc3e3a45.pth.log)) |
| DIA-PreResNet-56 | 2.80 | 869,970 | 129.28M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.343/diapreresnet56_svhn-0280-537ebc66.pth.log)) |
| DIA-PreResNet-110 | 2.42 | 1,744,914 | 264.69M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.343/diapreresnet110_svhn-0242-a156cfb5.pth.log)) |
| DIA-PreResNet-164(BN) | 2.56 | 1,922,106 | 343.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.343/diapreresnet164bn_svhn-0256-13404881.pth.log)) |

### CUB-200-2011

| Model | Error, % | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | --- |
| ResNet-10 | 27.77 | 5,008,392 | 893.63M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.335/resnet10_cub-2777-4525b593.pth.log)) |
| ResNet-12 | 27.27 | 5,082,376 | 1,125.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.336/resnet12_cub-2727-c1524883.pth.log)) |
| ResNet-14 | 24.77 | 5,377,800 | 1,357.53M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.337/resnet14_cub-2477-5051bbc6.pth.log)) |
| ResNet-16 | 23.65 | 6,558,472 | 1,588.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.338/resnet16_cub-2365-b831356c.pth.log)) |
| ResNet-18 | 23.33 | 11,279,112 | 1,820.00M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.344/resnet18_cub-2333-200d8b9c.pth.log)) |
| ResNet-26 | 23.16 | 17,549,832 | 2,746.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.345/resnet26_cub-2316-599ab467.pth.log)) |
| SE-ResNet-10 | 27.72 | 5,052,932 | 893.86M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.361/seresnet10_cub-2772-f52526ec.pth.log)) |
| SE-ResNet-12 | 26.51 | 5,127,496 | 1,126.17M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.361/seresnet12_cub-2651-5c0e7f83.pth.log)) |
| SE-ResNet-14 | 24.16 | 5,425,104 | 1,357.92M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.361/seresnet14_cub-2416-a4cda901.pth.log)) |
| SE-ResNet-16 | 23.32 | 6,614,240 | 1,589.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.361/seresnet16_cub-2332-43a819b7.pth.log)) |
| SE-ResNet-18 | 23.52 | 11,368,192 | 1,820.47M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.361/seresnet18_cub-2352-414fa277.pth.log)) |
| SE-ResNet-26 | 22.99 | 17,683,452 | 2,747.08M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.361/seresnet26_cub-2299-5aa0a7d1.pth.log)) |
| MobileNet x1.0 | 23.77 | 3,411,976 | 578.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.346/mobilenet_w1_cub-2377-8428471f.pth.log)) |
| ProxylessNAS Mobile | 22.66 | 3,055,712 | 331.44M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.347/proxylessnas_mobile_cub-2266-e4b5098a.pth.log)) |
| NTS-Net | 12.77 | 28,623,333 | 33,361.79M | From [yangze0930/NTS-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.334/ntsnet_cub-1277-f6f330ab.pth.log)) |

### Pascal VOC20102

| Model | Extractor | Pix.Acc.,% | mIoU,% | Params | FLOPs/2 | Remarks |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PSPNet | ResNet(D)-101b | 98.09 | 81.44 | 65,708,501 | 230,771.01M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.297/pspnet_resnetd101b_voc-8144-c22f0219.pth.log)) |
| DeepLabv3 | ResNet(D)-101b | 97.95 | 80.24 | 58,754,773 | 47,625.34M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.298/deeplabv3_resnetd101b_voc-8024-fd8bf74f.pth.log)) |
| DeepLabv3 | ResNet(D)-152b | 98.11 | 81.20 | 74,398,421 | 59,894.87M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.298/deeplabv3_resnetd152b_voc-8120-f2dae198.pth.log)) |
| FCN-8s(d) | ResNet(D)-101b | 97.80 | 80.40 | 52,072,917 | 196,562.96M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.299/fcn8sd_resnetd101b_voc-8040-66edc0b0.pth.log)) |

### ADE20K

| Model | Extractor | Pix.Acc.,% | mIoU,% | Params | FLOPs/2 | Remarks |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PSPNet | ResNet(D)-50b | 79.37 | 36.87 | 46,782,550 | 162,595.14M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.297/pspnet_resnetd50b_ade20k-3687-13f22137.pth.log)) |
| PSPNet | ResNet(D)-101b | 79.93 | 37.97 | 65,774,678 | 231,008.79M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.297/pspnet_resnetd101b_ade20k-3797-115d62bf.pth.log)) |
| DeepLabv3 | ResNet(D)-50b | 79.72 | 37.13 | 39,795,798 | 32,756.18M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.298/deeplabv3_resnetd50b_ade20k-3713-bddbb458.pth.log)) |
| DeepLabv3 | ResNet(D)-101b | 80.21 | 37.84 | 58,787,926 | 47,651.23M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.298/deeplabv3_resnetd101b_ade20k-3784-977446a5.pth.log)) |
| FCN-8s(d) | ResNet(D)-50b | 76.92 | 33.39 | 33,146,966 | 128,387.08M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.299/fcn8sd_resnetd50b_ade20k-3339-e1dad8a1.pth.log)) |
| FCN-8s(d) | ResNet(D)-101b | 79.01 | 35.88 | 52,139,094 | 196,800.73M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.299/fcn8sd_resnetd101b_ade20k-3588-30d05ca4.pth.log)) |

### Cityscapes

| Model | Extractor | Pix.Acc.,% | mIoU,% | Params | FLOPs/2 | Remarks |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PSPNet | ResNet(D)-101b | 96.17 | 71.72 | 65,707,475 | 230,767.33M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.297/pspnet_resnetd101b_cityscapes-7172-0a6efb49.pth.log)) |
| ICNet | ResNet(D)-50b | 95.50 | 64.02 | 47,489,184 | 14,253.43M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.457/icnet_resnetd50b_cityscapes-6402-b380f8cc.pth.log)) |
| SINet | - | 94.08 | 61.72 | 119,418 | 1,419.90M | From [clovaai/c3_sinet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.437/sinet_cityscapes-6172-8ecd1414.pth.log)) |
| Fast-SCNN | - | 95.14 | 65.76 | 1,138,051 | 3493.33M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.474/fastscnn_cityscapes-6576-b9859a25.pth.log)) |
| DANet | ResNet(D)-50b | 95.91 | 67.99 | 47,586,427 | 180,397.43M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.468/danet_resnetd50b_cityscapes-6799-c5740c9f.pth.log)) |
| DANet | ResNet(D)-101b | 96.03 | 68.10 | 66,578,555 | 248,811.08M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.468/danet_resnetd101b_cityscapes-6810-f1eeb724.pth.log)) |

### COCO Semantic Segmentation

| Model | Extractor | Pix.Acc.,% | mIoU,% | Params | FLOPs/2 | Remarks |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PSPNet | ResNet(D)-101b | 92.05 | 67.41 | 65,708,501 | 230,771.01M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.297/pspnet_resnetd101b_coco-6741-c8b13be6.pth.log)) |
| DeepLabv3 | ResNet(D)-101b | 92.19 | 67.73 | 58,754,773 | 47,625.34M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.298/deeplabv3_resnetd101b_coco-6773-e59c1d8f.pth.log)) |
| DeepLabv3 | ResNet(D)-152b | 92.24 | 68.99 | 74,398,421 | 275,087.91M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.298/deeplabv3_resnetd152b_coco-6899-7e946d7a.pth.log)) |
| FCN-8s(d) | ResNet(D)-101b | 91.44 | 60.11 | 52,072,917 | 196,562.96M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.299/fcn8sd_resnetd101b_coco-6011-ebe2ad0b.pth.log)) |

### CelebAMask-HQ

| Model | Extractor | Params | FLOPs/2 | Remarks |
| --- | --- | ---: | ---: | --- |
| BiSeNet | ResNet-18 | 13,300,416 | - | From [zllrunning/face...Torch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.462/bisenet_resnet18_celebamaskhq-0000-98affefd.pth.log)) |

### COCO Keypoints Detection

| Model | Extractor | OKS AP, % | Params | FLOPs/2 | Remarks |
| --- | --- | ---: | ---: | ---: | --- |
| AlphaPose | Fast-SE-ResNet-101b | 74.15/91.59/80.68 | 59,569,873 | 9,553.89M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.454/alphapose_fastseresnet101b_coco-7415-b9e3f64a.pth.log)) |
| SimplePose | ResNet-18 | 66.31/89.20/73.41 | 15,376,721 | 1,799.25M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.455/simplepose_resnet18_coco-6631-7c3656b3.pth.log)) |
| SimplePose | ResNet-50b | 71.02/91.23/78.57 | 33,999,697 | 4,041.06M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.455/simplepose_resnet50b_coco-7102-621d2545.pth.log)) |
| SimplePose | ResNet-101b | 72.44/92.18/79.76 | 52,991,825 | 7,685.04M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.455/simplepose_resnet101b_coco-7244-540c29ec.pth.log)) |
| SimplePose | ResNet-152b | 72.53/92.14/79.61 | 68,635,473 | 11,332.86M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.455/simplepose_resnet152b_coco-7253-3a358d7d.pth.log)) |
| SimplePose | ResNet(A)-50b | 71.70/91.31/78.66 | 34,018,929 | 4,278.56M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.455/simplepose_resneta50b_coco-7170-2d973dc5.pth.log)) |
| SimplePose | ResNet(A)-101b | 72.97/92.24/80.81 | 53,011,057 | 7,922.54M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.455/simplepose_resneta101b_coco-7297-08175610.pth.log)) |
| SimplePose | ResNet(A)-152b | 73.44/92.27/80.72 | 68,654,705 | 11,570.36M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.455/simplepose_resneta152b_coco-7344-dacb65cf.pth.log)) |
| SimplePose(Mobile) | ResNet-18 | 66.25/89.17/74.32 | 12,858,208 | 1,960.96M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.456/simplepose_mobile_resnet18_coco-6625-1e27b206.pth.log)) |
| SimplePose(Mobile) | ResNet-50b | 71.10/91.28/78.67 | 25,582,944 | 4,221.30M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.456/simplepose_mobile_resnet50b_coco-7110-023f910c.pth.log)) |
| SimplePose(Mobile) | 1.0 MobileNet-224 | 64.10/88.06/71.23 | 5,019,744 | 751.36M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.456/simplepose_mobile_mobilenet_w1_coco-6410-0ca46de0.pth.log)) |
| SimplePose(Mobile) | 1.0 MobileNetV2b-224 | 63.74/88.12/71.06 | 4,102,176 | 495.95M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.456/simplepose_mobile_mobilenetv2b_w1_coco-6374-94f86097.pth.log)) |
| SimplePose(Mobile) | MobileNetV3 Small 224/1.0 | 54.34/83.67/59.35 | 2,625,088 | 236.51M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.456/simplepose_mobile_mobilenetv3_small_w1_coco-5434-5cedb749.pth.log)) |
| SimplePose(Mobile) | MobileNetV3 Large 224/1.0 | 63.67/88.91/70.82 | 4,768,336 | 403.97M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.456/simplepose_mobile_mobilenetv3_large_w1_coco-6367-9515de07.pth.log)) |
| Lightweight OpenPose 2D | MobileNet | 39.99/65.95/40.70 | 4,091,698 | 8,948.96M | From [Daniil-Osokin/lighw...ch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.458/lwopenpose2d_mobilenet_cmupan_coco-3999-a6b9c66b.pth.log)) |
| Lightweight OpenPose 3D | MobileNet | 39.99/65.95/40.70 | 5,085,983 | 11,049.43M | From [Daniil-Osokin/li...3d...ch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.458/lwopenpose3d_mobilenet_cmupan_coco-3999-4c727e1d.pth.log)) |
| IBPPose | - | 64.87/83.62/70.13 | 95,827,784 | 57,195.91M | From [jialee93/Improved...Parts] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.459/ibppose_coco-6487-1958fe10.pth.log)) |

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
[soeaver/AirNet-PyTorch]: https://github.com/soeaver/AirNet-PyTorch
[soeaver/mxnet-model]: https://github.com/soeaver/mxnet-model
[Jongchan/attention-module]: https://github.com/Jongchan/attention-module
[XingangPan/IBN-Net]: https://github.com/XingangPan/IBN-Net
[cypw/CRU-Net]: https://github.com/cypw/CRU-Net
[kevin-ssy/FishNet]: https://github.com/kevin-ssy/FishNet
[ucbdrive/dla]: https://github.com/ucbdrive/dla
[sacmehta/ESPNetv2]: https://github.com/sacmehta/ESPNetv2
[jhjacobsen/pytorch-i-revnet]: https://github.com/jhjacobsen/pytorch-i-revnet
[wielandbrendel/bag...models]: https://github.com/wielandbrendel/bag-of-local-features-models
[MIT-HAN-LAB/ProxylessNAS]: https://github.com/MIT-HAN-LAB/ProxylessNAS
[yangze0930/NTS-Net]: https://github.com/yangze0930/NTS-Net
[rwightman/pyt...models]: https://github.com/rwightman/pytorch-image-models
[HRNet/HRNet...ation]: https://github.com/HRNet/HRNet-Image-Classification
[stigma0617/VoVNet.pytorch]: https://github.com/stigma0617/VoVNet.pytorch
[PingoLH/Pytorch-HarDNet]: https://github.com/PingoLH/Pytorch-HarDNet
[clovaai/c3_sinet]: https://github.com/clovaai/c3_sinet
[Daniil-Osokin/lighw...ch]: https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
[Daniil-Osokin/li...3d...ch]: https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch
[jialee93/Improved...Parts]: https://github.com/jialee93/Improved-Body-Parts
[zllrunning/face...Torch]: https://github.com/zllrunning/face-parsing.PyTorch
[MCG-NKU/SCNet]: https://github.com/MCG-NKU/SCNet