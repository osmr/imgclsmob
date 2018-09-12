# Large-scale image classification networks

Several large-scale image classification models on MXNet/Gluon, trained on the ImageNet-1k dataset.

##Installation

To install, use:
```
pip install gluoncv2 mxnet>=1.2.1
```
To enable different hardware supports such as GPUs, check out [mxnet variants](https://pypi.org/project/mxnet/).
For example, you can install with cuda-9.2 supported mxnet:
```
pip install gluoncv2 mxnet-cu92>=1.2.1
```

##Usage

Example of using the pretrained ResNet-18 model:
```
from gluoncv2.model_provider import get_model as glcv2_get_model
net = glcv2_get_model("resnet18", pretrained=True)
```
