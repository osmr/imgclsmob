# Large-scale image classification networks

Several large-scale image classification models on PyTorch, trained on the ImageNet-1k dataset.

##Installation

To install, use:
```
pip install pytorchcv torch>=0.4.1
```
To enable/disable different hardware supports such as GPUs, check out PyTorch installation [instructions](https://pytorch.org/).

##Usage

Example of using the pretrained ResNet-18 model:
```
from pytorchcv.model_provider import get_model as ptcv_get_model
net = ptcv_get_model("resnet18", pretrained=True)
```
