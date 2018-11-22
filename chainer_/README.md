# Large-scale image classification networks

Collection of large-scale image classification models on Chainer, pretrained on the ImageNet-1k dataset.

## Installation

To install, use:
```
pip install chainercv2
```

## Usage

Example of using the pretrained ResNet-18 model:
```
from chainercv2.model_provider import get_model as chcv2_get_model
net = chcv2_get_model("resnet18", pretrained=True)
```
