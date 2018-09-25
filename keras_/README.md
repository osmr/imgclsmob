# Large-scale image classification networks

Several large-scale image classification models on Keras (with MXNet backend), trained on the ImageNet-1k dataset.

## Installation

To install, use:
```
pip install kerascv mxnet>=1.2.1
```
To enable different hardware supports such as GPUs, check out [MXNet variants](https://pypi.org/project/mxnet/).
For example, you can install with CUDA-9.2 supported MXNet:
```
pip install kerascv mxnet-cu92>=1.2.1
```
After installation change the value of the field `image_data_format` to `channels_first` in the file `~/.keras/keras.json`. 

## Usage

Example of using the pretrained ResNet-18 model:
```
from kerascv.model_provider import get_model as kecv_get_model
net = kecv_get_model("resnet18", pretrained=True)
```
