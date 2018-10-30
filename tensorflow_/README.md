# Large-scale image classification networks

Several large-scale image classification models on TensorFlow, trained on the ImageNet-1k dataset.

## Installation

To install, use:
```
pip install tensorflowcv tensorflow-gpu>=1.11.0
```
To enable/disable different hardware supports, check out TensorFlow installation [instructions](https://www.tensorflow.org/).

Note that the models use NCHW data format. The current version of TensorFlow cannot work with them on CPU.

## Usage

Example of using the pretrained ResNet-18 model:
```
from tensorflowcv.model_provider import get_model as tfcv_get_model
from tensorflowcv.model_provider import init_variables_from_state_dict as tfcv_init_variables_from_state_dict
import tensorflow as tf
net = tfcv_get_model("resnet18", pretrained=True)
x = tf.placeholder(dtype=tf.float32, shape=(None, 3, 224, 224), name='xx')
y_net = net(x)
with tf.Session() as sess:
    tfcv_init_variables_from_state_dict(sess=sess, state_dict=net.state_dict)
```
