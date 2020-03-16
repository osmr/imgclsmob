"""
    GRMIPose (Google PoseNet) for COCO Keypoint, implemented in TensorFlow (Lite).
    Original paper: 'Towards Accurate Multi-person Pose Estimation in the Wild,' https://arxiv.org/abs/1701.01779.
"""

__all__ = ['GRMIPoseLite', 'grmiposelite_mobilenet_w1_coco']

import math
import numpy as np
import tensorflow as tf


class GRMIPoseLite(tf.keras.Model):
    """
    GRMIPose (Google PoseNet) model from 'Towards Accurate Multi-person Pose Estimation in the Wild,'
    https://arxiv.org/abs/1701.01779.

    Parameters:
    ----------
    interpreter : obj
        Instance of the TFLite model interpreter.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (257, 257)
        Spatial size of the expected input image.
    keypoints : int, default 17
        Number of keypoints.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 interpreter,
                 in_channels=3,
                 in_size=(257, 257),
                 keypoints=17,
                 data_format="channels_last",
                 **kwargs):
        super(GRMIPoseLite, self).__init__(**kwargs)
        assert (in_channels == 3)
        self.in_size = in_size
        self.keypoints = keypoints
        self.data_format = data_format
        self.interpreter = interpreter

        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        self.input_tensor_index = input_details[0]["index"]
        self.in_shape = tuple(input_details[0]["shape"])
        assert (self.in_size == self.in_shape[1:3])
        self.output_tensor_index_list = [i["index"] for i in self.interpreter.get_output_details()]

    def call(self, x, training=None):
        x_np = x.numpy()

        # import cv2
        # cv2.imshow("x_np", x_np[0])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        assert (x_np.shape == self.in_shape)
        self.interpreter.set_tensor(self.input_tensor_index, x_np)
        self.interpreter.invoke()
        heatmap = self.interpreter.get_tensor(self.output_tensor_index_list[0])
        offsets = self.interpreter.get_tensor(self.output_tensor_index_list[1])
        pts = np.zeros((self.keypoints, 3), np.float32)
        oh, ow = heatmap.shape[1:3]
        fh = self.in_size[0] / (oh - 1)
        fw = self.in_size[1] / (ow - 1)
        for k in range(self.keypoints):
            max_h = heatmap[0, 0, 0, 0]
            max_i = 0
            max_j = 0
            for i in range(oh):
                for j in range(ow):
                    h = heatmap[0, i, j, k]
                    if h > max_h:
                        max_h = h
                        max_i = i
                        max_j = j
            pts[k, 0] = max_i * fh + offsets[0, max_i, max_j, k]
            pts[k, 1] = max_j * fw + offsets[0, max_i, max_j, k + self.keypoints]
            pts[k, 2] = self.sigmoid(max_h)

        pts1 = pts.copy()
        for k in range(self.keypoints):
            pts1[k, 0] = 0.25 * pts[k, 1]
            pts1[k, 1] = 0.25 * pts[k, 0]
        y = tf.convert_to_tensor(np.expand_dims(pts1, axis=0))

        # import cv2
        # canvas = x_np[0]
        # canvas = cv2.cvtColor(canvas, code=cv2.COLOR_BGR2RGB)
        # for k in range(self.keypoints):
        #     cv2.circle(
        #         canvas,
        #         (pts[k, 1], pts[k, 0]),
        #         3,
        #         (0, 0, 255),
        #         -1)
        # scale_factor = 3
        # cv2.imshow(
        #     winname="canvas",
        #     mat=cv2.resize(
        #         src=canvas,
        #         dsize=None,
        #         fx=scale_factor,
        #         fy=scale_factor,
        #         interpolation=cv2.INTER_NEAREST))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return y

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))


def get_grmiposelite(model_path,
                     keypoints,
                     model_name=None,
                     data_format="channels_last",
                     pretrained=False,
                     **kwargs):
    """
    Create GRMIPose (Google PoseNet) model with specific parameters.

    Parameters:
    ----------
    model_path : str
        Path to pretrained model.
    keypoints : int
        Number of keypoints.
    model_name : str or None, default None
        Model name for loading pretrained model.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    """
    assert (pretrained is not None)
    assert (model_name is not None)

    if (model_path is None) or (not model_path):
        raise ValueError("Parameter `model_path` should be properly initialized for loading pretrained model.")
    interpreter = tf.lite.Interpreter(model_path=model_path)

    net = GRMIPoseLite(
        interpreter=interpreter,
        keypoints=keypoints,
        data_format=data_format,
        **kwargs)

    return net


def grmiposelite_mobilenet_w1_coco(model_path, keypoints=17, data_format="channels_last", pretrained=False, **kwargs):
    """
    GRMIPose (Google PoseNet) model on the base of 1.0 MobileNet-224 for COCO Keypoint from 'Towards Accurate
    Multi-person Pose Estimation in the Wild,' https://arxiv.org/abs/1701.01779.

    Parameters:
    ----------
    model_path : str
        Path to pretrained model.
    keypoints : int, default 17
        Number of keypoints.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_grmiposelite(model_path=model_path, keypoints=keypoints, model_name="grmiposelite_mobilenet_w1_coco",
                            data_format=data_format, pretrained=pretrained, **kwargs)


def _test():
    data_format = "channels_last"
    in_size = (257, 257)
    keypoints = 17
    pretrained = False
    model_path = ""

    models = [
        grmiposelite_mobilenet_w1_coco,
    ]

    for model in models:

        net = model(model_path=model_path, pretrained=pretrained, in_size=in_size, data_format=data_format)

        batch = 1
        x = tf.random.normal((batch, in_size[0], in_size[1], 3))
        y = net(x)
        assert (y.shape[0] == batch)
        assert ((y.shape[1] == keypoints) and (y.shape[2] == 3))


if __name__ == "__main__":
    _test()
