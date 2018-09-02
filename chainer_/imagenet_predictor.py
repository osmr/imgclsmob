import numpy as np

import chainer
from chainer import Chain

from chainercv.transforms import scale
from chainercv.transforms import center_crop


class ImagenetPredictor(Chain):

    def __init__(self,
                 base_model,
                 scale_size=256,
                 crop_size=224,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        super(ImagenetPredictor, self).__init__()
        self.scale_size = scale_size
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size
        self.mean = np.array(mean, np.float32)[:, np.newaxis, np.newaxis]
        self.std = np.array(std, np.float32)[:, np.newaxis, np.newaxis]
        with self.init_scope():
            self.model = base_model

    def _preprocess(self, img):
        img = scale(img=img, size=self.scale_size)
        img = center_crop(img, self.crop_size)
        img /= 255.0
        img -= self.mean
        img /= self.std
        return img

    def predict(self, imgs):
        imgs = self.xp.asarray([self._preprocess(img) for img in imgs])

        with chainer.using_config('train', False), chainer.function.no_backprop_mode():
            imgs = chainer.Variable(imgs)
            predictions = self.model(imgs)

        output = chainer.backends.cuda.to_cpu(predictions.array)
        return output

