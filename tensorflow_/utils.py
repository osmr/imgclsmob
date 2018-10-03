import logging
import os
import multiprocessing
import numpy as np
import cv2

from tensorpack.tfutils import get_model_loader
from tensorpack.dataflow import imgaug, dataset, AugmentImageComponent, PrefetchDataZMQ, MultiThreadMapData, BatchData

from .model_provider import get_model


class GoogleNetResize(imgaug.ImageAugmentor):
    """
    crop 8%~100% of the original image
    See `Going Deeper with Convolutions` by Google.
    """
    def __init__(self, crop_area_fraction=0.08,
                 aspect_ratio_low=0.75, aspect_ratio_high=1.333,
                 target_shape=224):
        self._init(locals())

    def _augment(self, img, _):
        h, w = img.shape[:2]
        area = h * w
        for _ in range(10):
            targetArea = self.rng.uniform(self.crop_area_fraction, 1.0) * area
            aspectR = self.rng.uniform(self.aspect_ratio_low, self.aspect_ratio_high)
            ww = int(np.sqrt(targetArea * aspectR) + 0.5)
            hh = int(np.sqrt(targetArea / aspectR) + 0.5)
            if self.rng.uniform() < 0.5:
                ww, hh = hh, ww
            if hh <= h and ww <= w:
                x1 = 0 if w == ww else self.rng.randint(0, w - ww)
                y1 = 0 if h == hh else self.rng.randint(0, h - hh)
                out = img[y1:y1 + hh, x1:x1 + ww]
                out = cv2.resize(out, (self.target_shape, self.target_shape), interpolation=cv2.INTER_CUBIC)
                return out
        out = imgaug.ResizeShortestEdge(self.target_shape, interp=cv2.INTER_CUBIC).augment(img)
        out = imgaug.CenterCrop(self.target_shape).augment(out)
        return out


def get_imagenet_dataflow(datadir,
                          is_train,
                          batch_size,
                          augmentors,
                          parallel=None):
    """
    See explanations in the tutorial:
    http://tensorpack.readthedocs.io/en/latest/tutorial/efficient-dataflow.html
    """
    assert datadir is not None
    assert isinstance(augmentors, list)
    if parallel is None:
        parallel = min(40, multiprocessing.cpu_count() // 2)  # assuming hyperthreading
    if is_train:
        ds = dataset.ILSVRC12(datadir, 'train', shuffle=True)
        ds = AugmentImageComponent(ds, augmentors, copy=False)
        if parallel < 16:
            logging.warning("DataFlow may become the bottleneck when too few processes are used.")
        ds = PrefetchDataZMQ(ds, parallel)
        ds = BatchData(ds, batch_size, remainder=False)
    else:
        ds = dataset.ILSVRC12Files(datadir, 'val', shuffle=False)
        aug = imgaug.AugmentorList(augmentors)

        def mapf(dp):
            fname, cls = dp
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            im = aug.augment(im)
            return im, cls
        ds = MultiThreadMapData(ds, parallel, mapf, buffer_size=2000, strict=True)
        ds = BatchData(ds, batch_size, remainder=True)
        ds = PrefetchDataZMQ(ds, 1)
    return ds


def prepare_tf_context(num_gpus,
                       batch_size):
    batch_size *= max(1, num_gpus)
    return batch_size


def prepare_model(model_name,
                  pretrained_model_file_path):

    net = get_model(model_name)

    inputs_desc = None
    if pretrained_model_file_path:
        assert (os.path.isfile(pretrained_model_file_path))
        logging.info('Loading model: {}'.format(pretrained_model_file_path))
        inputs_desc = get_model_loader(pretrained_model_file_path)

    return net, inputs_desc


def get_data(is_train,
             batch_size,
             data_dir_path):

    if is_train:
        augmentors = [
            GoogleNetResize(crop_area_fraction=0.08),
            imgaug.RandomOrderAug([
                imgaug.BrightnessScale((0.6, 1.4), clip=False),
                imgaug.Contrast((0.6, 1.4), clip=False),
                imgaug.Saturation(0.4, rgb=False),
                # rgb-bgr conversion for the constants copied from fb.resnet.torch
                imgaug.Lighting(
                    0.1,
                    eigval=np.asarray([0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                    eigvec=np.array([
                        [-0.5675, 0.7192, 0.4009],
                        [-0.5808, -0.0045, -0.8140],
                        [-0.5836, -0.6948, 0.4203]], dtype='float32')[::-1, ::-1])]),
            imgaug.Flip(horiz=True)]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.CenterCrop((224, 224))]

    return get_imagenet_dataflow(
        datadir=data_dir_path,
        is_train=is_train,
        batch_size=batch_size,
        augmentors=augmentors)
