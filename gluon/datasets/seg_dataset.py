import random
import numpy as np
import mxnet as mx
from PIL import Image, ImageOps, ImageFilter
from mxnet.gluon.data import dataset


class SegDataset(dataset.Dataset):
    """
    Segmentation base dataset.

    Parameters
    ----------
    root : str
        Path to data folder.
    mode : str
        'train', 'val', 'test', or 'demo'.
    transform : callable
        A function that transforms the image.
    """
    def __init__(self,
                 root,
                 mode,
                 transform,
                 base_size=520,
                 crop_size=480):
        assert (mode in ("train", "val", "test", "demo"))
        self.root = root
        self.mode = mode
        self.transform = transform
        self.base_size = base_size
        self.crop_size = crop_size

    def _val_sync_transform(self, image, mask, ctx=mx.cpu()):
        short_size = self.crop_size
        w, h = image.size
        if w > h:
            oh = short_size
            ow = int(float(w * oh) / h)
        else:
            ow = short_size
            oh = int(float(h * ow) / w)
        image = image.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # Center crop:
        outsize = self.crop_size
        x1 = int(round(0.5 * (ow - outsize)))
        y1 = int(round(0.5 * (oh - outsize)))
        image = image.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # Final transform:
        image, mask = self._img_transform(image, ctx=ctx), self._mask_transform(mask, ctx=ctx)
        return image, mask

    def _train_sync_transform(self, image, mask, ctx=mx.cpu()):
        # Random mirror:
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        # Random scale (short edge):
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = image.size
        if w > h:
            oh = short_size
            ow = int(float(w * oh) / h)
        else:
            ow = short_size
            oh = int(float(h * ow) / w)
        image = image.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # Pad crop:
        crop_size = self.crop_size
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            image = ImageOps.expand(image, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # Random crop crop_size:
        w, h = image.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        image = image.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # Gaussian blur as in PSP:
        if random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # Final transform:
        image, mask = self._img_transform(image, ctx=ctx), self._mask_transform(mask, ctx=ctx)
        return image, mask

    @staticmethod
    def _img_transform(image, ctx=mx.cpu()):
        return mx.nd.array(np.array(image), ctx=ctx)

    @staticmethod
    def _mask_transform(mask, ctx=mx.cpu()):
        return mx.nd.array(np.array(mask), ctx=ctx, dtype=np.int32)
