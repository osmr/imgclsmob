import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from chainercv.chainer_experimental.datasets.sliceable import GetterDataset


class SegDataset(GetterDataset):
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
        super(SegDataset, self).__init__()
        assert (mode in ("train", "val", "test", "demo"))
        assert (mode in ("test", "demo"))
        self.root = root
        self.mode = mode
        self.transform = transform
        self.base_size = base_size
        self.crop_size = crop_size

    def _val_sync_transform(self, image, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = image.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        image = image.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = image.size
        x1 = int(round(0.5 * (w - outsize)))
        y1 = int(round(0.5 * (h - outsize)))
        image = image.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        image, mask = self._img_transform(image), self._mask_transform(mask)
        return image, mask

    def _sync_transform(self, image, mask):
        # random mirror
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = image.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        image = image.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            image = ImageOps.expand(image, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = image.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        image = image.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        image, mask = self._img_transform(image), self._mask_transform(mask)
        return image, mask

    @staticmethod
    def _img_transform(image):
        return np.array(image)

    @staticmethod
    def _mask_transform(mask):
        return np.array(mask).astype(np.int32)
