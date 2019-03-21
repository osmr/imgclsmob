import os
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch.utils.data as data


class ADE20KSegmentation(data.Dataset):
    """
    ADE20K semantic segmentation dataset.

    Parameters
    ----------
    root : string
        Path to ADE20K folder.
    mode: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    target_transform : callable, optional
        A function/transform that takes in the target and transforms it.
    """

    def __init__(self,
                 root,
                 mode="train",
                 transform=None,
                 target_transform=None,
                 base_size=520,
                 crop_size=480):
        self.root = os.path.expanduser(root)
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.base_size = base_size
        self.crop_size = crop_size

        ade20k_root = os.path.join(self.root, "ADEChallengeData2016")
        image_dir_path = os.path.join(ade20k_root, 'images')
        mask_dir_path = os.path.join(ade20k_root, 'annotations')

        mode_dir_name = "training" if (mode == "train") else "validation"
        image_dir_path = os.path.join(image_dir_path, mode_dir_name)
        mask_dir_path = os.path.join(mask_dir_path, mode_dir_name)

        images = []
        masks = []
        for filename in os.listdir(image_dir_path):
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(image_dir_path, filename)
                maskname = basename + '.png'
                maskpath = os.path.join(mask_dir_path, maskname)
                if os.path.isfile(maskpath):
                    images.append(imgpath)
                    masks.append(maskpath)
                else:
                    print('cannot find the mask:', maskpath)

        self.images = images
        self.masks = masks
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.images)

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        return np.array(mask).astype(np.int32)
