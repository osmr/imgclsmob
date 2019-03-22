import os
import numpy as np
import mxnet as mx
from PIL import Image
from .seg_dataset import SegDataset


class VOCSegDataset(SegDataset):
    """
    Pascal VOC semantic segmentation dataset.

    Parameters
    ----------
    root : string
        Path to VOCdevkit folder.
    split: string, default 'train'
        'train', 'val' or 'test'.
    mode: string, default None
        'train', 'val' or 'test'.
    transform : callable, optional
        A function that transforms the image
    """
    def __init__(self,
                 root,
                 split="train",
                 mode=None,
                 transform=None,
                 **kwargs):
        super(VOCSegDataset, self).__init__(root, split, mode, transform, **kwargs)
        self.classes = 21

        base_dir_path = os.path.join(root, 'VOC2012')
        image_dir_path = os.path.join(base_dir_path, 'JPEGImages')
        mask_dir_path = os.path.join(base_dir_path, 'SegmentationClass')

        splits_dir_path = os.path.join(base_dir_path, 'ImageSets', 'Segmentation')
        if split == 'train':
            split_file_path = os.path.join(splits_dir_path, 'trainval.txt')
        elif split == 'val':
            split_file_path = os.path.join(splits_dir_path, 'val.txt')
        elif split == 'test':
            split_file_path = os.path.join(splits_dir_path, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split')

        self.images = []
        self.masks = []
        with open(os.path.join(split_file_path), "r") as lines:
            for line in lines:
                image_file_path = os.path.join(image_dir_path, line.rstrip('\n') + ".jpg")
                assert os.path.isfile(image_file_path)
                self.images.append(image_file_path)
                if split != "test":
                    mask_file_path = os.path.join(mask_dir_path, line.rstrip('\n') + ".png")
                    assert os.path.isfile(mask_file_path)
                    self.masks.append(mask_file_path)

        if split != "test":
            assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        if self.mode == "test":
            image = self._img_transform(image)
            if self.transform is not None:
                image = self.transform(image)
            return image, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])

        if self.mode == "train":
            image, mask = self._sync_transform(image, mask)
        elif self.mode == "val":
            image, mask = self._val_sync_transform(image, mask)
        else:
            assert self.mode == "testval"
            image, mask = self._img_transform(image), self._mask_transform(mask)

        if self.transform is not None:
            image = self.transform(image)

        return image, mask

    @staticmethod
    def _mask_transform(mask):
        np_mask = np.array(mask).astype(np.int32)
        np_mask[np_mask == 255] = -1
        return mx.nd.array(np_mask, mx.cpu())

    def __len__(self):
        return len(self.images)
