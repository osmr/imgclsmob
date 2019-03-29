import os
import numpy as np
from PIL import Image
from .seg_dataset import SegDataset


class VOCSegDataset(SegDataset):
    """
    Pascal VOC2012 semantic segmentation dataset.

    Parameters
    ----------
    root : string
        Path to VOCdevkit folder.
    mode: string, default 'train'
        'train', 'val', 'test', or 'demo'.
    transform : callable, optional
        A function that transforms the image.
    """
    def __init__(self,
                 root,
                 mode="train",
                 transform=None,
                 **kwargs):
        super(VOCSegDataset, self).__init__(
            root=root,
            mode=mode,
            transform=transform,
            **kwargs)
        self.classes = 21

        base_dir_path = os.path.join(root, "VOC2012")
        image_dir_path = os.path.join(base_dir_path, "JPEGImages")
        mask_dir_path = os.path.join(base_dir_path, "SegmentationClass")

        splits_dir_path = os.path.join(base_dir_path, "ImageSets", "Segmentation")
        if mode == "train":
            split_file_path = os.path.join(splits_dir_path, "train.txt")
        elif mode in ("val", "test", "demo"):
            split_file_path = os.path.join(splits_dir_path, "val.txt")
        else:
            raise RuntimeError("Unknown dataset splitting mode")

        self.images = []
        self.masks = []
        with open(os.path.join(split_file_path), "r") as lines:
            for line in lines:
                image_file_path = os.path.join(image_dir_path, line.rstrip('\n') + ".jpg")
                assert os.path.isfile(image_file_path)
                self.images.append(image_file_path)
                mask_file_path = os.path.join(mask_dir_path, line.rstrip('\n') + ".png")
                assert os.path.isfile(mask_file_path)
                self.masks.append(mask_file_path)

        assert (len(self.images) == len(self.masks))

        # self.images = self.images[:10]
        # self.masks = self.masks[:10]

        self.add_getter('img', self._get_image)
        self.add_getter('label', self._get_label)

    def _get_image(self, i):
        image = Image.open(self.images[i]).convert("RGB")
        assert (self.mode in ("test", "demo"))
        image = self._img_transform(image)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def _get_label(self, i):
        if self.mode == "demo":
            return os.path.basename(self.images[i])
        assert (self.mode == "test")
        mask = Image.open(self.masks[i])
        mask = self._mask_transform(mask)
        return mask

    vague_idx = 255
    use_vague = True
    background_idx = 0
    ignore_bg = True

    @staticmethod
    def _mask_transform(mask):
        np_mask = np.array(mask).astype(np.int32)
        # np_mask[np_mask == 255] = VOCSegDataset.vague_idx
        return np_mask

    def __len__(self):
        return len(self.images)
