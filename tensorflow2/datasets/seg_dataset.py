import random
import threading
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator


class SegDataset(object):
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

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SegDirectoryIterator(DirectoryIterator):
    allowed_class_modes = {'categorical', 'binary', 'sparse', 'input', None}

    def __init__(self,
                 directory,
                 image_data_generator,
                 target_size=(256, 256),
                 color_mode='rgb',
                 classes=None,
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest',
                 dtype='float32',
                 dataset=None):
        super(SegDirectoryIterator, self).set_processing_attrs(
            image_data_generator,
            target_size,
            color_mode,
            data_format,
            save_to_dir,
            save_prefix,
            save_format,
            subset,
            interpolation)

        self.dataset = dataset
        self.class_mode = class_mode
        self.dtype = dtype

        self.n = len(self.dataset)
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: Array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        # batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=self.dtype)
        # batch_y = np.zeros((len(index_array),) + self.image_shape, dtype=np.int32)
        batch_x = None
        batch_y = None
        for i, j in enumerate(index_array):
            x, y = self.dataset[j]
            if batch_x is None:
                batch_x = np.zeros((len(index_array),) + x.shape, dtype=self.dtype)
                batch_y = np.zeros((len(index_array),) + y.shape, dtype=np.int32)
            # if self.data_format == "channel_first":
            #     print("*")
            # print("batch_x.shape={}".format(batch_x.shape))
            # print("batch_y.shape={}".format(batch_y.shape))
            # print("x.shape={}".format(x.shape))
            # print("y.shape={}".format(y.shape))
            batch_x[i] = x
            batch_y[i] = y
        return batch_x, batch_y


class SegImageDataGenerator(ImageDataGenerator):

    def flow_from_directory(self,
                            directory,
                            target_size=(256, 256),
                            color_mode='rgb',
                            classes=None,
                            class_mode='categorical',
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation='nearest',
                            dataset=None):
        return SegDirectoryIterator(
            directory,
            self,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation,
            dataset=dataset)
