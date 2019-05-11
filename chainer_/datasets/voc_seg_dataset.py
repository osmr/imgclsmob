import os
import numpy as np
from PIL import Image
import chainer
from chainer import Chain
from .seg_dataset import SegDataset
from .dataset_metainfo import DatasetMetaInfo


class VOCSegDataset(SegDataset):
    """
    Pascal VOC2012 semantic segmentation dataset.

    Parameters
    ----------
    root : str
        Path to VOCdevkit folder.
    mode : str, default 'train'
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

    classes = 21
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


class VOCSegTestPredictor(Chain):

    def __init__(self,
                 base_model,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        super(VOCSegTestPredictor, self).__init__()
        self.mean = np.array(mean, np.float32)[:, np.newaxis, np.newaxis]
        self.std = np.array(std, np.float32)[:, np.newaxis, np.newaxis]
        with self.init_scope():
            self.model = base_model

    def _preprocess(self, img):
        dtype = chainer.get_dtype(None)
        img = img.transpose(2, 0, 1)
        img = img.astype(dtype)
        img *= 1.0 / 255.0

        img -= self.mean
        img /= self.std
        return img

    def predict(self, imgs):
        imgs = self.xp.asarray([self._preprocess(img) for img in imgs])

        with chainer.using_config("train", False), chainer.function.no_backprop_mode():
            imgs = chainer.Variable(imgs)
            predictions = self.model(imgs)

        output = chainer.backends.cuda.to_cpu(predictions.array)
        # output = np.argmax(output, axis=1).astype(np.int32)

        return output


class VOCMetaInfo(DatasetMetaInfo):
    def __init__(self):
        super(VOCMetaInfo, self).__init__()
        self.label = "VOC"
        self.short_label = "voc"
        self.root_dir_name = "voc"
        self.dataset_class = VOCSegDataset
        self.num_training_samples = None
        self.in_channels = 3
        self.num_classes = VOCSegDataset.classes
        self.input_image_size = (480, 480)
        self.train_metric_capts = None
        self.train_metric_names = None
        self.train_metric_extra_kwargs = None
        self.val_metric_capts = None
        self.val_metric_names = None
        self.test_metric_extra_kwargs = None
        self.test_metric_capts = ["Val.PixAcc", "Val.IoU"]
        self.test_metric_names = ["PixelAccuracyMetric", "MeanIoUMetric"]
        self.test_metric_extra_kwargs = [
            {"vague_idx": VOCSegDataset.vague_idx,
             "use_vague": VOCSegDataset.use_vague,
             "macro_average": False},
            {"num_classes": VOCSegDataset.classes,
             "vague_idx": VOCSegDataset.vague_idx,
             "use_vague": VOCSegDataset.use_vague,
             "bg_idx": VOCSegDataset.background_idx,
             "ignore_bg": VOCSegDataset.ignore_bg,
             "macro_average": False}]
        self.saver_acc_ind = 1
        self.train_transform = None
        self.val_transform = VOCSegTestPredictor
        self.test_transform = VOCSegTestPredictor
        self.ml_type = "imgseg"
        self.allow_hybridize = False
        self.net_extra_kwargs = {"aux": False, "fixed_size": False}
        self.load_ignore_extra = True
        self.image_base_size = 520
        self.image_crop_size = 480

    def add_dataset_parser_arguments(self,
                                     parser,
                                     work_dir_path):
        super(VOCMetaInfo, self).add_dataset_parser_arguments(parser, work_dir_path)
        parser.add_argument(
            '--image-base-size',
            type=int,
            default=520,
            help='base image size')
        parser.add_argument(
            '--image-crop-size',
            type=int,
            default=480,
            help='crop image size')

    def update(self,
               args):
        super(VOCMetaInfo, self).update(args)
        self.image_base_size = args.image_base_size
        self.image_crop_size = args.image_crop_size
