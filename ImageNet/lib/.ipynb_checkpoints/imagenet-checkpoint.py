import json
import numpy as np
import os
import random
import re
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
from iopath.common.file_io import PathManagerFactory
pathmgr = PathManagerFactory.get()


class Imagenet(torch.utils.data.Dataset):
    """ImageNet dataset."""

    def __init__(self, cfg, mode, num_retries=10):
        self.num_retries = num_retries
        self.cfg = cfg
        self.mode = mode
        self.data_path = cfg.DATA.PATH_TO_DATA_DIR
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for ImageNet".format(mode)
        print("Constructing ImageNet {}...".format(mode))
        if cfg.DATA.PATH_TO_PRELOAD_IMDB == "":
            self._construct_imdb()
        else:
            self._load_imdb()

    def _load_imdb(self):
        split_path = os.path.join(
            self.cfg.DATA.PATH_TO_PRELOAD_IMDB, f"{self.mode}.json"
        )
        with pathmgr.open(split_path, "r") as f:
            data = f.read()
        self._imdb = json.loads(data)

    def _construct_imdb(self):
        """Constructs the imdb."""
        # Compile the split data path
        split_path = os.path.join(self.data_path, self.mode)
        #logger.info("{} data path: {}".format(self.mode, split_path))
        # Images are stored per class in subdirs (format: n<number>)
        split_files = pathmgr.ls(split_path)
        self._class_ids = sorted(
            f for f in split_files if re.match(r"^n[0-9]+$", f)
        )
        # Map ImageNet class ids to contiguous ids
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}
        # Construct the image db
        self._imdb = []
        for class_id in self._class_ids:
            cont_id = self._class_id_cont_id[class_id]
            im_dir = os.path.join(split_path, class_id)
            for im_name in pathmgr.ls(im_dir):
                im_path = os.path.join(im_dir, im_name)
                self._imdb.append({"im_path": im_path, "class": cont_id})
        print("Number of images: {}".format(len(self._imdb)))
        print("Number of classes: {}".format(len(self._class_ids)))

    def load_image(self, im_path):
        """Prepares the image for network input with format of CHW RGB float"""
        with pathmgr.open(im_path, "rb") as f:
            with Image.open(f) as im:
                im = im.convert("RGB")
        im = torch.from_numpy(np.array(im).astype(np.float32) / 255.0)
        # H W C to C H W
        im = im.permute([2, 0, 1])
        return im

    def _prepare_im_tf(self, im_path):
        with pathmgr.open(im_path, "rb") as f:
            with Image.open(f) as im:
                im = im.convert("RGB")
        
        normalize = transforms.Normalize(mean=cfg.TRAIN.MEAN,std=cfg.TRAIN.STD)
        if cfg.DATA.IMG_SIZE > 0: 
                resize_transform = [ transforms.Resize(cfg.DATA.IMG_SIZE) ] 
        else:
                resize_transform = []
        # Watchout the normalize parameter -> Not seen in the original code
        if self.mode == "train":
            aug_transform = transforms.Compose( resize_transform + [
                                                                    transforms.RandomResizedCrop(cfg.DATA.CROP_SIZE),
                                                                    transforms.RandomHorizontalFlip(),
                                                                    transforms.ToTensor()])#,
                                                                    #normalize])
        else:
            aug_transform = transforms.Compose(resize_transform + [
                                                   transforms.CenterCrop(cfg.DATA.CROP_SIZE),
                                                   transforms.ToTensor()])#,
                                                   #normalize])
        
        im = aug_transform(im)
        return im

    def __load__(self, index):
        try:
            # Load the image
            im_path = self._imdb[index]["im_path"]
            # Prepare the image for training / testing
            im = self._prepare_im_tf(im_path)
            return im
        except Exception:
            #print('Error Loading Data')
            return None

    def __getitem__(self, index):
        # if the current image is corrupted, load a different image.
        for _ in range(self.num_retries):
            im = self.__load__(index)
            # Data corrupted, retry with a different image.
            if im is None:
                index = random.randint(0, len(self._imdb) - 1)
            else:
                break
        # Retrieve the label
        label = self._imdb[index]["class"]
        if isinstance(im, list):
            label = [label for _ in range(len(im))]
            #dummy = [torch.Tensor() for _ in range(len(im))]
            return im, label
            #return im, label, dummy, {}
        else:
            #dummy = torch.Tensor()
            return im, label
            #return [im], label, dummy, {}

    def __len__(self):
        return len(self._imdb)


if __name__ == '__main__':
    from easydict import EasyDict
    cfg = {'DATA' : {'PATH_TO_PRELOAD_IMDB' : "",
                     'PATH_TO_DATA_DIR'     : "../../../Dataset/ILSVRC/Data/CLS-LOC",
                     'IMG_SIZE'  : 0,
                     'CROP_SIZE' : 224},
           'TRAIN': {'MEAN': [0.485, 0.456, 0.406],'STD' : [0.229, 0.224, 0.225]}}
    cfg = EasyDict(cfg)
    dataset = Imagenet(cfg,'val',num_retries = 10)
    print(dataset.__getitem__(0))
