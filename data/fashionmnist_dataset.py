import os.path

import h5py
import numpy as np
from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform

class FashionMNISTDataset(BaseDataset):
    def __init__(self, opt, idx=None,dataType='train'):

        if dataType!='public' and dataType!='global':
            if idx is None:
                h5_name = "{}_fashionMNIST.h5".format(dataType)
            else:
                h5_name = '{}_fashionMNIST_unique_{:d}.h5'.format(dataType,idx)
        elif dataType == 'public':
            h5_name = "upload_0.05_fashionMNIST_unique.h5"
        else:
            h5_name = "public_test_unique.h5"

        print(f"Load: {h5_name}")
        self.is_test = True
        self.extend_len = 0
        BaseDataset.__init__(self, opt)
        self.file = h5py.File(os.path.join(opt.dataroot, h5_name), 'r')

        if 'train' in self.file:
            train_db = self.file['train']
        else:
            train_db = self.file
        self.image, self.label = self.build_pairs(train_db)

        assert (self.opt.load_size >= self.opt.crop_size)

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def build_pairs(self, dataset):
        image_arr = []
        label_arr = []

        images = dataset['images']
        labels = dataset['labels']

        keys = images.keys()
        for key in keys:
            img = images[key][()]
            label = labels[key][()]
            image_arr.append(img)
            label_arr.append(label)

        return image_arr, label_arr

    def __getitem__(self, index):

        A = self.label[index]
        B = self.image[index]
        if self.output_nc == 3 and len(B.shape) == 2:
            B = B[:, :, np.newaxis].repeat(3, axis=2)
        if len(B.shape) == 3:
            B = B.squeeze(0)
        B = Image.fromarray(np.uint8(B))

        transform_params = {}
        transform_params['crop_pos'] = (0, 0)
        transform_params['vflip'] = False
        transform_params['hflip'] = False

        self.opt.load_size = 32
        self.opt.preprocess = 'resize'
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))

        B = B_transform(B)

        return {'A': int(A), 'B': B, 'A_paths': str(index), 'B_paths': str(index)}
    def __len__(self):
        return len(self.image)