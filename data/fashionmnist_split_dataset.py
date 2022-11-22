import os.path
import random
import sys

# import cv2
import h5py
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform
from data.fashionmnist_dataset import FashionMNISTDataset

class FashionMNISTSplitDataset(BaseDataset):
    def __init__(self, opt,dataType= 'train'):
        self.dataType = dataType
        self.split_db = []
        self.opt = opt
        if dataType!='public' and dataType!='global':
            for i in range(opt.n_client):
                self.split_db.append(FashionMNISTDataset(opt, i,dataType))
            self.compute_weights()
        else:
            self.split_db.append(FashionMNISTDataset(opt, 0, dataType))
            self.weights = []
            self.max_num = None
            self.label_of_client = None

    def __getitem__(self, index):
        if self.dataType =='train':
            result = {'weights': self.weights, 'max_num':self.max_num,'label_of_client':self.label_of_client,'fedgen_weights':self.fedgen_weights}
        else:
            result = {'weights': self.weights}
        for k, v in enumerate(self.split_db):
            database = v
            if index >= len(database):
                index = index % len(database)

            index_value = database[index]
            result['A_' + str(k)] = index_value['A']
            result['B_' + str(k)] = index_value['B']
            result['A_paths_' + str(k)] = index_value['A_paths']
            result['B_paths_' + str(k)] = index_value['B_paths']

        return result

    def compute_weights(self):
        self.weights = []
        count = 0

        num_of_labels = np.ones(self.opt.n_client)

        for i in range(self.opt.n_client):
            database = self.split_db[i]
            client_len = len(database.label)
            count += client_len
            num_of_labels[i] = client_len


        self.weights = num_of_labels / count
        self.label_weight = np.zeros((self.opt.n_client, 10))
        self.label_of_client = np.ones((self.opt.n_client, 10))
        num_of_labels_uagan = np.zeros((self.opt.n_client, 10))
        for i in range(self.opt.n_client):
            database = self.split_db[i]
            for k in range(10):
                self.label_of_client[i][k] += np.sum(np.array(database.label) == k)
                num_of_labels_uagan[i][k] = np.sum(np.array(database.label) == k)

        num = np.sum(self.label_of_client, axis=1)

        self.label_info = num_of_labels_uagan

        num_of_labels_fedgen = np.transpose(num_of_labels_uagan)
        fedgen_weights = []
        for row in range(num_of_labels_fedgen.shape[0]):
            a_ = np.sum(num_of_labels_fedgen[row])
            fedgen_weights.append(num_of_labels_fedgen[row] / a_)
        self.fedgen_weights = np.array(fedgen_weights)

        self.max_num = np.max(self.label_of_client, axis=1)

        for i in range(self.label_of_client.shape[0]):
            self.label_weight[i] = self.label_of_client[i] / num[i]

    def __len__(self):
        length = 0
        for i in self.split_db:
            if len(i) > length:
                length = len(i)
        return length


