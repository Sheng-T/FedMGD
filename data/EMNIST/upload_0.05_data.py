import numpy as np
import torch,h5py
from PIL import Image
import torchvision.transforms as transforms
import math
import pickle,os
cpath = os.path.dirname(os.path.realpath(__file__))

def load_client_data(file):
    with h5py.File(file, "r") as f:
        # print(f)
        for key in f.keys():
            print(f[key], key, f[key].name)

        image_arr = []
        label_arr = []
        images = f["images"]
        labels = f["labels"]
        keys = images.keys()

        for key in keys:
            img = images[key][()]
            label = labels[key][()]

            image_arr.append(img.tolist())
            label_arr.append(label)

        return image_arr,label_arr

def load_client(num,client_path,data_name):


    print(">>> Data is non-i.i.d. distributed")

    upload_data = h5py.File(os.path.join(client_path, 'upload_0.05_{}_unique.h5'.format(data_name)), 'w')

    train_X = []
    train_y = []
    for user in range(num):
        load_train_data,load_train_label = load_client_data(os.path.join(client_path,'train_{}_unique_{}.h5'.format(data_name,user)))

        upload_data_len = math.ceil(len(load_train_label) * 0.05)
        train_X += load_train_data[:upload_data_len]
        train_y += load_train_label[:upload_data_len]

    indices = list(range(len(train_X)))
    for idx in indices:
        upload_data.create_dataset('images/MNIST_{:d}'.format(idx), data=train_X[idx])
        upload_data.create_dataset('labels/MNIST_{:d}'.format(idx), data=train_y[idx])

    upload_data.close()
    print('>>> Save data.')

if __name__ == '__main__':
    client_num = 10
    data_path = f'./c{client_num}/NonIID_class_dirichlet_0.1'
    load_client(client_num,data_path,'EMNIST')