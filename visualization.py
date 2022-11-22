import numpy as np
import os, random
from skimage import io
import h5py

params = [{'exp_name': 'exp_name', 'netG': 'cDCGANResnet'}]
epoch = 200

for i in range(len(params)):
    exp_name = params[i]['exp_name']
    netG = params[i]['netG']
    data_dir = './results/{:s}/test_{:d}'.format(exp_name, epoch)

    h5_file = h5py.File('{:s}/{:s}_{:s}_epoch{:d}_x1.h5'.format(data_dir, exp_name, netG, epoch), 'r')

    N = 20
    all_imgs = {}
    for i in range(10):
        all_imgs[i] = []

    counter = 0
    keys = list(h5_file['images'].keys())
    random.shuffle(keys)
    for key in keys:
        label = h5_file['labels/{:s}'.format(key)][()]

        if len(all_imgs[label]) >= N:
            continue
        img = h5_file['images/{:s}'.format(key)][()]
        all_imgs[label].append(img)
        counter += 1

        if counter == 10 * N:
            break

    h, w = all_imgs[0][0].shape
    space = 2
    grid_img = np.ones([h*10+space*9, w*N+space*(N-1)], dtype=np.uint8)*128
    for i in range(10):
        for j in range(N):
            grid_img[h*i+space*i:h*(i+1)+space*i, w*j+space*j:w*(j+1)+space*j] = all_imgs[i][j]

    io.imsave('{:s}/{:s}.png'.format(data_dir, exp_name), grid_img)

