from tqdm import trange
import numpy as np
import random
import json
import os, math
import argparse
from torchvision.datasets import EMNIST
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import h5py

random.seed(42)
np.random.seed(42)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

def rearrange_data_by_class(data, targets, n_class):
    new_data = []
    for i in trange(n_class):
        idx = targets == i
        new_data.append(data[idx])
    return new_data

def get_dataset(split='letters'):

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = EMNIST(root='./data', split=split, train=True , download=True,transform=transform)
    test_dataset = EMNIST(root='./data', split=split, train=True, download=True, transform=transform)
    n_sample = len(train_dataset.data)
    SRC_N_CLASS = len(train_dataset.classes)

    test_mnist_random = {'data': test_dataset.data.numpy(), 'targets': test_dataset.targets.numpy()}
    state = np.random.get_state()
    np.random.shuffle(test_mnist_random['data'])
    np.random.set_state(state)
    np.random.shuffle(test_mnist_random['targets'])

    public_test_len = math.ceil(test_dataset.targets.shape[0] * 0.1)
    public_test = {'data': test_mnist_random['data'][:public_test_len],'targets': test_mnist_random['targets'][:public_test_len]-1}

    dataset = {
        'data': np.concatenate((train_dataset.data, test_mnist_random['data'][public_test_len:]), axis=0),
        'targets': np.concatenate((train_dataset.targets-1, test_mnist_random['targets'][public_test_len:]-1), axis=0)}

    n_sample = len(dataset['data'])
    SRC_N_CLASS = len(train_dataset.classes)

    print("Rearrange data by class...")
    data_by_class = rearrange_data_by_class(
        dataset['data'],
        dataset['targets'],
        SRC_N_CLASS
    )
    print(f"Data SET:\n  Total #samples: {n_sample}. sample shape: {dataset['data'][0].shape}")
    print("  #samples per class:\n", [len(v) for v in data_by_class])

    print("  #public test data length:{}".format(len(public_test['data'])))
    return data_by_class, public_test, n_sample, SRC_N_CLASS

def sample_class(SRC_N_CLASS, NUM_LABELS, user_id, label_random=False):
    assert NUM_LABELS <= SRC_N_CLASS
    if label_random:
        source_classes = [n for n in range(SRC_N_CLASS)]
        random.shuffle(source_classes)
        return source_classes[:NUM_LABELS]
    else:
        return [(user_id + j) % SRC_N_CLASS for j in range(NUM_LABELS)]

def devide_train_data(data, n_sample, SRC_CLASSES, NUM_USERS, min_sample, alpha=0.5, sampling_ratio=0.5):

    min_sample = 10
    min_size = 0

    while min_size < min_sample:
        print("Try to find valid data separation")
        idx_batch=[{} for _ in range(NUM_USERS)]
        samples_per_user = [0 for _ in range(NUM_USERS)]
        max_samples_per_user = sampling_ratio * n_sample / NUM_USERS
        for l in SRC_CLASSES:
            idx_l = [i for i in range(len(data[l]))]
            np.random.shuffle(idx_l)
            if sampling_ratio < 1:
                samples_for_l = int( min(max_samples_per_user, int(sampling_ratio * len(data[l]))) )
                idx_l = idx_l[:samples_for_l]
                print(l, len(data[l]), len(idx_l))

            proportions=np.random.dirichlet(np.repeat(alpha, NUM_USERS))
            proportions=np.array([p * (n_per_user < max_samples_per_user) for p, n_per_user in zip(proportions, samples_per_user)])
            proportions=proportions / proportions.sum()
            proportions=(np.cumsum(proportions) * len(idx_l)).astype(int)[:-1]
            for u, new_idx in enumerate(np.split(idx_l, proportions)):
                idx_batch[u][l] = new_idx.tolist()
                samples_per_user[u] += len(idx_batch[u][l])
        min_size=min(samples_per_user)
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    Labels=[set() for _ in range(NUM_USERS)]
    print("processing users...")
    for u, user_idx_batch in enumerate(idx_batch):
        for l, indices in user_idx_batch.items():
            if len(indices) == 0: continue
            X[u] += data[l][indices].tolist()
            y[u] += (l * np.ones(len(indices))).tolist()
            Labels[u].add(l)

    return X, y, Labels, idx_batch, samples_per_user

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--format", "-f", type=str, default="h5", help="Format of saving: pt (torch.save), json, h5", choices=["pt", "json", "h5"])
    parser.add_argument("--n_class", type=int, default=26, help="number of classification labels")
    parser.add_argument("--min_sample", type=int, default=500, help="Min number of samples per user.")
    parser.add_argument("--sampling_ratio", type=float, default=0.1, help="Ratio for sampling training samples.")
    parser.add_argument("--unknown_test", type=int, default=0, help="Whether allow test label unseen for each user.")
    parser.add_argument("--alpha", type=float, default=0.1, help="alpha in Dirichelt distribution (smaller means larger heterogeneity)")
    parser.add_argument("--n_user", type=int, default=5,
                        help="number of local clients, should be muitiple of 10.")
    parser.add_argument("--save_dir", type=str, default='./NonIID_class_dirichlet',
                        help="save path for file.h5 format.")
    args = parser.parse_args()
    print()
    print("Number of users: {}".format(args.n_user))
    print("Number of classes: {}".format(args.n_class))
    print("Min # of samples per uesr: {}".format(args.min_sample))
    print("Alpha for Dirichlet Distribution: {}".format(args.alpha))
    print("Ratio for Sampling Training Data: {}".format(args.sampling_ratio))
    NUM_USERS = args.n_user

    args.save_dir = f'c{args.n_user}/'+args.save_dir + f'_{args.alpha}'

    path_prefix = f'u{args.n_user}c{args.n_class}-alpha{args.alpha}-ratio{args.sampling_ratio}'

    def process_user_data( data, n_sample, SRC_CLASSES, Labels=None, unknown_test=0, public_test=None):
        if public_test is None:
            public_test = {}

        X, y, Labels, idx_batch, samples_per_user  = devide_train_data(
                data, n_sample, SRC_CLASSES, NUM_USERS, args.min_sample, args.alpha, args.sampling_ratio)

        for u in range(NUM_USERS):
            print("{} samples in total".format(samples_per_user[u]))
            train_info = ''
            n_samples_for_u = 0
            for l in sorted(list(Labels[u])):
                n_samples_for_l = len(idx_batch[u][l])
                n_samples_for_u += n_samples_for_l
                train_info += "c={},n={}| ".format(l, n_samples_for_l)
            print(train_info)
            print("{} Labels/ {} Number of training samples for user [{}]:".format(len(Labels[u]), n_samples_for_u, u))

        if args.format == "h5":
            mkdir(args.save_dir)
            train_files = []
            test_files = []
            train_X = [[] for _ in range(10)]
            train_y = [[] for _ in range(10)]
            test_X = [[] for _ in range(10)]
            test_y = [[] for _ in range(10)]
            for i in range(NUM_USERS):
                train_files.append(
                    h5py.File(os.path.join(args.save_dir, 'train_{}_unique_{:d}.h5'.format('EMNIST', i)), 'w'))
                test_files.append(
                    h5py.File(os.path.join(args.save_dir, 'test_{}_unique_{:d}.h5'.format('EMNIST', i)), 'w'))

            print('>> start split train data: 80% and test data: 20%')
            for user in range(NUM_USERS):
                state = np.random.get_state()
                np.random.shuffle(X[user])
                np.random.set_state(state)
                np.random.shuffle(y[user])

                train_len = math.ceil(len(X[user]) * 0.8)
                train_X[user] = X[user][:train_len]
                train_y[user] = y[user][:train_len]

                test_X[user] = X[user][train_len:]
                test_y[user] = y[user][train_len:]

                indices = list(range(len(train_X[user])))
                for idx in indices:
                    train_files[user].create_dataset('images/MNIST_{:d}'.format(idx), data=train_X[user][idx])
                    train_files[user].create_dataset('labels/MNIST_{:d}'.format(idx), data=train_y[user][idx])

                indices = list(range(len(test_X[user])))
                for idx in indices:
                    test_files[user].create_dataset('images/MNIST_{:d}'.format(idx), data=test_X[user][idx])
                    test_files[user].create_dataset('labels/MNIST_{:d}'.format(idx), data=test_y[user][idx])

            for i in range(NUM_USERS):
                print('{:d}: train {:d}, test {:d}'.format(i, len(train_files[i]['images']),len(test_files[i]['images'])))
                train_files[i].close()
                test_files[i].close()

            public_file = h5py.File(os.path.join(args.save_dir, 'public_test_unique.h5'), 'w')
            indices = list(range(len(public_test['data'])))
            for idx in indices:
                public_file.create_dataset('images/MNIST_{:d}'.format(idx), data=public_test['data'][idx].tolist())
                public_file.create_dataset('labels/MNIST_{:d}'.format(idx),data=public_test['targets'][idx].tolist())
            print('public test data:{}'.format(len(public_test['data'])))
            public_file.close()

            for i in range(NUM_USERS):
                str = f'client {i} set label is {set(y[i])}'
                print(str)


        return Labels, idx_batch, samples_per_user


    print(f"Reading source dataset.")
    data_by_class, public_test, n_sample, SRC_N_CLASS = get_dataset()
    SRC_CLASSES=[l for l in range(SRC_N_CLASS)]
    random.shuffle(SRC_CLASSES)
    print("{} labels in total.".format(len(SRC_CLASSES)))
    process_user_data( data_by_class, n_sample, SRC_CLASSES, public_test=public_test)

    print("Finish Generating User samples")

if __name__ == "__main__":
    main()
