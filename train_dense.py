import sys
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import os
import gc
import torch
import numpy as np
from matplotlib import pyplot as plt
import re
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    opt = TrainOptions().parse()

    gtest_dataset = create_dataset(opt, 'global', opt.ctest_batch_size)
    ctrain_dataset = create_dataset(opt, 'train', opt.ctrain_batch_size)
    ctest_dataset = create_dataset(opt, 'test', opt.ctest_batch_size)

    dataset_size = len(ctrain_dataset)
    print('The number of training images = %d' % dataset_size)
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    g_acc = []
    g_loss = []
    cost_time_mid = []
    cost_time = []

    c_acc = [0 for _ in range(opt.n_client)]
    c_loss = [0 for _ in range(opt.n_client)]
    for k in range(opt.n_fold):
        print(f'run in {k} fold:')

        model = create_model(opt)
        model.setup(opt)
        epoch_start_time = time.time()

        model.train_C(ctrain_dataset, ctest_dataset,k)

        epoch_mid_time = time.time()

        for epoch in range(opt.num_epochs * 100):

            model.train_G()

            model.train_S()

            if epoch % 10==0:
                model.test_and_save(gtest_dataset, k, 'aggregate')

        epoch_end_time = time.time()

        cost_time_mid.append(epoch_mid_time - epoch_start_time)
        cost_time.append(epoch_end_time - epoch_mid_time)
        print(f'cost mid time : {epoch_mid_time - epoch_start_time}, cost end time :{epoch_end_time - epoch_mid_time}')

        model.save_C()
        closs, c_correct, c_num_all_samples = model.test_C(ctest_dataset, k)
        for i in range(opt.n_client):
            c_acc[i] += (100. * c_correct[i] / c_num_all_samples[i]).cpu().numpy().tolist()
            c_loss[i] += closs[i]

        loss, correct, num_all_samples, acc = model.test_and_save(gtest_dataset,k,'global')
        g_acc.append(acc.cpu().numpy().tolist())
        g_loss.append(loss)

        gc.collect()
        torch.cuda.empty_cache()

    print(f'total result acc:{np.mean(g_acc)}, loss:{np.mean(g_loss)}')

    file = save_dir + '/result.txt'
    with open(file, 'a') as f:
        f.write(f'cost mid time : {np.mean(cost_time_mid)}, cost end time :{np.mean(cost_time)}\n')
        f.write(f'global total result acc:{np.mean(g_acc)}, loss:{np.mean(g_loss)}\n')
        for i in range(opt.n_client):
            f.write(f'client {i} local result acc:{c_acc[i] / opt.n_fold}, loss:{c_loss[i] / opt.n_fold}\n')
        f.close()

    for fold in range(opt.n_fold):
        with open(save_dir + f'/{fold}_aggregate_global_test.txt', 'r', encoding='utf-8') as f:
            contents = f.readlines()
            x = []
            for j in contents:
                matchObj = re.match(r'.*\((.*)%\)', j, re.M | re.I)
                if matchObj:
                    val = float(matchObj.group(1))
                    x.append(val)
                else:
                    print("No match!!")
            if len(x) != 0:
                plt.xlabel('epoch')
                plt.ylabel('acc')
                plt.title(f'{opt.n_fold} fold result')
                plt.plot(x, label='fold {}'.format(fold))
                plt.legend(loc='best')
    plt.savefig(save_dir+ '/fold_result.png')
    plt.show()
