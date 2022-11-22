
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
    train_dataset = create_dataset(opt,'train',opt.batch_size)

    ctrain_dataset = create_dataset(opt,'train',opt.ctrain_batch_size)
    ctest_dataset = create_dataset(opt, 'test',opt.ctest_batch_size)
    gtest_dataset = create_dataset(opt, 'global', opt.ctest_batch_size)

    dataset_size = len(train_dataset)
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    g_acc = []
    g_loss = []
    c_acc = [0 for _ in range(opt.n_client)]
    c_loss = [0 for _ in range(opt.n_client)]
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_dataset):
            sys.stdout.flush()
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0: # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

    for k in range(opt.n_fold):
        print(f'run in {k} fold:')
        for epoch in range(opt.epoch_count, opt.rounds * opt.num_epochs + 1):
            print('>> train C in ({})/({})'.format(epoch, opt.rounds + 1))
            for i, data in enumerate(ctrain_dataset):
                model.set_input(data)
                model.train_C()

            # if epoch % opt.num_epochs == 0:
            if epoch % 1 == 0:
                model.sever_train(epoch)
                model.test_and_save(gtest_dataset, k, 'aggregate')

            closs, c_correct, c_num_all_samples = model.test_C(ctest_dataset, k)
            gc.collect()
            torch.cuda.empty_cache()
            if epoch % opt.save_epoch_freq == 0:
                model.save_C()

            if epoch == opt.rounds:
                for i in range(opt.n_client):
                    c_acc[i] += (100. * c_correct[i] / c_num_all_samples[i]).cpu().numpy().tolist()
                    c_loss[i] += closs[i]
        loss, correct, num_all_samples, acc = model.test_and_save(gtest_dataset, k, 'global')
        g_acc.append(acc.cpu().numpy().tolist())
        g_loss.append(loss)

        gc.collect()
        torch.cuda.empty_cache()
    print(f'total result acc:{np.mean(g_acc)}, loss:{np.mean(g_loss)}')

    file = save_dir + '/result.txt'
    with open(file, 'a') as f:
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