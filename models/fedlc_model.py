import torch
from torch import nn
from torch.autograd import grad as torch_grad
import copy
import random
import h5py,math,os

from .base_model import BaseModel
from . import networks
from parse_config import ConfigParser
import parse_config
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class FEDLCModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--nz', type=int, default=128, help='length of noise vector')
        parser.add_argument('--calibration_temp', type=float, default=0.1, help='calibration temperature')
        parser.add_argument('--rho', type=float, default=0.9,
                            help='momentum of sgd optimizer')
        parser.add_argument('--weight_decay_cl', type=float, default=1e-5,
                            help="weight decay during local training")

        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='bce')

        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        self.opt = opt

        if self.isTrain:
            self.netC = []
            for i in range(opt.n_client):
                self.netC.append(networks.define_D(opt.input_nc, opt.ndf, 'Classifier', opt.n_layers_D,
                                                   opt.norm, opt.init_type, opt.init_gain, opt.n_class, self.gpu_ids,
                                                   opt.img_size))
        if self.isTrain:
            self.adversarial_loss = networks.GANLoss('bce').to(self.device)
            self.auxiliary_loss = networks.GANLoss('ce').to(self.device)


            self.optimizer_C = []
            for i in self.netC:
                opt_C = torch.optim.SGD(filter(lambda p: p.requires_grad, i.parameters()),
                              lr=opt.lr, momentum=opt.rho, weight_decay=opt.weight_decay_cl)
                self.optimizer_C.append(opt_C)

        self.weights = []
        self.class2data = None

        for net in self.netC:
            net.load_state_dict(self.netC[0].state_dict())


    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        if self.opt.isTrain:
            self.real_A = []
            self.real_B = []
            self.image_paths = []
            for i in range(self.opt.n_client):
                self.real_A.append(input['A_' + str(i)].to(self.device))
                self.real_B.append(input['B_' + str(i)].to(self.device))
                self.image_paths.append(input['A_paths_' + str(i)])

            if len(self.weights) == 0:
                self.weights = input['weights'][0].to(self.device).float()
                self.class2data = input['fedlc_weights'][0].to(self.device)
                print(self.weights)
        else:
            self.real_A = input['A'].to(self.device)
            self.real_B = input['B'].to(self.device)
            self.image_paths = input['A_paths']

    def forward(self):
        pass

    def test(self):
        pass


    def backward_C(self):
        self.loss_C_real = []
        for i in range(len(self.real_A)):
            self.real_B[i].requires_grad_()
            pred_real = self.netC[i](self.real_B[i])
            self.loss_C_real.append(self.auxiliary_loss(pred_real, self.real_A[i]))

        self.loss_C_all = None
        for i in range(len(self.loss_C_real)):
            if self.loss_C_all is None:
                self.loss_C_all = self.loss_C_real[i]
            else:
                self.loss_C_all += self.loss_C_real[i]

        self.loss_C = self.loss_C_all
        self.loss_C.backward()


    def optimize_parameters(self):
        pass

    def train_C(self):

        for i in range(len(self.real_A)):
            self.netC[i].train()
        for opt in self.optimizer_C:
            opt.zero_grad()
        loss_C_alone = []
        for i in range(len(self.real_A)):
            self.real_B[i].requires_grad_()

            img = self.real_B[i]
            label = self.real_A[i]

            pred_real = self.netC[i](img)

            # calibrate logit
            cl = self.opt.calibration_temp * self.class2data[i].unsqueeze(dim=0) ** (-0.25)
            pred_real -= cl

            loss = self.auxiliary_loss(pred_real, label)
            loss_C_alone.append(loss)

        loss_C_all = None
        for i in range(len(loss_C_alone)):
            if loss_C_all is None:
                loss_C_all = loss_C_alone[i]
            else:
                loss_C_all += loss_C_alone[i]

        loss_C_all.backward()
        for opt in self.optimizer_C:
            opt.step()

    def count_correct(self, pred, targets):
        _, predicted = torch.max(pred, 1)
        correct = predicted.eq(targets).sum()
        return correct

    def test_C(self, dataloader,fold):
        num_all_samples = [0 for i in range(self.opt.n_client)]
        loss = [0 for i in range(self.opt.n_client)]
        correct = [0 for i in range(self.opt.n_client)]

        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):

                for i in range(len(self.real_A)):
                    self.netC[i].eval()

                    y = data['A_' + str(i)].to(self.device)
                    X = data['B_' + str(i)].to(self.device)

                    pred = self.netC[i](X)
                    loss[i] += self.auxiliary_loss(pred, y).item()
                    correct[i] += self.count_correct(pred, y)

                    num_samples = y.size(0)

                    num_all_samples[i] += num_samples

            for i in range(len(self.real_A)):
                loss[i] = torch.true_divide(loss[i], batch_idx).item()
                file = self.save_dir + '/{}_client{}_test_acc.txt'.format(fold,i)
                print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(loss[i], correct[i],
                                                                                      num_all_samples[i],
                                                                                      100. * correct[i] /
                                                                                      num_all_samples[i]))
                with open(file, 'a') as f:

                    f.write('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(loss[i], correct[i],
                                                                                              num_all_samples[i],
                                                                                              100. * correct[i] /
                                                                                              num_all_samples[i]))
                    f.close()
        return loss , correct , num_all_samples

    def test_and_save(self, dataloader, fold, mode = 'global'):
        num_all_samples = 0
        loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                self.netC[0].eval()

                y = data['A_0'].to(self.device)
                X = data['B_0'].to(self.device)

                pred = self.netC[0](X)
                loss += self.auxiliary_loss(pred, y).item()
                correct += self.count_correct(pred, y)
                num_samples = y.size(0)
                num_all_samples += num_samples

        loss = torch.true_divide(loss, batch_idx).item()

        print('In fold {}: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(fold, loss, correct, num_all_samples,  100. * correct /num_all_samples))

        if mode=='global':
            file = self.save_dir + '/global_test.txt'
        else:
            file = self.save_dir + f'/{fold}_aggregate_global_test.txt'
        with open(file, 'a') as f:
            f.write('In fold {}: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(fold, loss, correct,
                                                                                      num_all_samples,
                                                                                      100. * correct / num_all_samples))
            f.close()


        return loss, correct, num_all_samples,  100. * correct /num_all_samples


    def save_C(self):
        for i in range(self.opt.n_client):
            save_filename = 'client%s_net_C.pth' % (i)
            save_path = os.path.join(self.save_dir, save_filename)
            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(self.netC[i].module.cpu().state_dict(), save_path)
                self.netC[i].cuda(self.gpu_ids[0])
            else:
                torch.save(self.netC[i].cpu().state_dict(), save_path)


    def aggregate_parameters_weighted(self, solns):
        result = dict()
        for p_name in solns[0].keys():
            new = torch.zeros_like(solns[0][p_name])
            for factor, sol in zip(self.weights, solns):
                new.add_(sol[p_name], alpha=factor)
            result[p_name] = new
        return result

    def aggregate(self, epoch):
        solns = []
        for i in range(len(self.real_A)):
            solns.append(self.netC[i].state_dict())

        result = self.aggregate_parameters_weighted(solns)

        for i in range(len(self.real_A)):
            self.netC[i].load_state_dict(result)

        print('>> aggregate client model at epoch({})'.format(epoch))
