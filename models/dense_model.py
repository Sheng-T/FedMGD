import torch
from torch import nn
from torch.autograd import grad as torch_grad
import copy
import random
import h5py,math,os
import torch.nn.functional as F
import torch.utils.data
from util.dense_util import reset_model, DeepInversionHook, Ensemble_A, kldiv, KLDiv, Ensemble, ImagePool
from util.fixmatch_util import TransformFixMatch
from .base_model import BaseModel
from . import networks
from parse_config import ConfigParser
import parse_config
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

class DENSEModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--nz', type=int, default=128, help='length of noise vector')
        parser.add_argument('--dense_local_epoch', type=int, default=100, help='dense local epoch')
        parser.add_argument('--synthesis_batch_size', type=int, default=128)
        parser.add_argument('--g_steps', type=int, default=30)
        parser.add_argument('--lr_g', default=1e-3, type=float, help='initial learning rate for generation')
        parser.add_argument('--lr_s', default=0.01, type=float, help='initial learning rate for generation')
        parser.add_argument('--adv', default=1.0, type=float, help='scaling factor for adv loss')

        parser.add_argument('--bn', default=1.0, type=float, help='scaling factor for BN regularization')
        parser.add_argument('--oh', default=1.0, type=float, help='scaling factor for one hot loss (cross entropy)')
        parser.add_argument('--T_', default=10, type=float)
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='bce')

        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        self.opt = opt

        if self.isTrain:
            self.netC = []
            for i in range(opt.n_client):
                self.netC.append(networks.define_D(opt.input_nc, opt.ndf, 'ClassifierBN', opt.n_layers_D,
                                                   opt.norm, opt.init_type, opt.init_gain, opt.n_class, self.gpu_ids,
                                                   opt.img_size))
            self.netG = networks.define_G(opt.nz, opt.input_nc, opt.ngf, 'Dense', opt.norm,
                                          not opt.no_dropout, opt.init_type, opt.init_gain, opt.n_class, self.gpu_ids,
                                          opt.img_size)
            self.netS = networks.define_D(opt.input_nc, opt.ndf, 'ClassifierBN', opt.n_layers_D, opt.norm, opt.init_type,
                                      opt.init_gain, opt.n_class, self.gpu_ids,
                                      opt.img_size)

        if self.isTrain:

            self.adversarial_loss = networks.GANLoss('bce').to(self.device)
            self.auxiliary_loss = networks.GANLoss('ce').to(self.device)


            self.optimizer_S = torch.optim.SGD(self.netS.parameters(), lr=opt.lr_s,momentum=0.9)
            self.optimizer_C = []
            for i in self.netC:
                opt_C = torch.optim.SGD(i.parameters(), lr=opt.lr_s,momentum=0.9)
                self.optimizer_C.append(opt_C)

        if self.opt.input_nc == 1:
            mean = (0.5,)
            std = (0.5,)
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)

        self.aug = TransformFixMatch(mean=mean, std=std,img_size=opt.img_size,twoReturn=False)
        self.criterion = KLDiv(T=opt.T_)

        save_dir = os.path.join(opt.checkpoints_dir, opt.name, 'images')
        self.data_pool = ImagePool(root=save_dir)

        self.weights = []


    def set_input(self, input):
        pass

    def forward(self):
        pass

    def test(self):
        pass

    def optimize_parameters(self):
        pass

    def train_C(self,train_data, test_data,fold):
        for epoch in range(self.opt.dense_local_epoch):
            for batch_idx, data in enumerate(train_data):
                for i in range(self.opt.n_client):
                    self.netC[i].train()
                    self.optimizer_C[i].zero_grad()
                    y = data['A_' + str(i)].to(self.device)
                    X = data['B_' + str(i)].to(self.device)
                    X.requires_grad_()

                    pred_real = self.netC[i](X)
                    loss = self.auxiliary_loss(pred_real, y)

                    loss.backward()
                    self.optimizer_C[i].step()
            self.test_C(test_data, fold)

        self.ensemble_model = Ensemble(self.netC)

    def train_G(self):
        z = torch.randn(size=(self.opt.synthesis_batch_size, 256)).to(self.device)
        z.requires_grad = True
        targets = torch.randint(low=0, high=self.opt.n_class, size=(self.opt.synthesis_batch_size,))
        targets = targets.sort()[0]
        targets = targets.to(self.device)

        reset_model(self.netG)

        best_cost = 1e6
        best_inputs = None

        optimizer_G = torch.optim.Adam([{'params': self.netG.parameters()}, {'params': [z]}], self.opt.lr_g, betas=(0.5, 0.999))

        hooks = []
        net = Ensemble_A(self.netC)
        net.eval()
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                hooks.append(DeepInversionHook(m))

        self.netG.train()
        for it in range(self.opt.g_steps):
            print(f'>>> train G in {it}')
            optimizer_G.zero_grad()

            inputs = self.netG(z)
            global_view = self.aug(inputs)

            t_out = net(global_view)
            loss_bn = sum([h.r_feature for h in hooks])
            loss_oh = F.cross_entropy(t_out, targets)

            s_out = self.netS(global_view)
            mask = (s_out.max(1)[1] != t_out.max(1)[1]).float()
            loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(1) * mask).mean()
            loss = self.opt.bn * loss_bn + self.opt.oh * loss_oh + self.opt.adv * loss_adv

            if best_cost > loss.item() or best_inputs is None:
                best_cost = loss.item()
                best_inputs = inputs.data

            loss.backward()

            optimizer_G.step()
        self.data_pool.add(best_inputs)


    def train_S(self):
        self.netS.train()
        self.ensemble_model.eval()
        for idx, (images) in enumerate(self.get_data()):
            self.optimizer_S.zero_grad()

            images = images.to(self.device)
            with torch.no_grad():
                t_out = self.ensemble_model(images)
            s_out = self.netS(images.detach())
            loss_s = self.criterion(s_out, t_out.detach())

            loss_s.backward()
            self.optimizer_S.step()


    def get_data(self):
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        datasets = self.data_pool.get_dataset(transform=transform)
        self.data_loader = torch.utils.data.DataLoader(
            datasets, batch_size=self.opt.synthesis_batch_size, shuffle=True,
            num_workers=4, pin_memory=True, )
        return self.data_loader

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

                for i in range(self.opt.n_client):
                    self.netC[i].eval()

                    y = data['A_' + str(i)].to(self.device)
                    X = data['B_' + str(i)].to(self.device)

                    pred = self.netC[i](X)
                    loss[i] += self.auxiliary_loss(pred, y).item()
                    correct[i] += self.count_correct(pred, y)

                    num_samples = y.size(0)

                    num_all_samples[i] += num_samples

            for i in range(self.opt.n_client):
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
            save_path_ = os.path.join(self.save_dir, 'server_net_S.pth')

            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(self.netC[i].module.cpu().state_dict(), save_path)
                self.netC[i].cuda(self.gpu_ids[0])
            else:
                torch.save(self.netC[i].cpu().state_dict(), save_path)

            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(self.netS.module.cpu().state_dict(), save_path_)
                self.netC[i].cuda(self.gpu_ids[0])
            else:
                torch.save(self.netS.cpu().state_dict(), save_path_)



