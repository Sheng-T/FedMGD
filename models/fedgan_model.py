import torch
from torch import nn
from torch.autograd import grad as torch_grad
import copy

from .base_model import BaseModel
from . import networks
from parse_config import ConfigParser
import parse_config


class FedGANModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(norm='batch', netG='cDCGANResnet', netD='cDCGANResnet', load_size=32, crop_size=32)
        parser.add_argument('--nz', type=int, default=128, help='length of noise vector')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='bce')
            parser.add_argument('--lambda_G', type=float, default=1, help='weight for dadgan G ')
            parser.add_argument('--lambda_D', type=float, default=0.1, help='weight for dadgan D')
            parser.add_argument('--lambda_reg', type=float, default=10, help='weight for gradient penalty')

        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        self.loss_names = ['G_GAN_all', 'D_real_all', 'D_fake_all']
        if self.isTrain:
            self.visual_names = ['fake_B_0', 'real_B_0', 'fake_B_1', 'real_B_1', 'fake_B_2', 'real_B_2', 'fake_B_3', 'real_B_3']
        else:
            self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        if self.isTrain:
            self.netD = []
            self.netG = []
            for i in range(self.opt.n_client):
                self.netG.append(networks.define_G(opt.nz, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                              not opt.no_dropout, opt.init_type, opt.init_gain, opt.n_class,
                                              self.gpu_ids))
                self.netD.append(networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D,
                                                   opt.norm, opt.init_type, opt.init_gain, opt.n_class, self.gpu_ids))

        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.optimizer_G = []
            self.optimizer_D = []
            for i in self.netG:
                opt_G = torch.optim.Adam(i.parameters(), lr=opt.lr_G, betas=(opt.beta1, 0.999))
                self.optimizer_G.append(opt_G)
                self.optimizers.append(opt_G)
            for i in self.netD:
                opt_D = torch.optim.Adam(i.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999))
                self.optimizer_D.append(opt_D)
                self.optimizers.append(opt_D)

        self.weights = []

        self.onehot = torch.zeros(opt.n_class, opt.n_class)
        self.onehot = self.onehot.scatter_(1, torch.arange(opt.n_class).view(opt.n_class, 1), 1).view(
            opt.n_class, opt.n_class, 1, 1)
        self.fill = torch.zeros([opt.n_class, opt.n_class, opt.load_size, opt.load_size])
        for i in range(opt.n_class):
            self.fill[i, i, :, :] = 1

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

            self.real_B_0 = self.real_B[0]
            self.real_B_1 = self.real_B[1]
            self.real_B_2 = self.real_B[2]
            self.real_B_3 = self.real_B[3]

            if len(self.weights) == 0:
                self.weights = input['weights'][0].to(self.device).float()
                print(self.weights)
        else:
            self.real_A = input['A'].to(self.device)
            self.real_B = input['B'].to(self.device)
            self.image_paths = input['A_paths']

    def forward(self):

        noise = torch.randn(self.real_A[0].size(0), self.opt.nz, 1, 1).to(self.device)
        rand_label = torch.randint(self.opt.n_class, (1, self.real_A[0].size(0))).squeeze().to(
            self.device)
        onehot_label = self.onehot[rand_label].to(self.device)

        fake_B = []
        for i in range(self.opt.n_client):
            fake_B.append(self.netG[i](noise, onehot_label)[0:5])

        self.fake_B_0 = fake_B[0][0]
        self.fake_B_1 = fake_B[1][0]
        self.fake_B_2 = fake_B[2][0]
        self.fake_B_3 = fake_B[3][0]
        self.fake_B_4 = fake_B[4][0]

    def test(self):
        with torch.no_grad():
            noise = torch.randn(self.real_A[0].size(0), self.opt.nz, 1, 1).to(self.device)
            self.fake_B = self.netG[0](noise)
            self.compute_visuals()

    def aggregate_parameters_weighted(self,solns):
        result = dict()
        for p_name in solns[0].keys():
            new = torch.zeros_like(solns[0][p_name])
            if solns[0][p_name].dtype == torch.int64:
                for factor, sol in zip(self.weights, solns):
                    new += (sol[p_name] * factor).long()
            else:
                for factor, sol in zip(self.weights, solns):
                    new.add_(sol[p_name], alpha=factor)
            result[p_name] = new
        return result


    def aggregate(self, epoch):
        solns_D = []
        solns_G = []
        for i in range(len(self.real_A)):
            solns_D.append(self.netD[i].state_dict())
            solns_G.append(self.netG[i].state_dict())

        result_D = self.aggregate_parameters_weighted(solns_D)
        result_G = self.aggregate_parameters_weighted(solns_G)

        for i in range(len(self.real_A)):
            self.netD[i].load_state_dict(result_D)
            self.netG[i].load_state_dict(result_G)

        print('>> aggregate client model at epoch({})'.format(epoch))

    def train_GAN(self, i):

        self.set_requires_grad(self.netG[i], False)
        self.set_requires_grad(self.netD[i], True)
        self.optimizer_D[i].zero_grad()
        noise = torch.randn(self.real_A[0].size(0), self.opt.nz, 1, 1).to(self.device)
        rand_label = torch.randint(self.opt.n_class, (1, self.real_A[0].size(0))).squeeze().to(
            self.device)
        onehot_label = self.onehot[rand_label].to(self.device)
        fake_B = self.netG[i](noise, onehot_label)
        label_img_fake = self.fill[rand_label].to(self.device)
        pred_fake = self.netD[i](fake_B.detach(), label_img_fake)
        loss_D_fake = self.criterionGAN(pred_fake, False)

        self.real_B[i].requires_grad_()
        label_img_real = self.fill[self.real_A[i]].to(self.device)
        pred_real = self.netD[i](self.real_B[i], label_img_real)
        loss_D_real = self.criterionGAN(pred_real, True)

        reg = networks.compute_grad2(pred_real, self.real_B[i]).mean()

        if self.loss_D_real_all is None:
            self.loss_D_fake_all = loss_D_fake
            self.loss_D_real_all = loss_D_real
        else:
            self.loss_D_fake_all += loss_D_fake
            self.loss_D_real_all += loss_D_real

        loss_D =  (loss_D_fake + loss_D_real) * self.opt.lambda_D + reg * self.opt.lambda_reg
        loss_D.backward()
        self.optimizer_D[i].step()

        self.set_requires_grad(self.netG[i], True)
        self.set_requires_grad(self.netD[i], False)

        noise_G = torch.randn(self.real_A[0].size(0), self.opt.nz, 1, 1).to(self.device)
        rand_label_G = torch.randint(self.opt.n_class, (1, self.real_A[0].size(0))).squeeze().to(
            self.device)
        onehot_label_G = self.onehot[rand_label_G].to(self.device)

        fake_B_G = self.netG[i](noise_G, onehot_label_G)
        label_img_fake_G = self.fill[rand_label_G].to(self.device)
        pred_fake_G = self.netD[i](fake_B_G, label_img_fake_G)

        self.optimizer_G[i].zero_grad()
        loss_G_GAN = self.criterionGAN(pred_fake_G, True)
        loss_G = loss_G_GAN * self.opt.lambda_G
        if self.loss_G_GAN_all is None:
            self.loss_G_GAN_all = loss_G
        else:
            self.loss_G_GAN_all += loss_G
        loss_G.backward()
        self.optimizer_G[i].step()


    def optimize_parameters(self):
        self.forward()
        self.loss_D_fake_all = None
        self.loss_D_real_all = None
        self.loss_G_GAN_all = None
        self.set_requires_grad(self.netD, True)
        for i in range(self.opt.n_client):
            self.train_GAN(i)