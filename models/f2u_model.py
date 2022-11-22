import torch
from torch import nn
from torch.autograd import grad as torch_grad
import copy

from .base_model import BaseModel
from . import networks
from parse_config import ConfigParser
import parse_config


class F2UModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='DCGANResnet', netD='DCGANResnet', load_size=32, crop_size=32)
        parser.add_argument('--nz', type=int, default=128, help='length of noise vector')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='bce')
            parser.add_argument('--lambda_G', type=float, default=1, help='weight for dadgan G ')
            parser.add_argument('--lambda_D', type=float, default=0.1, help='weight for dadgan D')
            parser.add_argument('--lambda_reg', type=float, default=10, help='weight for gradient penalty')

        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        self.loss_names = ['G_GAN_all', 'D_real_all', 'D_fake_all','D_GAN_all']
        if self.isTrain:
            self.visual_names = ['fake_B_0', 'real_B_0', 'fake_B_1', 'real_B_1', 'fake_B_2', 'real_B_2', 'fake_B_3', 'real_B_3']
        else:
            self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        self.netG = networks.define_G(opt.nz, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, opt.n_class, self.gpu_ids)

        if self.isTrain:
            self.netD = []

            for i in range(self.opt.n_client):
                self.netD.append(networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D,
                                                   opt.norm, opt.init_type, opt.init_gain, opt.n_class, self.gpu_ids))


        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr_G, betas=(opt.beta1, 0.999))
            self.optimizer_D = []
            for i in self.netD:
                opt_D = torch.optim.Adam(i.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999))
                self.optimizer_D.append(opt_D)
                self.optimizers.append(opt_D)
            self.optimizers.append(self.optimizer_G)




        self.weights = []

        print(self.netG)
        if self.isTrain:
            print(self.netD[0])

    def set_input(self, input):

        AtoB = self.opt.direction == 'AtoB'
        if self.opt.isTrain:
            self.real_A = []   # labels
            self.real_B = []   # images
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
        self.fake_B = self.netG(noise)
        self.fake_B_0 = self.fake_B[0:1]
        self.fake_B_1 = self.fake_B[1:2]
        self.fake_B_2 = self.fake_B[2:3]
        self.fake_B_3 = self.fake_B[3:4]

    def test(self):
        with torch.no_grad():
            noise = torch.randn(self.real_A.size(0), self.opt.nz, 1, 1).to(self.device)
            self.fake_B = self.netG(noise)  # label as onehot encoding in G
            self.compute_visuals()

    def backward_D(self):
        self.loss_D_fake = []
        self.loss_D_real = []
        self.loss_D_reg = []
        for i in range(len(self.real_A)):
            pred_fake = self.netD[i](self.fake_B.detach())
            self.loss_D_fake.append(self.criterionGAN(pred_fake, False))

            self.real_B[i].requires_grad_()
            pred_real = self.netD[i](self.real_B[i])
            self.loss_D_real.append(self.criterionGAN(pred_real, True))

            reg = networks.compute_grad2(pred_real, self.real_B[i]).mean()
            self.loss_D_reg.append(reg)

        self.loss_D_fake_all = None
        self.loss_D_real_all = None
        self.loss_D_reg_all = None
        for i in range(len(self.loss_D_real)):
            if self.loss_D_real_all is None:
                self.loss_D_fake_all = self.loss_D_fake[i]
                self.loss_D_real_all = self.loss_D_real[i]
                self.loss_D_reg_all = self.loss_D_reg[i]
            else:
                self.loss_D_fake_all += self.loss_D_fake[i]
                self.loss_D_real_all += self.loss_D_real[i]
                self.loss_D_reg_all += self.loss_D_reg[i]

        self.loss_D_GAN_all = (self.loss_D_fake_all + self.loss_D_real_all)*self.opt.lambda_D + self.loss_D_reg_all * self.opt.lambda_reg
        self.loss_D_GAN_all.backward()

    def backward_G(self):
        pred_fake_weighted = None
        for i in range(len(self.real_A)):
            pred_fake = self.netD[i](self.fake_B)

            if pred_fake_weighted is None:
                pred_fake_weighted = pred_fake
            else:
                pred_fake_weighted = torch.cat((pred_fake_weighted, pred_fake), 1)

        pred_fake_weighted, _ = pred_fake_weighted.max(1)
        pred_fake_weighted = pred_fake_weighted.view(pred_fake_weighted.shape[0],1)
        self.loss_G_GAN_all = self.criterionGAN(pred_fake_weighted, True)
        self.loss_G = self.loss_G_GAN_all*self.opt.lambda_G
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        for opt in self.optimizer_D:
            opt.zero_grad()
        self.backward_D()
        for opt in self.optimizer_D:
            opt.step()
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()