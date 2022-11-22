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
from models.networks import cDCGANResnetGenerator,DCGANResnetGenerator
from torch.cuda.amp import autocast as autocast, GradScaler
def cdgan(model_path,input_nz, output_nc, ngf, nf_max=256, img_size=32, num_classes=10,gpu_ids=[]):

    G = cDCGANResnetGenerator(input_nz, output_nc, ngf, nf_max=256, img_size=img_size, num_classes=num_classes)
    state_dict = torch.load(model_path)
    G.load_state_dict(state_dict)

    G.to(gpu_ids[0])
    G = torch.nn.DataParallel(G, gpu_ids)

    return G

def dgan(model_path,input_nz, output_nc, ngf, nf_max=256, img_size=32, num_classes=10,gpu_ids=[]):
    model = DCGANResnetGenerator(input_nz,output_nc,ngf, nf_max=nf_max, img_size=img_size,num_classes = num_classes)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    model.to(gpu_ids[0])
    model = torch.nn.DataParallel(model, gpu_ids)  # multi-GPUs
    return model

class FEDDFAfterF2UModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(norm='batch', netG='cDCGANResnet', netD='DCGANResnet', load_size=32, crop_size=32)
        parser.add_argument('--nz', type=int, default=128, help='length of noise vector')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='bce')
            parser.add_argument('--lambda_G', type=float, default=1, help='weight for dadgan G ')
            parser.add_argument('--lambda_D', type=float, default=0.1, help='weight for dadgan D')
            parser.add_argument('--lambda_reg', type=float, default=10, help='weight for gradient penalty')
            # parser.add_argument('--G_path', type=str,default='your_path/latest_net_G.pth',help='weight for gradient penalty')
            parser.add_argument('--alpha', type=float, default=0.1, help='Distillation loss ratio')

        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        self.opt = opt

        self.netG = dgan(opt.G_path, opt.nz, opt.input_nc, opt.ngf, nf_max=256, img_size=opt.img_size,
                          num_classes=opt.n_class, gpu_ids=self.gpu_ids)

        self.netS = networks.define_D(opt.input_nc, opt.ndf, 'Classifier', opt.n_layers_D, opt.norm, opt.init_type,
                                      opt.init_gain, opt.n_class, self.gpu_ids,
                                      opt.img_size)
        if self.isTrain:

            self.netC = []
            for i in range(self.opt.n_client):

                self.netC.append(networks.define_D(opt.input_nc, opt.ndf, 'Classifier', opt.n_layers_D,
                                                   opt.norm, opt.init_type, opt.init_gain, opt.n_class, self.gpu_ids,
                                                   opt.img_size))

        if self.isTrain:

            self.adversarial_loss = networks.GANLoss('bce').to(self.device)
            self.auxiliary_loss = networks.GANLoss('ce').to(self.device)
            self.kl_loss = networks.GANLoss('kl').to(self.device)

            self.optimizer_S = torch.optim.Adam(self.netS.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr_G, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)

            self.optimizer_C = []
            for i in self.netC:
                opt_C = torch.optim.Adam(i.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_C.append(opt_C)


        self.weights = []

        for net in self.netC:
            net.load_state_dict(self.netS.state_dict())


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

            if len(self.weights) == 0:
                self.weights = input['weights'][0].to(self.device).float()
                print(self.weights)
        else:
            self.real_A = input['A'].to(self.device)
            self.real_B = input['B'].to(self.device)
            self.image_paths = input['A_paths']

    def forward(self):
        pass

    def test(self):
        pass

    def backward_D(self):
        pass

    def backward_C(self):
        self.loss_C_real = []
        for i in range(len(self.real_A)):
            self.real_B[i].requires_grad_()
            pred_real = self.netC[i](self.real_B[i])
            self.loss_C_real.append(self.auxiliary_loss(pred_real, self.real_A[i]))

        self.loss_C_all =None
        for i in range(len(self.loss_C_real)):
            if self.loss_C_all is None:
                self.loss_C_all = self.loss_C_real[i]
            else:
                self.loss_C_all += self.loss_C_real[i]

        self.loss_C = self.loss_C_all
        self.loss_C.backward()

    def backward_G(self):
        pass

    def loss_fun(self,input, target):
        classifier_score = []
        for i in range(target.shape[0]):
            a = torch.unsqueeze(input[i], 0)
            b = torch.tensor([target[i]]).to(self.device)
            output = - self.auxiliary_loss(a, b)
            classifier_score.append(output)
        classifier_score = torch.tensor(classifier_score).unsqueeze(1).to(self.device)
        return classifier_score

    def optimize_parameters(self):
        pass

    def train_C(self):

        self.netG.eval()
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
            loss  = self.auxiliary_loss(pred_real, label)
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

    def count_correct(self,pred, targets):
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

        self.netS.load_state_dict(result)

        print('>> aggregate client model at epoch({})'.format(epoch))

    def sever_train(self, epoch):
        solns = []
        for i in range(len(self.real_A)):
            solns.append(self.netC[i].state_dict())

        result = self.aggregate_parameters_weighted(solns)
        self.netS.load_state_dict(result)
        print('>> Sever update by GAN at epoch({})'.format(epoch))


        self.distillation(epoch)

    def distillation(self,epoch):
        logist_aggregate = None


        self.netG.eval()

        with torch.no_grad():

            noise = torch.randn(self.opt.gen_batch, self.opt.nz, 1, 1).to(self.device)  # 1024

            data_origin = self.netG(noise)

            pred_origin = self.netS(data_origin.detach())
            pred_softmax_origin = nn.functional.softmax(pred_origin)
            values, indices = torch.max(pred_softmax_origin, 1)

            screen_img_ = []

            screen_img_count = 0
            for i in range(indices.shape[0]):
                screen_img_count += 1
                screen_img_.append(data_origin[i])

            screen_img = [x.unsqueeze(dim=0) for x in screen_img_]
            screen_img = torch.cat(screen_img, dim=0)

            print(f'>>> This round generate distillation data num: {screen_img_count}')

            logist = []
            for i in range(self.opt.n_client):
                self.netC[i].eval()
                pred = self.netC[i](screen_img.detach())
                logist.append(pred)

            for i in range(len(logist)):
                if i == 0:
                    logist_aggregate = logist[i]
                else:
                    logist_aggregate = torch.add(logist_aggregate, logist[i])

        logist_aggregate = logist_aggregate / self.opt.n_client

        print('>> aggregate client logist at epoch({})'.format(epoch))

        self.train_S(screen_img,logist_aggregate)

        distillation_result = self.netS.state_dict()

        for i in range(len(self.real_A)):
            self.netC[i].load_state_dict(distillation_result)

        print('>> distillation client logist at epoch({})'.format(epoch))

    def train_S(self,data,logist_aggregate):
        self.netS.train()
        scaler = GradScaler()
        self.optimizer_S.zero_grad()
        with autocast():
            pred_student = self.netS(data.detach())

            loss_soft = self.kl_loss(pred_student, logist_aggregate)

            loss = loss_soft


        scaler.scale(loss).backward()


        scaler.step(self.optimizer_S)

        scaler.update()



