import torch
from torch import nn
from torch.autograd import grad as torch_grad
import copy
import random
import h5py,math,os
import torch.nn.functional as F

from .base_model import BaseModel
from . import networks
from parse_config import ConfigParser
import parse_config
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

class DiversityLoss(nn.Module):


    def __init__(self, metric):

        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):

        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))

class FedGENModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--nz', type=int, default=128, help='length of noise vector')
        parser.add_argument('--early_stop', type=int, default=100, help='early_stop')
        parser.add_argument("--gen_batch_size", type=int, default=32, help='number of samples from generator')
        parser.add_argument('--generative_alpha', type=int, default=10, help='generative_alpha')
        parser.add_argument('--generative_beta', type=int, default=1, help='generative_beta')
        parser.add_argument('--ensemble_alpha', type=int, default=1, help='teacher loss (server side)')
        parser.add_argument('--ensemble_beta', type=int, default=0, help='adversarial student loss')
        parser.add_argument('--ensemble_eta', type=int, default=1, help='diversity loss')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='bce')
            parser.set_defaults(norm='batch', netG='FedGEN', load_size=32, crop_size=32)
            parser.add_argument('--lambda_G', type=float, default=1, help='weight for dadgan G ')
            parser.add_argument('--lambda_reg', type=float, default=10, help='weight for gradient penalty')
        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        self.opt = opt

        self.netG = networks.define_G(opt.nz, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, opt.n_class, self.gpu_ids,
                                      opt.img_size)
        print(self.netG)

        self.netS = networks.define_D(opt.input_nc, opt.ndf, 'Classifier', opt.n_layers_D, opt.norm, opt.init_type,
                                      opt.init_gain, opt.n_class, self.gpu_ids,
                                      opt.img_size)
        if self.isTrain:
            self.netC = []
            for i in range(opt.n_client):
                self.netC.append(networks.define_D(opt.input_nc, opt.ndf, 'Classifier', opt.n_layers_D,
                                                   opt.norm, opt.init_type, opt.init_gain, opt.n_class, self.gpu_ids,
                                                   opt.img_size))

        if self.isTrain:

            self.adversarial_loss = networks.GANLoss('bce').to(self.device)
            self.auxiliary_loss = networks.GANLoss('ce').to(self.device)
            self.nll_loss = nn.NLLLoss().to(self.device)
            self.ensemble_loss = nn.KLDivLoss(reduction="batchmean").to(self.device)
            self.diversity_loss = DiversityLoss(metric='l1').to(self.device)


            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr_G, betas=(opt.beta1, 0.999))
            self.optimizer_S = torch.optim.Adam(self.netS.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_C = []
            self.lr_scheduler = []
            for i in self.netC:
                opt_C = torch.optim.Adam(i.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.lr_scheduler.append(torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_C, gamma=0.99))
                self.optimizer_C.append(opt_C)

        self.weights = []

        self.n_teacher_iters = 5

        self.available_labels = np.array([i for i in range(opt.n_class)])

        self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_G, gamma=0.98)


        for net in self.netC:
            net.load_state_dict(self.netC[0].state_dict())

    def optimize_parameters(self):
        pass
    def forward(self):
        pass

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

                self.fedgen_weights = input['fedgen_weights'][0].to(self.device).float()
        else:
            self.real_A = input['A'].to(self.device)
            self.real_B = input['B'].to(self.device)
            self.image_paths = input['A_paths']

    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        lr= max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def train_C(self,epoch):
        glob_iter = epoch / self.opt.num_epochs

        for i in range(len(self.real_A)):
            self.netC[i].train()
            self.optimizer_C[i].zero_grad()
            self.real_B[i].requires_grad_()

            img = self.real_B[i]
            label = self.real_A[i]


            pred_real = self.netC[i](img)
            user_output_logp = F.log_softmax(pred_real, dim=1)

            predictive_loss = self.nll_loss(user_output_logp, label)

            if epoch % self.opt.num_epochs < self.opt.early_stop:
                generative_alpha = self.exp_lr_scheduler(glob_iter, decay=0.98,init_lr=self.opt.generative_alpha)
                generative_beta = self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.opt.generative_beta)

                gen_output = self.netG(label)['output']
                logit_given_gen = self.netC[i](gen_output, start_layer_idx = -1)

                target_p = F.softmax(logit_given_gen, dim=1).clone().detach()
                user_latent_loss = generative_beta * self.ensemble_loss(user_output_logp, target_p)
                sampled_y = np.random.choice(self.available_labels, self.opt.gen_batch_size)
                sampled_y = torch.tensor(sampled_y).to(self.device)
                gen_result = self.netG(sampled_y)

                gen_output = gen_result['output']
                user_output_logp_ = self.netC[i](gen_output, start_layer_idx = -1)
                user_output_logp =  F.log_softmax(user_output_logp_, dim=1)
                teacher_loss = generative_alpha * torch.mean(
                    self.nll_loss(user_output_logp, sampled_y)
                )
                gen_ratio = self.opt.gen_batch_size / self.opt.ctrain_batch_size
                loss = predictive_loss + gen_ratio * teacher_loss + user_latent_loss
            else:
                loss = predictive_loss

            loss.backward()
            self.optimizer_C[i].step()

    def train_G(self):

        TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS, STUDENT_LOSS2 = 0, 0, 0, 0

        def update_generator_(n_iters, student_model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS):
            student_model.eval()
            for i in range(n_iters):
                self.netG.train()
                self.optimizer_G.zero_grad()
                y=np.random.choice(self.available_labels, self.opt.ctrain_batch_size)
                y_input=torch.LongTensor(y).to(self.device)
                gen_result=self.netG(y_input, verbose=True)
                gen_output, eps=gen_result['output'], gen_result['eps']
                teacher_loss=0
                teacher_logit=0
                for i in range(self.opt.n_client):
                    self.netC[i].eval()
                    weight=self.fedgen_weights[y][:, i].reshape(-1, 1)
                    expand_weight = np.tile(weight.cpu(), (1, self.opt.n_class))
                    user_result_given_gen=self.netC[i](gen_output, start_layer_idx = -1)
                    user_output_logp_=F.log_softmax(user_result_given_gen, dim=1)
                    teacher_loss_=torch.mean(
                        self.nll_loss(user_output_logp_, y_input) * \
                        torch.tensor(weight, dtype=torch.float32).to(self.device))
                    teacher_loss+=teacher_loss_
                    teacher_logit+=user_result_given_gen * torch.tensor(expand_weight, dtype=torch.float32).to(self.device)

                student_output=student_model(gen_output, start_layer_idx = -1)
                student_loss=F.kl_div(F.log_softmax(student_output, dim=1), F.softmax(teacher_logit, dim=1))
                if self.opt.ensemble_beta > 0:
                    loss=self.opt.ensemble_alpha * teacher_loss - self.opt.ensemble_beta * student_loss + self.opt.ensemble_eta * diversity_loss
                else:
                    loss=self.opt.ensemble_alpha * teacher_loss + self.opt.ensemble_eta * diversity_loss
                loss.backward()
                self.optimizer_G.step()
                TEACHER_LOSS += self.opt.ensemble_alpha * teacher_loss
                STUDENT_LOSS += self.opt.ensemble_beta * student_loss
                DIVERSITY_LOSS += self.opt.ensemble_eta * diversity_loss
            return TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS

        for i in range(self.opt.num_epochs):
            TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS=update_generator_(self.n_teacher_iters, self.netS, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)

        TEACHER_LOSS = TEACHER_LOSS.detach().cpu().numpy() / (self.n_teacher_iters * self.opt.num_epochs)
        STUDENT_LOSS = STUDENT_LOSS.detach().cpu().numpy() / (self.n_teacher_iters * self.opt.num_epochs)
        DIVERSITY_LOSS = DIVERSITY_LOSS.detach().cpu().numpy() / (self.n_teacher_iters * self.opt.num_epochs)
        info = "Generator: Teacher Loss= {:.4f}, Student Loss= {:.4f}, Diversity Loss = {:.4f}, ". \
            format(TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)
        print(info)
        self.generative_lr_scheduler.step()

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
        glob_iter = epoch / self.opt.num_epochs
        for i in range(len(self.real_A)):
            solns.append(self.netC[i].state_dict())
            self.lr_scheduler[i].step(glob_iter)

        result = self.aggregate_parameters_weighted(solns)
        self.netS.load_state_dict(result)
        for i in range(len(self.real_A)):
            self.netC[i].load_state_dict(result)

        print('>> aggregate client model at epoch({})'.format(epoch))
