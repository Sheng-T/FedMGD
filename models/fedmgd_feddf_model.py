import torch
from torch import nn

import h5py, math, os

from .base_model import BaseModel
from . import networks



class FedMGDFedDFModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(norm='batch', netG='cDCGANResnet', netD='DCGANResnet', load_size=32, crop_size=32)
        parser.add_argument('--nz', type=int, default=128, help='length of noise vector')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='bce')
            parser.add_argument('--lambda_G', type=float, default=1, help='weight for dadgan G ')
            parser.add_argument('--lambda_D', type=float, default=0.1, help='weight for dadgan D')
            parser.add_argument('--lambda_reg', type=float, default=10, help='weight for gradient penalty')
            parser.add_argument('--alpha', type=float, default=0.1, help='Distillation loss ratio')

        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        self.opt = opt
        self.loss_names = ['G_GAN_all', 'D_real_all', 'D_fake_all', 'D_GAN_all']
        if self.isTrain:
            self.visual_names = ['fake_B_0', 'real_B_0', 'fake_B_1', 'real_B_1', 'fake_B_2', 'real_B_2', 'fake_B_3']
        else:
            self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        self.netG = networks.define_G(opt.nz, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, opt.n_class, self.gpu_ids,
                                      opt.img_size)

        self.netS = networks.define_D(opt.input_nc, opt.ndf, 'Classifier', opt.n_layers_D, opt.norm, opt.init_type,
                                      opt.init_gain, opt.n_class, self.gpu_ids,
                                      opt.img_size)
        if self.isTrain:
            self.netD = []
            self.netC = []
            for i in range(self.opt.n_client):
                self.netD.append(networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                                   opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.n_class,
                                                   self.gpu_ids,
                                                   opt.img_size))
                self.netC.append(networks.define_D(opt.input_nc, opt.ndf, 'Classifier', opt.n_layers_D,
                                                   opt.norm, opt.init_type, opt.init_gain, opt.n_class, self.gpu_ids,
                                                   opt.img_size))

        if self.isTrain:

            self.adversarial_loss = networks.GANLoss('bce').to(self.device)
            self.auxiliary_loss = networks.GANLoss('ce').to(self.device)
            self.kl_loss = networks.GANLoss('kl').to(self.device)

            self.optimizer_S = torch.optim.Adam(self.netS.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr_G, betas=(opt.beta1, 0.999))
            self.optimizer_D = []
            for i in self.netD:
                opt_D = torch.optim.Adam(i.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999))
                self.optimizer_D.append(opt_D)
                self.optimizers.append(opt_D)
            self.optimizers.append(self.optimizer_G)

            self.optimizer_C = []
            for i in self.netC:
                opt_C = torch.optim.Adam(i.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_C.append(opt_C)

        self.weights = []

        for net in self.netC:
            net.load_state_dict(self.netS.state_dict())

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

            if len(self.weights) == 0:
                self.weights = input['weights'][0].to(self.device).float()
                print(self.weights)
        else:
            self.real_A = input['A'].to(self.device)
            self.real_B = input['B'].to(self.device)
            self.image_paths = input['A_paths']

    def forward(self):

        noise = torch.randn(self.real_A[0].size(0), self.opt.nz, 1, 1).to(self.device)
        self.rand_label = torch.randint(self.opt.n_class, (1, self.real_A[0].size(0))).squeeze().to(self.device)
        onehot_label = self.onehot[self.rand_label].to(self.device)
        self.fake_B = self.netG(noise, onehot_label)

        self.fake_B_0 = self.fake_B[0:1]
        self.fake_B_1 = self.fake_B[1:2]
        self.fake_B_2 = self.fake_B[2:3]
        self.fake_B_3 = self.fake_B[3:4]

    def test(self):
        with torch.no_grad():
            noise = torch.randn(self.real_A.size(0), self.opt.nz, 1, 1).to(self.device)
            onehot_label = self.onehot[self.real_A].to(self.device)
            self.fake_B = self.netG(noise, onehot_label)
            self.compute_visuals()

    def backward_D(self):
        self.loss_D_fake = []
        self.loss_D_real = []
        self.loss_D_reg = []
        for i in range(len(self.real_A)):
            pred_fake = self.netD[i](self.fake_B.detach())
            self.loss_D_fake.append(self.adversarial_loss(pred_fake, False))

            self.real_B[i].requires_grad_()
            pred_real = self.netD[i](self.real_B[i])
            self.loss_D_real.append(self.adversarial_loss(pred_real, True))

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

        self.loss_D_GAN_all = (
                                          self.loss_D_fake_all + self.loss_D_real_all) * self.opt.lambda_D + self.loss_D_reg_all * self.opt.lambda_reg
        self.loss_D_GAN_all.backward()

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
        # print(self.loss_C)
        self.loss_C.backward()

    def backward_G(self):
        pred_fake_weighted = torch.zeros(len(self.real_A), self.fake_B.shape[0], 1)
        aux_fake_weighted = torch.zeros(len(self.real_A), self.fake_B.shape[0], self.opt.n_class)

        choose_fake_weighted = None
        for i in range(len(self.real_A)):
            fake_pred = self.netD[i](self.fake_B)
            fake_aux = self.netC[i](self.fake_B)

            fake_aux_score = self.loss_fun(fake_aux, self.rand_label)

            score = fake_pred + fake_aux_score

            if choose_fake_weighted is None:
                choose_fake_weighted = score
            else:
                choose_fake_weighted = torch.cat((choose_fake_weighted, score), 1)

            pred_fake_weighted[i] = fake_pred
            aux_fake_weighted[i] = fake_aux

        max_score, index = choose_fake_weighted.max(1)

        aux_fake_upload = torch.zeros(self.fake_B.shape[0], self.opt.n_class)
        pred_fake_upload = torch.zeros(self.fake_B.shape[0], 1)

        for i in range(self.fake_B.shape[0]):
            aux_fake_upload[i] = aux_fake_weighted[index[i]][i]
            pred_fake_upload[i] = pred_fake_weighted[index[i]][i]

        pred_fake_upload = pred_fake_upload.to(self.device)
        aux_fake_upload = aux_fake_upload.to(self.device)

        self.loss_G_GAN_all = self.adversarial_loss(pred_fake_upload, True) + self.auxiliary_loss(aux_fake_upload,
                                                                                                  self.rand_label)
        self.loss_G = self.loss_G_GAN_all
        self.loss_G.backward()

    def loss_fun(self, input, target):
        classifier_score = []
        for i in range(target.shape[0]):
            a = torch.unsqueeze(input[i], 0)
            b = torch.tensor([target[i]]).to(self.device)
            output = - self.auxiliary_loss(a, b)
            classifier_score.append(output)
        classifier_score = torch.tensor(classifier_score).unsqueeze(1).to(self.device)
        return classifier_score

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad(self.netD, False)

        for opt in self.optimizer_C:
            opt.zero_grad()
        self.backward_C()
        for opt in self.optimizer_C:
            opt.step()

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

    def test_C(self, dataloader, fold):
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
                file = self.save_dir + '/{}_client{}_test_acc.txt'.format(fold, i)
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
        return loss, correct, num_all_samples

    def test_and_save(self, dataloader, fold, mode='global'):
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

        print('In fold {}: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(fold, loss, correct, num_all_samples,
                                                                                100. * correct / num_all_samples))

        if mode == 'global':
            file = self.save_dir + '/global_test.txt'
        else:
            file = self.save_dir + f'/{fold}_aggregate_global_test.txt'
        with open(file, 'a') as f:
            f.write('In fold {}: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(fold, loss, correct,
                                                                                        num_all_samples,
                                                                                        100. * correct / num_all_samples))
            f.close()

        return loss, correct, num_all_samples, 100. * correct / num_all_samples

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

    def distillation(self, epoch):
        logist_aggregate = None

        self.netG.eval()

        noise = torch.randn(self.opt.gen_batch, self.opt.nz, 1, 1).to(self.device)
        label_origin = torch.randint(self.opt.n_class, (1, self.opt.gen_batch)).squeeze().to(
            self.device)
        onehot_label = self.onehot[label_origin].to(self.device)
        data_origin = self.netG(noise, onehot_label)

        with torch.no_grad():

            pred_origin = self.netS(data_origin)
            pred_softmax_origin = nn.functional.softmax(pred_origin)
            values, indices = torch.max(pred_softmax_origin, 1)

            screen_img_ = []
            screen_label_ = []

            screen_img_count = 0
            for i in range(indices.shape[0]):
                if indices[i] == label_origin[i]:
                    screen_img_count += 1
                    screen_img_.append(data_origin[i])
                    screen_label_.append(label_origin[i])
            if len(screen_label_) > 1:
                screen_label = [y.unsqueeze(dim=0) for y in screen_label_]
                screen_label = torch.cat(screen_label, dim=0)
                screen_label = screen_label.squeeze(-1)

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
        if len(screen_label_) > 1:
            logist_aggregate = logist_aggregate / self.opt.n_client

            print('>> aggregate client logist at epoch({})'.format(epoch))

            self.train_S(screen_img, screen_label, logist_aggregate)

        distillation_result = self.netS.state_dict()

        for i in range(len(self.real_A)):
            self.netC[i].load_state_dict(distillation_result)

        print('>> distillation client logist at epoch({})'.format(epoch))

    def train_S(self, data, label, logist_aggregate):
        self.netS.train()

        self.optimizer_S.zero_grad()

        pred_student = self.netS(data.detach())

        loss_hard = self.auxiliary_loss(pred_student, label)

        loss_soft = self.kl_loss(pred_student, logist_aggregate)

        loss = (1 - self.opt.alpha) * loss_hard + self.opt.alpha * loss_soft

        loss.backward()

        self.optimizer_S.step()


