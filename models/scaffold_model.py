import torch
from torch import nn
from torch.autograd import grad as torch_grad
import copy
import random
import os
import gc
from .base_model import BaseModel
from . import networks

class  SCAFFOLDModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.add_argument('--nz', type=int, default=128, help='length of noise vector')
        parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='bce')

        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        self.opt = opt

        if self.isTrain:
            self.local_controls = []
            for i in range(opt.n_client):
                self.local_controls.append(networks.define_D(opt.input_nc, opt.ndf, 'Classifier', opt.n_layers_D,
                                                   opt.norm, opt.init_type, opt.init_gain, opt.n_class, self.gpu_ids,
                                                   opt.img_size))
            self.netC = []
            for i in range(opt.n_client):
                self.netC.append(networks.define_D(opt.input_nc, opt.ndf, 'Classifier', opt.n_layers_D,
                                                   opt.norm, opt.init_type, opt.init_gain, opt.n_class, self.gpu_ids,
                                                   opt.img_size))

            self.global_model = networks.define_D(opt.input_nc, opt.ndf, 'Classifier', opt.n_layers_D, opt.norm, opt.init_type,
                                          opt.init_gain, opt.n_class, self.gpu_ids)

            self.control_global = networks.define_D(opt.input_nc, opt.ndf, 'Classifier', opt.n_layers_D, opt.norm,
                                                  opt.init_type,
                                                  opt.init_gain, opt.n_class, self.gpu_ids)


            self.control_weights = self.control_global.state_dict()
            for net in self.local_controls:
                net.load_state_dict(self.control_weights)

            for net in self.netC:
                net.load_state_dict(self.global_model.state_dict())


        if self.isTrain:
            self.adversarial_loss = networks.GANLoss('bce').to(self.device)
            self.auxiliary_loss = networks.GANLoss('ce').to(self.device)


            self.optimizer_C = []
            for i in self.netC:
                opt_C = torch.optim.Adam(i.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_C.append(opt_C)

        self.weights = []
        self.count = 0



    def set_input(self, input):

        AtoB = self.opt.direction == 'AtoB'
        if self.opt.isTrain:
            self.real_A = []  # labels
            self.real_B = []  # images
            self.image_paths = []
            for i in range(self.opt.n_client):
                self.real_A.append(input['A_' + str(i)].to(self.device))
                self.real_B.append(input['B_' + str(i)].to(self.device))
                self.image_paths.append(input['A_paths_' + str(i)])

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

    def train_C(self,epoch):
        self.count += 1
        control_global_w = self.control_global.state_dict()  # c

        for i in range(self.opt.n_client):
            control_local_w = self.local_controls[i].state_dict()  # ci
            self.netC[i].train()

            self.optimizer_C[i].zero_grad()

            global_weights = self.global_model.state_dict()

            self.real_B[i].requires_grad_()

            img = self.real_B[i]
            label = self.real_A[i]

            pred_real = self.netC[i](img)
            loss = self.auxiliary_loss(pred_real, label)

            loss.backward()
            self.optimizer_C[i].step()

            local_weights = self.netC[i].state_dict()
            for w in local_weights:
                # line 10 in algo
                # yi←yi−η(gi(yi)−ci+c)
                local_weights[w] = local_weights[w] - self.opt.lr * (control_global_w[w] - control_local_w[w])

            self.netC[i].load_state_dict(local_weights)

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

    def aggregate(self, epoch):
        print(f'aggregate in {epoch}')
        delta_c = copy.deepcopy(self.global_model.state_dict())  # ∆c
        # sum of delta_y / sample size
        delta_x = copy.deepcopy(self.global_model.state_dict())  # ∆x

        for ci in delta_c:
            delta_c[ci] = 0.0
        for ci in delta_x:
            delta_x[ci] = 0.0

        control_global_w = self.control_global.state_dict()
        global_weights = self.global_model.state_dict()

        for i in range(self.opt.n_client):
            new_control_local_w = self.local_controls[i].state_dict()  # ci+
            control_local_w = self.local_controls[i].state_dict() # ci

            control_delta = copy.deepcopy(control_local_w)
            model_weights = self.netC[i].state_dict()
            local_delta = copy.deepcopy(model_weights)
            for w in model_weights:
                new_control_local_w[w] = new_control_local_w[w] - control_global_w[w] + (
                        global_weights[w] - model_weights[w]) / (self.count * self.opt.lr)
                # line 13
                control_delta[w] = new_control_local_w[w] - control_local_w[w]  # Δci
                local_delta[w] -= global_weights[w]  # Δyi

            if epoch != 0:
                self.local_controls[i].load_state_dict(new_control_local_w) # ci ← ci+

            weights = self.netC[i].state_dict()

            for w in delta_c:
                if epoch == 0:
                    delta_x[w] += weights[w]
                else:
                    delta_x[w] += local_delta[w]  # Δyi
                    delta_c[w] += control_delta[w]  # Δci

        gc.collect()
        torch.cuda.empty_cache()

        for w in delta_c:
            delta_c[w] /= self.opt.n_client
            delta_x[w] /= self.opt.n_client

        control_global_W = self.control_global.state_dict()
        global_weights = self.global_model.state_dict()

        for w in control_global_W:
            if epoch == 0:
                global_weights[w] = delta_x[w]
            else:
                global_weights[w] += delta_x[w]
                control_global_W[w] += delta_c[w]

        self.control_global.load_state_dict(control_global_W)
        self.global_model.load_state_dict(global_weights)

        for i in range(self.opt.n_client):
            self.netC[i].load_state_dict(global_weights)

        self.count = 0