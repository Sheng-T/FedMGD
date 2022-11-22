import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks


class BaseModel(ABC):

    def __init__(self, opt):

        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain

        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if opt.preprocess != 'scale_width':
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    def set_input(self, input):

        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

    def setup(self, opt):
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):

        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        pass

    def get_image_paths(self):
        return self.image_paths

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def save_networks(self, epoch):

        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                module = getattr(net, "module", None)
                if module is not None:
                    save_filename = '%s_net_%s.pth' % (epoch, name)
                    save_path = os.path.join(self.save_dir, save_filename)
                    if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                        torch.save(net.module.cpu().state_dict(), save_path)
                        net.cuda(self.gpu_ids[0])
                    else:
                        torch.save(net.cpu().state_dict(), save_path)
                else:
                    for i in range(len(net)):
                        save_filename = '%s_net_%s_%s.pth' % (epoch, name, i)
                        save_path = os.path.join(self.save_dir, save_filename)
                        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                            torch.save(net[i].module.cpu().state_dict(), save_path)
                            net[i].cuda(self.gpu_ids[0])
                        else:
                            torch.save(net[i].cpu().state_dict(), save_path)


    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                for key in list(state_dict.keys()):
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                parameters = getattr(net, "parameters", None)
                if parameters is not None:
                    for param in net.parameters():
                        num_params += param.numel()
                else:
                    for i in net:
                        for param in i.parameters():
                            num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):

        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
