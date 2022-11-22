import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torch.nn.utils import spectral_norm
from torchvision import models
import numpy as np
from itertools import chain


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):

    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):

    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            lr_l = max(lr_l, 0.1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())

        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nz, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, num_classes=10, gpu_ids=[], img_size=32):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'cDCGAN':
        net = cDCGANGenerator(input_nz, output_nc, ngf, norm_layer=nn.BatchNorm2d, num_classes=num_classes)
    elif netG == 'cDCGANResnet':
        net = cDCGANResnetGenerator(input_nz, output_nc, ngf, nf_max=256, img_size=img_size, num_classes=num_classes)
    elif netG == 'DCGANResnet':
        net = DCGANResnetGenerator(input_nz, output_nc, ngf, nf_max=256, img_size=img_size, num_classes=num_classes)
    elif netG == 'ACGANGenerator':
        net = ACGANGenerator(input_nz, output_nc,img_size=img_size, num_classes=num_classes)
    elif netG == 'FedGEN':
        net = FedGENGenerator(n_class=num_classes, input_channel=output_nc,img_size = img_size ,gpu_ids = gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, num_classes=10, gpu_ids=[], img_size = 32):

    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'cDCGAN':
        net = cDCGANDiscriminator(input_nc, ndf, norm_layer=nn.BatchNorm2d, num_classes=num_classes)
    elif netD == 'cDCGANResnet':
        net = cDCGANResnetDiscriminator(input_nc, ndf, nf_max=256, img_size=img_size, num_classes=num_classes)
    elif netD == 'DCGANResnet':
        net = DCGANResnetDiscriminator(input_nc, ndf, nf_max=256, img_size=img_size)
    elif netD =='ACGANDiscriminator':
        net = ACGANDiscriminator(input_nc,img_size=img_size,num_classes=num_classes)
    elif netD == 'Classifier':
        net = Classifier(input_nc,img_size=img_size,num_classes=num_classes)
    elif netD == 'resnet50':
        net = models.resnet50(pretrained=True)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, num_classes)
        if len(gpu_ids) > 0:
            assert (torch.cuda.is_available())
            net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
        return net
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)



class GANLoss(nn.Module):


    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):

        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'bce':
            self.loss = nn.BCELoss()
        elif gan_mode == 'ce':
            self.loss = nn.CrossEntropyLoss()
        elif gan_mode == 'kl':
            self.loss = nn.KLDivLoss()
        elif gan_mode == 'nll':
            self.loss = nn.NLLLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):


        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):

        if self.gan_mode in ['lsgan', 'vanilla', 'bce']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            if prediction.dtype == torch.float64:
                target_tensor = target_tensor.double()
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'ce':
            loss = self.loss(prediction, target_is_real)
        elif self.gan_mode == 'kl':
            loss = self.loss(prediction, target_is_real)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class SoftBCELoss(nn.Module):
    def __init__(self):
        super(SoftBCELoss, self).__init__()
        return

    def forward(self, inputs, target):

        loss = - target * inputs.log() - (1-target) * (1-inputs).log()
        loss = loss.mean()

        return loss


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def update_average(model_tgt, model_src, beta):
    toggle_grad(model_src, False)
    toggle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)

    toggle_grad(model_src, True)


class cDCGANGenerator(nn.Module):

    def __init__(self, input_nz, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, num_classes=10):

        super(cDCGANGenerator, self).__init__()

        self.deconv1_1 = nn.ConvTranspose2d(input_nz, ngf * 2, kernel_size=4, stride=1, padding=0, bias=False)
        self.deconv1_1_bn = norm_layer(ngf * 2)
        # class one-hot vector input
        self.deconv1_2 = nn.ConvTranspose2d(num_classes, ngf * 2, kernel_size=4, stride=1, padding=0, bias=False)
        self.deconv1_2_bn = norm_layer(ngf * 2)
        self.deconv2 = nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv2_bn = norm_layer(ngf * 2)
        self.deconv3 = nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3_bn = norm_layer(ngf)
        self.deconv4 = nn.ConvTranspose2d(ngf, output_nc, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, input, label):
        """Standard forward"""
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], dim=1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))
        return x


class cDCGANDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, num_classes=10):

        super(cDCGANDiscriminator, self).__init__()
        ndf = ndf + 1 if ndf % 2 == 1 else ndf
        self.conv1_1 = spectral_norm(nn.Conv2d(input_nc, ndf // 2, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv1_2 = spectral_norm(nn.Conv2d(num_classes, ndf // 2, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv3 = spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv4 = spectral_norm(nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=0, bias=False))

    def forward(self, input, label):
        """Standard forward."""
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = torch.sigmoid(self.conv4(x))
        return x


class ResNetBlock(nn.Module):
    def __init__(self, input_nc, output_nc, hidden_nc=None, bn=True, res_ratio=0.1):
        super().__init__()
        # Attributes
        self.bn = bn
        self.is_bias = not bn
        self.learned_shortcut = (input_nc != output_nc)
        self.input_nc = input_nc
        self.output_nc = output_nc
        if hidden_nc is None:
            self.hidden_nc = min(input_nc, output_nc)
        else:
            self.hidden_nc = hidden_nc
        self.res_ratio = res_ratio

        self.conv_0 = nn.Conv2d(self.input_nc, self.hidden_nc, 3, stride=1, padding=1, bias=self.is_bias)
        if self.bn:
            self.bn2d_0 = nn.BatchNorm2d(self.hidden_nc)
        self.conv_1 = nn.Conv2d(self.hidden_nc, self.output_nc, 3, stride=1, padding=1, bias=self.is_bias)
        if self.bn:
            self.bn2d_1 = nn.BatchNorm2d(self.output_nc)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.input_nc, self.output_nc, 1, stride=1, padding=0, bias=False)
            if self.bn:
                self.bn2d_s = nn.BatchNorm2d(self.output_nc)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(x)
        if self.bn:
            dx = self.bn2d_0(dx)
        dx = self.relu(dx)
        dx = self.conv_1(dx)
        if self.bn:
            dx = self.bn2d_1(dx)
        out = self.relu(x_s + self.res_ratio * dx)
        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
            if self.bn:
                x_s = self.bn2d_s(x_s)
        else:
            x_s = x
        return x_s


class cDCGANResnetGenerator(nn.Module):

    def __init__(self, input_nz, output_nc, nf=64, nf_max=512, img_size=32, num_classes=10, bn=True, res_ratio=0.1, **kwargs):
        super().__init__()
        self.nf = nf
        self.nf_max = nf_max
        s0 = self.s0 = 4
        self.bn = bn
        self.input_nz = input_nz

        nlayers = int(np.log2(img_size / s0))
        self.nf0 = min(nf_max, nf * 2**(nlayers+1))

        self.fc = nn.Linear(input_nz, self.nf0 * s0 * s0 // 2)
        self.fc_label = nn.Linear(num_classes, self.nf0 * s0 * s0 // 2)
        if self.bn:
            self.bn1d = nn.BatchNorm1d(self.nf0 * s0 * s0 // 2)
            self.bn1d_label = nn.BatchNorm1d(self.nf0 * s0 * s0 // 2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        blocks = []
        for i in range(nlayers, 0, -1):
            nf0 = min(nf * 2 ** (i + 1), nf_max)
            nf1 = min(nf * 2 ** i, nf_max)
            blocks += [
                ResNetBlock(nf0, nf1, bn=self.bn, res_ratio=res_ratio),
                nn.Upsample(scale_factor=2)
            ]

        nf0 = min(nf * 2, nf_max)
        nf1 = min(nf, nf_max)
        blocks += [
            ResNetBlock(nf0, nf1, bn=self.bn, res_ratio=res_ratio),
            ResNetBlock(nf1, nf1, bn=self.bn, res_ratio=res_ratio)
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, output_nc, 3, padding=1)

    def forward(self, *inputs):
        z = inputs[0]   # noise
        y = inputs[1]   # label
        batch_size = z.size(0)

        z = z.view(batch_size, -1)
        out_z = self.fc(z)
        if self.bn:
            out_z = self.bn1d(out_z)
        out_z = self.relu(out_z)
        out_z = out_z.view(batch_size, self.nf0 // 2, self.s0, self.s0)

        y = y.view(batch_size, -1)
        out_y = self.fc_label(y)
        if self.bn:
            out_y = self.bn1d_label(out_y)
        out_y = self.relu(out_y)
        out_y = out_y.view(batch_size, self.nf0 // 2, self.s0, self.s0)

        out = torch.cat([out_z, out_y], dim=1)

        out = self.resnet(out)
        out = self.conv_img(out)
        out = torch.tanh(out)

        return out


class cDCGANResnetDiscriminator(nn.Module):

    def __init__(self, input_nc, nf=64, nf_max=512, img_size=32, num_classes=10, res_ratio=0.1):
        super().__init__()
        s0 = self.s0 = 4
        self.nf = nf
        self.nf_max = nf_max

        nlayers = int(np.log2(img_size / s0))
        self.nf0 = min(nf_max, nf * 2 ** nlayers)

        nf0 = min(nf, nf_max)
        nf1 = min(nf * 2, nf_max)
        blocks = [
            ResNetBlock(nf0, nf0, bn=False, res_ratio=res_ratio),
            ResNetBlock(nf0, nf1, bn=False, res_ratio=res_ratio)
        ]

        for i in range(1, nlayers + 1):
            nf0 = min(nf * 2 ** i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResNetBlock(nf0, nf1, bn=False, res_ratio=res_ratio),
            ]

        self.conv_img = nn.Conv2d(input_nc, 1 * nf // 2, 3, padding=1)
        self.conv_label = nn.Conv2d(num_classes, 1 * nf // 2, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0 * s0 * s0, 1)

    def forward(self, *inputs):
        x, y = inputs[0], inputs[1]
        assert (x.size(0) == y.size(0))
        batch_size = x.size(0)

        out_img = self.relu(self.conv_img(x))
        out_label = self.relu(self.conv_label(y))
        out = torch.cat([out_img, out_label], dim=1)
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0 * self.s0 * self.s0)
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out


class DCGANResnetGenerator(nn.Module):

    def __init__(self, input_nz, output_nc, nf=64, nf_max=512, img_size=32, bn=True, res_ratio=0.1, num_classes=10, **kwargs):
        super().__init__()
        self.nf = nf
        self.nf_max = nf_max
        s0 = self.s0 = 4
        self.bn = bn
        self.input_nz = input_nz

        nlayers = int(np.log2(img_size / s0))
        self.nf0 = min(nf_max, nf * 2**(nlayers+1))

        self.fc = nn.Linear(input_nz, self.nf0 * s0 * s0)
        if self.bn:
            self.bn1d = nn.BatchNorm1d(self.nf0 * s0 * s0)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        blocks = []
        for i in range(nlayers, 0, -1):
            nf0 = min(nf * 2 ** (i + 1), nf_max)
            nf1 = min(nf * 2 ** i, nf_max)
            blocks += [
                ResNetBlock(nf0, nf1, bn=self.bn, res_ratio=res_ratio),
                nn.Upsample(scale_factor=2)
            ]

        nf0 = min(nf * 2, nf_max)
        nf1 = min(nf, nf_max)
        blocks += [
            ResNetBlock(nf0, nf1, bn=self.bn, res_ratio=res_ratio),
            ResNetBlock(nf1, nf1, bn=self.bn, res_ratio=res_ratio)
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, output_nc, 3, padding=1)

    def forward(self, z):
        batch_size = z.size(0)

        # for noise
        z = z.view(batch_size, -1)
        out_z = self.fc(z)
        if self.bn:
            out_z = self.bn1d(out_z)
        out_z = self.relu(out_z)
        out_z = out_z.view(batch_size, self.nf0, self.s0, self.s0)
        out = self.resnet(out_z)
        out = self.conv_img(out)
        out = torch.tanh(out)

        return out


class DCGANResnetDiscriminator(nn.Module):

    def __init__(self, input_nc, nf=64, nf_max=512, img_size=32, num_classes=10, res_ratio=0.1):
        super().__init__()
        s0 = self.s0 = 4
        self.nf = nf
        self.nf_max = nf_max

        # Submodules
        nlayers = int(np.log2(img_size / s0))
        self.nf0 = min(nf_max, nf * 2 ** nlayers)

        nf0 = min(nf, nf_max)
        nf1 = min(nf * 2, nf_max)
        blocks = [
            ResNetBlock(nf0, nf0, bn=False, res_ratio=res_ratio),
            ResNetBlock(nf0, nf1, bn=False, res_ratio=res_ratio)
        ]

        for i in range(1, nlayers + 1):
            nf0 = min(nf * 2 ** i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResNetBlock(nf0, nf1, bn=False, res_ratio=res_ratio),
            ]

        self.conv_img = nn.Conv2d(input_nc, 1 * nf, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0 * s0 * s0, 1)

    def forward(self, x):
        batch_size = x.size(0)

        out = self.relu(self.conv_img(x))
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0 * self.s0 * self.s0)
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out

class ACGANDiscriminator(nn.Module):
    def __init__(self,input_nc,img_size=32, num_classes=10):
        super(ACGANDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(input_nc, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        ds_size = img_size // 2 ** 4

        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, num_classes), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        return validity, label

class ACGANGenerator(nn.Module):
    def __init__(self,input_nc,output_nc,img_size=32, num_classes=10):
        super(ACGANGenerator, self).__init__()

        self.label_emb = nn.Embedding(num_classes, input_nc)

        self.init_size = img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(input_nc, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, output_nc, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


def computeConv(input_size, kernel_size=3, stride=1, padding=1):
    return (input_size - kernel_size + 2 * padding)//stride +1

def computeMaxPool2d(input_size, kernel_size=3, stride=1, padding=1, dilation =1):
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride +1

class Classifier(nn.Module):

    def __init__(self, input_nc,img_size=32, num_classes=10):
        super().__init__()
        self.image_size = img_size
        self.features = nn.Sequential(
            nn.Conv2d(input_nc, img_size, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(img_size, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

        )
        linear_input = computeMaxPool2d(computeConv(computeMaxPool2d(computeConv(img_size, 5, 1, 2), 3, 2, 1), 3, 1, 1),
                                        3, 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(64 * linear_input * linear_input, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )
        for layer in chain(self.features, self.classifier):
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x, start_layer_idx = 0):
        if start_layer_idx < 0:
            return self.classifier(x)
        x = self.features(x)
        out = x.view(x.size(0), -1)
        out = self.classifier(out)
        return out


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

class FedGENGenerator(nn.Module):
    def __init__(self, dataset='mnist', n_class=10, embedding=False, latent_layer_idx=-1, input_channel = 1,img_size = 32,gpu_ids = []):
        super(FedGENGenerator, self).__init__()
        print("Dataset {}".format(dataset))
        self.gpu_ids = gpu_ids
        self.embedding = embedding
        self.dataset = dataset
        self.latent_layer_idx = latent_layer_idx
        self.hidden_dim = 256
        line_dim =  computeMaxPool2d(computeConv(computeMaxPool2d(computeConv(img_size, 5, 1, 2), 3, 2, 1), 3, 1, 1),3, 2, 1)
        self.latent_dim = 64 * line_dim *line_dim
        self.noise_dim = 32
        self.input_channel = input_channel
        self.n_class = n_class
        input_dim = self.noise_dim * 2 if self.embedding else self.noise_dim + self.n_class
        self.fc_configs = [input_dim, self.hidden_dim]
        self.init_loss_fn()
        self.build_network()

    def get_number_of_parameters(self):
        pytorch_total_params=sum(p.numel() for p in self.parameters() if p.requires_grad)
        return pytorch_total_params

    def init_loss_fn(self):
        self.crossentropy_loss=nn.NLLLoss(reduce=False)
        self.diversity_loss = DiversityLoss(metric='l1')
        self.dist_loss = nn.MSELoss()

    def build_network(self):
        if self.embedding:
            self.embedding_layer = nn.Embedding(self.n_class, self.noise_dim)
        self.fc_layers = nn.ModuleList()
        for i in range(len(self.fc_configs) - 1):
            input_dim, out_dim = self.fc_configs[i], self.fc_configs[i + 1]
            print("Build layer {} X {}".format(input_dim, out_dim))
            fc = nn.Linear(input_dim, out_dim)
            bn = nn.BatchNorm1d(out_dim)
            act = nn.ReLU()
            self.fc_layers += [fc, bn, act]
        self.representation_layer = nn.Linear(self.fc_configs[-1], self.latent_dim)
        print("Build last layer {} X {}".format(self.fc_configs[-1], self.latent_dim))

    def forward(self, labels, latent_layer_idx=-1, verbose=True):

        result = {}
        batch_size = labels.shape[0] # 64
        eps = torch.rand((batch_size, self.noise_dim)).to(self.gpu_ids[0])
        if verbose:
            result['eps'] = eps
        if self.embedding: # embedded dense vector
            y_input = self.embedding_layer(labels).to(self.gpu_ids[0])
        else: # one-hot (sparse) vector
            y_input = torch.FloatTensor(batch_size, self.n_class).to(self.gpu_ids[0])
            y_input.zero_()
            #labels = labels.view
            labels_int64 = labels.type(torch.LongTensor).to(self.gpu_ids[0])
            y_input.scatter_(1, labels_int64.view(-1,1), 1)
        z = torch.cat((eps, y_input), dim=1)
        ### FC layers
        for layer in self.fc_layers:
            z = layer(z)
        z = self.representation_layer(z)
        result['output'] = z
        return result

    @staticmethod
    def normalize_images(layer):
        """
        Normalize images into zero-mean and unit-variance.
        """
        mean = layer.mean(dim=(2, 3), keepdim=True)
        std = layer.view((layer.size(0), layer.size(1), -1)) \
            .std(dim=2, keepdim=True).unsqueeze(3)
        return (layer - mean) / std
