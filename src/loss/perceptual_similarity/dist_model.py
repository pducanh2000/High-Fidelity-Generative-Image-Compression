from __future__ import absolute_import

import numpy as np
import torch
from torch.autograd import Variable

import os
from collections import OrderedDict
from scipy.ndimage import zoom
from src.loss.perceptual_similarity.base_model import BaseModel
from src.loss.perceptual_similarity.utils import tensor2im

from src.loss.perceptual_similarity import networks_basic as networks
from src.loss.perceptual_similarity.utils import print_network


class DistModel(BaseModel):
    def __init__(self):
        super(DistModel, self).__init__()
        # Default values for model
        self.model = None
        self.net = None
        self.is_train = None
        self.spatial = None
        self.network_base = None

        # Default values for optimizing
        self.rank_loss = None
        self.optim = None
        self.lr = None
        self.old_lr = None

    def initialize(self, model='net-lin', net='alex', colorspace='Lab', pnet_rand=False, pnet_tune=False,
                   model_path=None, use_gpu=True, printNet=False, spatial=False, is_train=False, lr=.0001, beta1=0.5,
                   gpu_ids=(0,)):
        BaseModel.initialize(self, use_gpu=True, gpu_ids=gpu_ids)

        self.model = model
        self.name = "%s_[%s]" % (self.model, self.net)
        self.net = net
        self.is_train = is_train
        self.spatial = spatial
        self.gpu_ids = gpu_ids

        # Initialize the model
        if self.model == "net-lin":
            # Pretrain net + Linear layer
            self.network_base = networks.PNetLin(
                pnet_rand=pnet_rand,
                pnet_tune=pnet_tune,
                pnet_type=net,
                use_dropout=True,
                spatial=spatial,
                lpips=True
            )

            if model_path is None:
                import inspect
                model_path = os.path.abspath(
                    os.path.join(inspect.getfile(self.initialize), "..", "weights/%s.pth" % self.net)
                )
            if not self.is_train:
                print("Loading model from model path: {}".format(model_path))
                if not use_gpu:
                    self.network_base.load_state_dict(torch.load(model_path, map_location="cpu"))
                else:
                    self.network_base.load_state_dict(torch.load(model_path))

        elif self.model == "net":
            # Pretrained net
            self.network_base = networks.PNetLin(pnet_rand=pnet_rand, pnet_tune=pnet_tune, pnet_type=net, lpips=False)
        elif self.model.upper() == "L2":
            # Not really a network use for testing only
            self.network_base = networks.L2(use_gpu=use_gpu, colorspace=colorspace)
        elif self.model in ['DSSIM', 'dssim', 'SSIM', 'ssim']:
            self.network_base = networks.DSSIM(use_gpu=self.use_gpu, colorspace=colorspace)
            self.name = "SSIM"

        else:
            raise ValueError("Model [{}] is not regconized".format(self.model))

        # Init the parameters
        if is_train:
            self.network_base.train()
            self.rank_loss = networks.BCERankingLoss(hidden_channel=32)
            self.lr = lr
            self.old_lr = lr
            self.optim = torch.optim.Adam(
                list(self.network_base.parameters()) + list(self.rank_loss.parameters()),
                lr=self.lr,
                betas=(beta1, 0.999)
            )
        else:
            self.network_base.eval()

        if self.use_gpu:
            self.network_base.to(gpu_ids[0])
            self.network_base = torch.nn.DataParallel(self.network_base, device_ids=gpu_ids)
            if self.is_train:
                self.rank_loss = self.rank_loss.to(device=gpu_ids[0])  # just put this on GPU0

        # Verbose print the network_base
        if printNet:
            print("-" * 15 + "Network Initializing" + "-" * 15)
            print_network(self.network_base)
            print("-" * 45)

    def forward(self, in0, in1, return_per_layer=False):
        return self.network_base.forward(in0, in1, return_per_layer=return_per_layer)

    # For optimizing, training functions inside
    # I think the functions below is redundant if im not wrong i will delete them

    # Get the training data
    def set_input(self, data):
        # This is a redundant function, im confused cuz we need to train the BCE RankingLoss
        self.input_ref = data['ref']
        self.input_p0 = data['p0']
        self.input_p1 = data['p1']
        self.input_judge = data['judge']

        if self.use_gpu:
            self.input_ref = self.input_ref.to(device=self.gpu_ids[0])
            self.input_p0 = self.input_p0.to(device=self.gpu_ids[0])
            self.input_p1 = self.input_p1.to(device=self.gpu_ids[0])
            self.input_judge = self.input_judge.to(device=self.gpu_ids[0])

        # Make the variables trainable
        self.var_ref = Variable(self.input_ref, requires_grad=True)
        self.var_p0 = Variable(self.input_p0, requires_grad=True)
        self.var_p1 = Variable(self.input_p1, requires_grad=True)

    def forward_training(self):
        # Another redundant function i guess
        self.d0 = self.forward(self.var_ref, self.var_p0)
        self.d1 = self.forward(self.var_ref, self.var_p1)
        self.acc_r = self.compute_accuracy(self.d0, self.d1, judge=self.input_judge)

        self.var_judge = Variable(1. * self.input_judge).view(self.d0.size())

        self.loss_total = self.rank_loss.forward(self.d0, self.d1, self.var_judge * 2. - 1.)

        return self.loss_total

    def backward_train(self):
        torch.mean(self.loss_total).backward()

    @staticmethod
    def compute_accuracy(d0, d1, judge):
        """ d0, d1 are Variables, judge is a Tensor """
        d1_lt_d0 = (d1 < d0).cpu().data.numpy().flatten()
        judge_per = judge.cpu().numpy().flatten()
        return d1_lt_d0 * judge_per + (1 - d1_lt_d0) * (1 - judge_per)

    def get_current_errors(self):
        retDict = OrderedDict([('loss_total', self.loss_total.data.cpu().numpy()),
                               ('acc_r', self.acc_r)])

        for key in retDict.keys():
            retDict[key] = np.mean(retDict[key])

        return retDict

    def get_current_visuals(self):
        zoom_factor = 256 / self.var_ref.data.size()[2]

        ref_img = tensor2im(self.var_ref.data)
        p0_img = tensor2im(self.var_p0.data)
        p1_img = tensor2im(self.var_p1.data)

        ref_img_vis = zoom(ref_img, [zoom_factor, zoom_factor, 1], order=0)
        p0_img_vis = zoom(p0_img, [zoom_factor, zoom_factor, 1], order=0)
        p1_img_vis = zoom(p1_img, [zoom_factor, zoom_factor, 1], order=0)

        return OrderedDict([('ref', ref_img_vis),
                            ('p0', p0_img_vis),
                            ('p1', p1_img_vis)])

    def save(self, path, label):
        if self.use_gpu:
            self.save_network(self.net.module, path, '', label)
        else:
            self.save_network(self.net, path, '', label)
        self.save_network(self.rank_loss.net, path, 'rank', label)

    def update_learning_rate(self, nepoch_decay):
        lrd = self.lr / nepoch_decay
        lr = self.old_lr - lrd

        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

        print('update lr [%s] decay: %f -> %f' % (type, self.old_lr, lr))
        self.old_lr = lr
