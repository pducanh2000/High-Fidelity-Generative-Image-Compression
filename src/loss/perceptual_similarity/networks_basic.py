from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable

from src.loss.perceptual_similarity import pretrained_networks as pretrained
from src.loss.perceptual_similarity.utils import normalize_tensor, spatial_average, upsample, tensor2tensorlab, \
    tensor2np, l2, tensor2im, dssim


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(), ] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Learned perceptual metric
class PNetLin(nn.Module):
    def __init__(self, pnet_type='vgg', pnet_rand=False, pnet_tune=False, use_dropout=True, spatial=False, lpips=True):
        super(PNetLin, self).__init__()
        self.pnet_type = pnet_type
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lipips = lpips
        self.scale_layer = ScalingLayer()

        net_type = None
        if self.pnet_type in ['vgg', 'vgg16']:
            net_type = pretrained.VGG16
            self.channels = [64, 128, 256, 512, 512]
        elif self.pnet_type == 'alex':
            net_type = pretrained.AlexNet
            self.channels = [64, 192, 384, 256, 256]
        elif self.pnet_type == 'squeeze':
            net_type = pretrained.SqueezeNet
            self.channels = [64, 128, 256, 384, 384, 512, 512]
        self.L = len(self.channels)

        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        if self.lpips:
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            if self.pnet_type == 'squeeze':
                # 7 layers for squeezenet
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins += [self.lin5, self.lin6]

    def forward(self, in0, in1, return_per_layer=False):

        input0, input1 = (self.scale_layer(in0), self.scale_layer(in1))  # (B, C, H, W), (B, C, H, W)
        output0, output1 = self.net(input0), self.net(input1)  # (B, self.channels[i], H', W')
        features_0, features_1, diffs = {}, {}, {}

        for i in range(self.L):
            features_0[i], features_1[i] = normalize_tensor(output0[i]), normalize_tensor(output1[i])
            diffs[i] = (features_0[i] - features_1[i]) ** 2  # list of tensors, size: (B, self.channels[i], H', W')

        # res is a list of feature maps
        if self.lpips:
            if self.spatial:
                # list of tensors shape (B, 1, in0.shape[2], in0.shape[3])
                res = [upsample(self.lins[kk].model(diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                # list of tensors shape (B, 1, 1, 1)
                res = [spatial_average(self.lins[kk].model(diffs[kk]), keepdim=True) for kk in range(self.L)]
        else:
            if self.spatial:
                # list of tensors shape (B, 1, in0.shape[2], in0.shape[3])
                res = [upsample(diffs[kk].sum(dim=1, keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                # list of tensors shape (B, 1, 1, 1)
                res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]

        # Sum up all the results
        val = res[0]
        for i in range(1, self.L):
            val += res[i]

        if return_per_layer:
            return val, res
        else:
            return val


# BCERankingLoss
class Dist2LogitLayer(nn.Module):
    """ takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) """

    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()

        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True), ]
        layers += [nn.LeakyReLU(0.2, True), ]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True), ]
        layers += [nn.LeakyReLU(0.2, True), ]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True), ]
        if use_sigmoid:
            layers += [nn.Sigmoid(), ]
        self.model = nn.Sequential(*layers)

    def forward(self, d0, d1, eps=0.1):
        return self.model.forward(torch.cat((d0, d1, d0 - d1, d0 / (d1 + eps), d1 / (d0 + eps)), dim=1))


class BCERankingLoss(nn.Module):
    def __init__(self, hidden_channel=32):
        super(BCERankingLoss, self).__init__()
        self.hidden_channel = hidden_channel
        self.net = Dist2LogitLayer(chn_mid=self.hidden_channel)
        self.loss = nn.BCELoss()

    def forward(self, d0, d1, judge):
        per = (judge + 1.) / 2.
        logit = self.net.forward(d0, d1)

        return self.loss(logit, per)


# L2, DSSIM metrics
class FakeNet(nn.Module):
    def __init__(self, use_gpu=True, colorspace='Lab'):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace = colorspace


class L2(FakeNet):
    def forward(self, in0, in1, return_per_layer=False):
        assert (in0.size()[0] == 1)  # currently only supports batchSize 1

        if self.colorspace == 'RGB':
            (N, C, X, Y) = in0.size()
            value = torch.mean(torch.mean(torch.mean((in0 - in1) ** 2, dim=1).view(N, 1, X, Y), dim=2).view(N, 1, 1, Y),
                               dim=3).view(N)
            return value
        elif self.colorspace == 'Lab':
            value = l2(tensor2np(tensor2tensorlab(in0.data, to_norm=False)),
                       tensor2np(tensor2tensorlab(in1.data, to_norm=False)), data_range=100.).astype('float')
            ret_var = Variable(torch.Tensor((value,)))
            if self.use_gpu:
                ret_var = ret_var.cuda()

            return ret_var


class DSSIM(FakeNet):
    def forward(self, in0, in1, return_per_layer=False):
        # Only support with batch size = 1
        assert in0.size()[0] == 1

        # Init the return values
        value = None

        # Calculate the return values
        if self.colorspace == "RGB":
            value = dssim(1. * tensor2im(in0), 1.0 * tensor2im(in1), data_range=255.0).astype("float")
        elif self.colorspace == "Lab":
            value = dssim(
                tensor2np(tensor2tensorlab(in0.data, to_norm=False)),
                tensor2np(tensor2tensorlab(in1.data, to_norm=False)),
                data_range=100
            ).astype("Float")

        ret_var = Variable(torch.Tensor((value,)))
        if self.use_gpu:
            ret_var = ret_var.cuda()
        return ret_var
