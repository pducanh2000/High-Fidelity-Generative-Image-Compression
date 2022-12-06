import torch
import torch.nn as nn


class ChannelNorm2D(nn.Module):
    def __init__(self, input_channels, momentum=0.1, eps=1e-3, affine=True, **kwargs):
        """
        Similar to the implementation of InstanceNorm2D but calculate
        the moments over channel dimension instead of spatial dims
        """
        super(ChannelNorm2D, self).__init__()
        self.momentum = momentum
        self.kwargs = kwargs

        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, input_channels, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, input_channels, 1, 1))

    def forward(self, x):
        """
        :param x: the input tensor which should be in the format of (B, C, H, W)
        :return: tensor after normalizing with the shape of ()
        """
        mu, var = torch.mean(x, dim=1, keepdim=True), torch.var(x, dim=1, keepdim=True)
        x_normed = (x-mu) / torch.sqrt(var + self.eps)
        if self.affine:
            x_normed = x_normed * self.gamma + self.beta

        return x_normed


def channel_normalize(input_channels, momentum=0.1, eps=1e-3, affine=True, track_running_stats=False):
    channel_norm = ChannelNorm2D(
        input_channels,
        momentum=momentum,
        eps=eps,
        affine=affine,
        track_running_stats=track_running_stats,
    )
    return channel_norm


def instance_normalize(input_channels, momentum=0.1, affine=True, track_running_stats=False, **kwargs):
    instance_norm = nn.InstanceNorm2d(
        input_channels,
        momentum=momentum,
        affine=affine,
        track_running_stats=track_running_stats,
        **kwargs
    )
    return instance_norm
