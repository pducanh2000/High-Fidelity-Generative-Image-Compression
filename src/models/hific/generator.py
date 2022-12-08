import torch
import torch.nn as nn
import torch.nn.functional as f

from src.models.hific.normaliztion import channel_normalize, instance_normalize


class ResidualBlock(nn.Module):
    def __init__(self, input_shape, kernel_size=3, stride=1, channel_norm=True, activation="relu"):
        """
        Residual connection in the Generator module. keep the shape of input feature unchanged
        :param input_shape: shape of the input feature in the format of (C, H, W)
        :param channel_norm: Use channel_normalize or not
        :param activation: type of the activate function will be used ["relu", "elu", "leaky_relu"]
        """
        super(ResidualBlock, self).__init__()
        # Init params
        self.in_channels = input_shape[0]
        self.kernel_size = kernel_size
        self.stride = stride

        # Init layers
        self.activation_layer = getattr(f, activation)
        if channel_norm:
            interlayer_norm = channel_normalize
        else:
            interlayer_norm = instance_normalize

        self.pad = nn.ReflectionPad2d(int((self.kernel_size-1)/2))
        self.conv1 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=self.kernel_size, stride=self.stride)
        self.norm1 = interlayer_norm(input_channels=self.in_channels)

        self.conv2 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=self.kernel_size, stride=self.stride)
        self.norm2 = interlayer_norm(input_channels=self.in_channels)

    def forward(self, x):
        """
        Pass input feature through 2 convolution blocks which keep the channel unchanged
        :param x: input feature
        :return:
        """
        out = self.pad(x)
        out = self.norm1(self.conv1(out))
        out = self.activation_layer(out)

        out = self.pad(out)
        out = self.norm2(self.conv2(out))

        return torch.add(x, out)


class Generator(nn.Module):
    def __init__(self,
                 quantized_shape,
                 num_up=4,
                 out_channels_base=60,
                 activation="relu",
                 n_residual_blocks=8,
                 channel_norm=True,
                 sample_noise=False,
                 noise_dim=32
                 ):
        """
        Generator up scales quantized encoder output into feature map of size (C, H, W)
        In default case: the shape of quantized encoder output aka the quantized_shape is (220, 16, 16)
        """
        super(Generator, self).__init__()
        # Init parameters
        self.in_channels = quantized_shape[0]    # Cuz the quantized_shape in the format of (C, H, W)
        self.out_channels = list(reversed([out_channels_base * 2**i for i in range(0, num_up+1)]))
        self.sample_noise = sample_noise
        self.noise_dim = noise_dim
        self.num_up_sample = num_up
        self.n_residual_blocks = n_residual_blocks

        # Init layers
        activation_dict = dict(relu='ReLU', elu='ELU', leaky_relu='Leaky_ReLU')
        self.activation_layer = getattr(nn, activation_dict[activation])

        if channel_norm:
            self.interlayer_norm = channel_normalize
        else:
            self.interlayer_norm = instance_normalize

        pre_pad = nn.ReflectionPad2d(1)
        post_pad = nn.ReflectionPad2d(3)

        # First conv block does not up sample feature
        self.first_conv_block = nn.Sequential(
            self.interlayer_norm(input_channels=self.in_channels),
            pre_pad,
            nn.Conv2d(self.in_channels, self.out_channels[0], kernel_size=3, stride=1),
            self.interlayer_norm(input_channels=self.out_channels[0])
        )
        # Concat noise with latent representation
        if self.sample_noise:
            self.out_channels[0] += self.noise_dim

        # Add residual blocks
        for m in range(n_residual_blocks):
            self.add_module(f'residual_block{str(m)}',
                            ResidualBlock(input_shape=(self.out_channels[0], quantized_shape[1], quantized_shape[2]),
                                          kernel_size=3,
                                          stride=1,
                                          channel_norm=True,
                                          activation="relu"
                                          )
                            )
        for i in range(self.num_up_sample):
            self.add_module(f'up_sample{str(i)}',
                            nn.Sequential(
                                nn.ConvTranspose2d(
                                    self.out_channels[i],
                                    self.out_channels[i+1],
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    output_padding=1
                                ),
                                self.interlayer_norm(self.out_channels[i+1]),
                                self.activation_layer()
                            )
                            )
        self.last_conv_block = nn.Sequential(
            post_pad,
            nn.Conv2d(self.out_channels[-1], 3, kernel_size=7, stride=1)
        )

    def forward(self, x):
        """
        :param x: input feature of the shape (B, C, H, W)
        :return:
        """
        # Pass through first convolution block
        out_first_conv = self.first_conv_block(x)

        # Add noise
        if self.sample_noise:
            b, c, h, w = tuple(out_first_conv.size())   # C = self.out_channels[0]
            noise = torch.randn((b, self.noise_dim, h, w))  # Generate a noise to concat with the latent code
            out_first_conv = torch.cat((out_first_conv, noise), dim=1)  # Concat in channel dimension

        # Pass through residual blocks
        out = None
        for m in range(self.n_residual_blocks):
            residual_block = getattr(self, f'residual_block{str(m)}')
            if m == 0:
                out = residual_block(out_first_conv)
            else:
                out = residual_block(out)

        # Add a residual connection before up sampling
        out = torch.add(out, out_first_conv)

        # Pass through up sample convolution blocks
        for i in range(self.num_up_sample):
            up_sample_conv = getattr(self, f'up_sample{str(i)}')
            out = up_sample_conv(out)

        # Pass through last convolution block
        out = self.last_conv_block(out)

        return out


if __name__ == "__main__":
    from torchsummary import summary
    model = Generator((220, 16, 16), 4, 60, "relu", 8)
    summary(model, (220, 16, 16), 32, "cpu")
