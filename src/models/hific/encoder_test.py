import torch.nn as nn

from src.models.hific.normaliztion import channel_normalize, instance_normalize


class Encoder(nn.Module):
    def __init__(self,
                 image_shape,
                 num_down=4,
                 out_channels_base=60,
                 out_channels_bottleneck=220,
                 activation="relu",
                 channels_norm=True):
        """
        Encoder with convolution layers to convert the input image with the shape of (B, C, H, W)
        to a feature tensor with the shape of (B, C, H/16, W/16)
        """
        super(Encoder, self).__init__()
        # Init params
        self.in_channels = image_shape[0]  # image_shape in the format of (C, H, W)
        self.num_down = num_down
        # out_channels for CNN layers, with base=60 [60, 120, 240, 480, 960]
        self.out_channels = [out_channels_base * 2 ** i for i in range(num_down + 1)]
        self.out_channels_bottleneck = out_channels_bottleneck

        # Init necessary layers
        # Normalization
        if channels_norm:
            interlayer_norm = channel_normalize
        else:
            interlayer_norm = instance_normalize
        # Activations
        activation_dict = dict(relu='ReLU', elu='ELU', leaky_relu='Leaky_ReLU')
        activation_layer = getattr(nn, activation_dict[activation])
        # Padding-original tf repos padding zeros but in the torch version by Justin  pad reflection
        pre_pad = nn.ReflectionPad2d(3)
        # self.asymmetric_pad = nn.ReflectionPad2d((0, 1, 1, 0))
        asymmetric_pad = nn.ReflectionPad2d(1)
        post_pad = nn.ReflectionPad2d(1)

        # Sequence of layers first_layer + num_down * down_sample_layers + last_layer

        for i in range(0, num_down + 2):
            if i == 0:
                # First layer does not down sample
                self.add_module(f'conv_block{str(i)}', nn.Sequential(
                    pre_pad,
                    nn.Conv2d(self.in_channels, self.out_channels[i], kernel_size=7, stride=1),
                    interlayer_norm(input_channels=self.out_channels[i]),
                    activation_layer()
                ))
            elif i == num_down+1:
                # Last layer does not down sample
                self.add_module(f'conv_block{str(i)}', nn.Sequential(
                    post_pad,
                    nn.Conv2d(self.out_channels[i-1], self.out_channels_bottleneck, kernel_size=3, stride=1)
                ))
            else:
                # Down-sampling layers (default 4 layers)
                self.add_module(f'conv_block{str(i)}', nn.Sequential(
                    asymmetric_pad,
                    nn.Conv2d(
                        self.out_channels[i-1],
                        self.out_channels[i],
                        kernel_size=3,
                        stride=2,
                        padding=0,
                        padding_mode="reflect"
                    ),
                    interlayer_norm(input_channels=self.out_channels[i]),
                    activation_layer()
                ))

    def forward(self, x):
        """
        Extract feature map from input image by down-sampling through CNN layers
        :param x: input image with the shape of (B, C, H, W)
        :return: feature map with the shape of (B, C_bottleneck, H/16, W/16)
        """
        for i in range(self.num_down+2):
            conv_block = getattr(self, f'conv_block{str(i)}')
            x = conv_block(x)
        return x


if __name__ == "__main__":
    from torchsummary import summary
    model = Encoder((3, 768, 768), num_down=4)
    summary(model, (3, 768, 768), batch_size=1, device="cpu")
