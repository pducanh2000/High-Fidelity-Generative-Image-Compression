import torch.nn as nn

from src.models.hific.normaliztion import channel_normalize, instance_normalize


class Encoder(nn.Module):
    def __init__(self,
                 image_shape,
                 num_down=4,
                 out_channels_based=60,
                 out_channels_bottleneck=220,
                 activation="relu",
                 channels_norm=True):
        """
        Encoder with convolution layers to convert the input image with the shape of (B, C, H, W)
        to a feature tensor with the shape of (B, C, H/16, W/16)
        """
        super(Encoder, self).__init__()
        # Get input_channels
        self.in_channels = image_shape[0]  # image_shape in the format of (C, H, W)
        # out_channels for CNN layers, with base=60 [60, 120, 240, 480, 960]
        self.out_channels = [out_channels_based * 2 ** i for i in range(num_down + 1)]
        self.out_channels_bottleneck = out_channels_bottleneck

        # Init necessary layers

        # Normalization
        if channels_norm:
            self.interlayer_norm = channel_normalize
        else:
            self.interlayer_norm = instance_normalize
        # Activations
        activation_dict = dict(relu='ReLU', elu='ELU', leaky_relu='Leaky_ReLU')
        self.activation_layer = getattr(nn, activation_dict[activation])
        # Padding-original tf repos padding zeros but in the torch version by Justin  pad reflection
        self.pre_pad = nn.ReflectionPad2d(3)
        self.asymmetric_pad = nn.ReflectionPad2d((0, 1, 1, 0))
        self.post_pad = nn.ReflectionPad2d(1)

        # Sequence of layers first_layer + num_down * down_sample_layers + last_layer
        layers = []
        for i in range(0, num_down + 2):
            if i == 0:
                # First layer does not down sample
                layers.append(nn.Sequential(
                    self.pre_pad,
                    nn.Conv2d(self.in_channels, self.out_channels[i], kernel_size=7, stride=1),
                    self.interlayer_norm(input_channels=self.out_channels[i]),
                    self.activation_layer()
                ))
            elif i == num_down+1:
                # Last layer does not down sample
                layers.append(nn.Sequential(
                    self.post_pad,
                    nn.Conv2d(self.out_channels[i-1], self.out_channels_bottleneck, kernel_size=3, stride=1)
                ))
            else:
                # Down-sampling layers (default 4 layers)
                layers.append(nn.Sequential(
                    self.asymmetric_pad,
                    nn.Conv2d(
                        self.out_channels[i-1],
                        self.out_channels[i],
                        kernel_size=3,
                        stride=2,
                        padding=0,
                        # padding_mode="reflect"
                    ),
                    self.interlayer_norm(input_channels=self.out_channels[i]),
                    self.activation_layer()
                ))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Extract feature map from input image by down-sampling through CNN layers
        :param x: input image with the shape of (B, C, H, W)
        :return: feature map with the shape of (B, C_bottleneck, H/16, W/16)
        """
        out = self.model(x)
        return out


if __name__ == "__main__":
    from torchsummary import summary
    model = Encoder((1, 3, 256, 256))
    summary(model, (1, 3, 256, 256))