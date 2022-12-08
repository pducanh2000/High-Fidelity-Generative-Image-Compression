import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_shape, context_shape, num_up_sample, spectral_norm=True):
        """
        Convolutional PatchGAN discriminator
        Accepts as input generator output G(z) or x ~ p*(x) where
        p*(x) is the true data distribution.
        Contextual information provided is encoder output y = E(x)

        :param input_shape: input image shape (C_in, H, W)
        :param context_shape: encoder output (C_in, H, W)
        :param num_up_sample: factor used to upscale the context to have same size as input shape
        :param spectral_norm:
        """
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        self.context_shape = context_shape
        self.in_channels = self.input_shape[0]

        self.context_out_channels = 12
        self.out_channels = (64, 128, 256, 512)

        # Upscale encoder output
        self.context_conv = nn.Conv2d(
            self.context_shape[0],
            self.context_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect"
        )
        self.context_up_sample = nn.Upsample(scale_factor=2**num_up_sample, mode="nearest")
        self.context_activation = nn.LeakyReLU(negative_slope=0.2)

        # Normalize
        if spectral_norm:
            norm = nn.utils.spectral_norm
        else:
            norm = nn.utils.weight_norm

        # Conv1
        self.conv1 = norm(nn.Conv2d(
            self.in_channels + self.context_out_channels,
            self.out_channels[0],
            kernel_size=4,
            stride=2,
            padding=1,
            padding_mode="reflect"
        ))
        self.activation1 = nn.LeakyReLU(negative_slope=0.2)

        # Conv2
        self.conv2 = norm(nn.Conv2d(
            self.out_channels[0],
            self.out_channels[1],
            kernel_size=4,
            stride=2,
            padding=1,
            padding_mode="reflect"
        ))
        self.activation2 = nn.LeakyReLU(negative_slope=0.2)

        # Conv3
        self.conv3 = norm(nn.Conv2d(
            self.out_channels[1],
            self.out_channels[2],
            kernel_size=4,
            stride=2,
            padding=1,
            padding_mode="reflect"
        ))
        self.activation3 = nn.LeakyReLU(negative_slope=0.2)

        # Conv4
        self.conv4 = norm(nn.Conv2d(
            self.out_channels[2],
            self.out_channels[3],
            kernel_size=4,
            stride=2,
            padding=1,
            padding_mode="reflect"
        ))
        self.activation4 = nn.LeakyReLU(negative_slope=0.2)

        # Conv out
        self.conv_out = nn.Conv2d(
            self.out_channels[3],
            out_channels=1,
            kernel_size=1,
            stride=1
        )

    def forward(self, gen_images, quantized_latent):
        """

        :param gen_images: generator output  default (B, 3, H, W)
        :param quantized_latent: Quantized latent code (B, context_shape[0], H/scale, W/scale)
        :return:
        """
        quantized_latent = self.context_activation(self.context_conv(quantized_latent))  # (B, 12, H/scale, W/scale)
        quantized_latent = self.context_up_sample(quantized_latent)  # (B, 12, H, W)

        cat_images = torch.cat((gen_images, quantized_latent), dim=1)  # (B, 15, H, W)
        out = self.activation1(self.conv1(cat_images))  # (B, 64, H/2, W/2)
        out = self.activation2(self.conv2(out))  # (B, 128, H/4, W/4)
        out = self.activation3(self.conv3(out))  # (B, 256, H/8, H/8)
        out = self.activation4(self.conv4(out))  # (B, 512, H/16, W/16)

        out_logits = self.conv_out(out).view(-1, 1)  # (B * H * W / 256, 1)
        out = torch.softmax(out_logits, dim=0)

        return out, out_logits


if __name__ == "__main__":
    from torchsummary import summary
    D = Discriminator(
        input_shape=(3, 256, 256),
        context_shape=(220, 16, 16),
        num_up_sample=4
    )
    summary(D, [(3, 256, 256), (220, 16, 16)], 32, "cpu")

    x = torch.randn((32, 3, 256, 256))  # gen_images
    y = torch.randn((32, 220, 16, 16))  # context_latent_code
    d_out, d_out_logits = D(x, y)
    print("Discriminator outputs: \n{} \n{}".format(
        d_out.size(),
        d_out_logits.size()
    ))

    # print(out_logits)
