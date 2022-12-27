import os
import time
import numpy as np
from collections import namedtuple, defaultdict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as f

# Custom modules
from src.models.hific import hyperprior
from src.loss import losses
from src.helpers import math_material, utils
from src.dataset import dataload
from src.models.hific import encoder, generator, discriminator, hyper
from src.loss.perceptual_similarity import perceptual_loss as ps

from config import ModelModes, ModelTypes, hific_args, directories

Intermediates = namedtuple(
    "Intermediates",
    ["input_image",  # [0, 1] (after scaling from [0, 255])
     "reconstruction",  # [0, 1]
     "latents_quantized",  # Latents post-quantization.
     "n_bpp",  # Differential entropy estimate.
     "q_bpp"]  # Shannon entropy estimate.
)

Disc_out = namedtuple(
    "disc_out",
    ["D_real", "D_gen", "D_real_logits", "D_gen_logits"]
)


class HIFICModel(nn.Module):
    def __init__(self,
                 args,
                 logger,
                 storage_train=defaultdict(list),
                 storage_test=defaultdict(list),
                 model_mode=ModelModes.TRAINING,
                 model_type=ModelTypes.COMPRESSION
                 ):
        super(HIFICModel, self).__init__()
        # Build model from submodules
        self.args = args
        self.model_mode = model_mode
        self.model_type = model_type

        # For logging
        self.logger = logger
        self.log_interval = args.log_interval

        # For storing
        self.storage_train = storage_train
        self.storage_test = storage_test
        self.step_counter = 0

        if self.args.use_latent_mixture_model is True:
            self.args.latent_channels = self.args.latent_channels_DLMM

        if not hasattr(ModelTypes, self.model_type.upper()):
            raise ValueError("Invalid model_type: [{}]".format(self.model_type))
        if not hasattr(ModelModes, self.model_mode.upper()):
            raise ValueError("Invalid model_mode: [{}]".format(self.model_mode))

        self.image_dims = self.args.image_dims  # (C, H, W)
        self.batch_size = self.args.batch_size  # B

        self.entropy_code = False
        if model_mode == ModelModes.EVALUATION:
            self.entropy_code = True

        self.Encoder = encoder.Encoder(image_shape=self.image_dims,
                                       num_down=self.args.num_down,
                                       out_channels_bottleneck=self.args.latent_channels,
                                       channels_norm=True)

        quantized_shape = (self.args.latent_channels,
                           self.image_dims[1] // 2 ** self.args.num_down,
                           self.image_dims[2] // 2 ** self.args.num_down
                           )
        self.Generator = generator.Generator(quantized_shape=quantized_shape,
                                             num_up=self.args.num_down,
                                             n_residual_blocks=self.args.n_residual_blocks,
                                             channel_norm=self.args.use_channel_norm,
                                             sample_noise=self.args.sample_noise,
                                             noise_dim=self.args.noise_dim
                                             )
        if self.args.use_latent_mixture_model is True:
            self.Hyperprior = hyperprior.HyperpriorDLMM(bottleneck_capacity=self.args.latent_channels,
                                                        likelihood_type=self.args.likelihood_type,
                                                        mixture_components=self.args.mixture_components,
                                                        entropy_code=self.entropy_code)
        else:
            self.Hyperprior = hyperprior.Hyperprior(bottleneck_capacity=self.args.latent_channels,
                                                    likelihood_type=self.args.likelihood_type,
                                                    entropy_code=self.entropy_code)

        # Amortization models
        self.amortization_models = [self.Encoder, self.Generator]
        self.amortization_models.extend(self.Hyperprior.amortization_model)

        # Use discriminator if GAN mode enabled and in training/validation
        self.use_discriminator = (
                self.model_type == ModelTypes.COMPRESSION_GAN
                and (self.model_mode != ModelModes.EVALUATION)
        )

        if self.use_discriminator is True:
            assert self.args.discriminator_steps > 0, 'Must specify nonzero training steps for D!'
            self.discriminator_steps = self.args.discriminator_steps
            self.logger.info('GAN mode enabled. Training discriminator for {} steps.'.format(
                self.discriminator_steps))
            self.Discriminator = discriminator.Discriminator(
                input_shape=self.image_dims,
                context_shape=quantized_shape,
                num_up_sample=self.args.num_down,
                spectral_norm=True
            )
            self.gan_loss = partial(losses.gan_loss, args.gan_loss_type)
        else:
            self.discriminator_steps = 0
            self.Discriminator = None

        # Define losses
        self.squared_difference = torch.nn.MSELoss(reduction='none')
        # Expects [-1,1] images or [0,1] with normalize=True flag
        self.perceptual_loss = ps.PerceptualLoss(model='net-lin',
                                                 net='alex',
                                                 use_gpu=torch.cuda.is_available(),
                                                 gpu_ids=[args.gpu])

    def store_loss(self, key, loss):
        assert type(loss) == float, 'Call .item() on loss before storage'

        if self.training is True:
            storage = self.storage_train
        else:
            storage = self.storage_test

        if self.writeout is True:
            storage[key].append(loss)

    def compression_forward(self, x):
        image_dims = tuple(x.size()[1:])  # (C, H, W)
        if self.model_mode == ModelModes.EVALUATION and self.training is False:
            n_encoder_downsamples = self.args.num_down
            factor = 2 ** n_encoder_downsamples
            x = utils.pad_factor(input_image=x, spatial_dims=x.size()[2:], factor=factor)

        # Pass through the Encoder
        y = self.Encoder(x)  # (B, bottleneck_channels, H // factor, W // factor)

        if self.model_mode == ModelModes.EVALUATION and (self.training is False):
            n_hyperencoder_downsamples = self.Hyperprior.analysis_net.n_downsampling_layers
            factor = 2 ** n_hyperencoder_downsamples
            y = utils.pad_factor(y, y.size()[2:], factor)

        # Pass through the Hyperprior encoder
        hyperinfo = self.Hyperprior(y, spatial_shape=y.size()[2:])
        latents_quantized = hyperinfo.decoded
        total_nbpp = hyperinfo.total_nbpp
        total_qbpp = hyperinfo.total_qbpp

        # Use quantized latents as input to G
        reconstruction = self.Generator(latents_quantized)

        # If we use the image_input normalization
        if self.args.normalize_input_image is True:
            reconstruction = torch.tanh(reconstruction)

        # Undo padding
        if self.model_mode == ModelModes.EVALUATION and (self.training is False):
            reconstruction = reconstruction[:, :, :image_dims[1], :image_dims[2]]

        intermediates = Intermediates(x, reconstruction, latents_quantized,
                                      total_nbpp, total_qbpp)

        return intermediates, hyperinfo

    def discriminator_forward(self, intermediates: Intermediates, train_generator):
        """ Train on gen/real batches simultaneously. """
        x_gen = intermediates.reconstruction
        x_real = intermediates.input_image

        # Alternate between the compression models and training discriminator
        if train_generator is False:
            x_gen = x_gen.detach()

        D_in = torch.cat([x_real, x_gen], dim=0)
        latents = intermediates.latents_quantized.detach()
        latents = torch.repeat_interleave(latents, 2, dim=0)

        D_out, D_out_logits = self.Discriminator(D_in, latents)
        D_out, D_out_logits = torch.squeeze(D_out), torch.squeeze(D_out_logits)

        D_real, D_gen = torch.chunk(D_out, 2, dim=0)
        D_real_logits, D_gen_logits = torch.chunk(D_out_logits, 2, dim=0)

        return Disc_out(D_real, D_gen, D_real_logits, D_gen_logits)

    def distortion_loss(self, x_gen, x_real):
        # loss in [0,255] space but normalized by 255 to not be too big
        # - Delegate scaling to weighting
        sq_err = self.squared_difference(x_gen * 255., x_real * 255.)  # / 255.
        return torch.mean(sq_err)

    def perceptual_loss_wrapper(self, x_gen, x_real, normalize=True):
        """ Assumes inputs are in [0, 1] if normalize=True, else [-1, 1] """
        LPIPS_loss = self.perceptual_loss.forward(x_gen, x_real, normalize=normalize)
        return torch.mean(LPIPS_loss)

    def compression_loss(self, intermediates, hyperinfo):
        x_real = intermediates.input_image
        x_gen = intermediates.reconstruction

        if self.args.normalize_input_image:
            # Convert the range of value from [-1, 1] to [0, 1]
            x_real = (x_real + 1.0) / 2
            x_gen = (x_gen + 1.0) / 2

        distortion_loss = self.distortion_loss(x_gen=x_gen, x_real=x_real)
        perceptual_loss = self.perceptual_loss(pred=x_gen, target=x_real, normalize=True)

        weighted_distortion = self.args.k_M * distortion_loss
        weighted_perceptual = self.args.k_P * perceptual_loss

        # Adaptive rate penalty
        weighted_rate, rate_penalty = losses.weighted_rate_loss(self.args, total_nbpp=intermediates.n_bpp,
                                                                total_qbpp=intermediates.q_bpp,
                                                                step_counter=self.step_counter,
                                                                ignore_schedule=self.args.ignore_schedule)

        weighted_R_D_loss = weighted_rate + weighted_distortion
        weighted_compression_loss = weighted_R_D_loss + weighted_perceptual

        # Bookkeeping
        if self.step_counter % self.log_interval == 1:
            self.store_loss('rate_penalty', rate_penalty)
            self.store_loss('distortion', distortion_loss.item())
            self.store_loss('perceptual', perceptual_loss.item())
            self.store_loss('n_rate', intermediates.n_bpp.item())
            self.store_loss('q_rate', intermediates.q_bpp.item())
            self.store_loss('n_rate_latent', hyperinfo.latent_nbpp.item())
            self.store_loss('q_rate_latent', hyperinfo.latent_qbpp.item())
            self.store_loss('n_rate_hyperlatent', hyperinfo.hyperlatent_nbpp.item())
            self.store_loss('q_rate_hyperlatent', hyperinfo.hyperlatent_qbpp.item())

            self.store_loss('weighted_rate', weighted_rate.item())
            self.store_loss('weighted_distortion', weighted_distortion.item())
            self.store_loss('weighted_perceptual', weighted_perceptual.item())
            self.store_loss('weighted_R_D', weighted_R_D_loss.item())
            self.store_loss('weighted_compression_loss_sans_G', weighted_compression_loss.item())

        return weighted_compression_loss

    def GAN_loss(self, intermediates, train_generator=False):
        """
        :param intermediates: output of compression forward, include input_image,
        reconstruction, latents_qunatized and estimation bits
        :param train_generator: flag to send gradients to generator
        :return:
        """

        # Disc out includes D_real, D_gen, D_real_logits, D_gen_logits
        disc_out = self.discriminator_forward(intermediates=intermediates, train_generator=train_generator)

        D_loss = self.gan_loss(disc_out, mode='discriminator_loss')
        G_loss = self.gan_loss(disc_out, mode='generator_loss')

        # Bookkeeping
        if self.step_counter % self.log_interval == 1:
            self.store_loss('D_gen', torch.mean(disc_out.D_gen).item())
            self.store_loss('D_real', torch.mean(disc_out.D_real).item())
            self.store_loss('disc_loss', D_loss.item())
            self.store_loss('gen_loss', G_loss.item())
            self.store_loss('weighted_gen_loss', (self.args.beta * G_loss).item())

        return D_loss, G_loss

    def compress(self, x, silent=False):
        """
            * Pass image through encoder to obtain latents: x -> Encoder() -> y
            * Pass latents through hyperprior encoder to obtain hyperlatents:
            y -> hyperencoder() -> z
            * Encode hyperlatents via nonparametric entropy model.
            * Pass hyperlatents through mean-scale hyperprior decoder to obtain mean,
            scale over latents: z -> hyperdecoder() -> (mu, sigma).
            * Encode latents via entropy model derived from (mean, scale) hyperprior output.
        """
        assert self.model_mode.ModelModes.EVALUATION and self.training is False, (
            f'Set model mode to {ModelModes.EVALUATION} to be able to compress'
        )
        spatial_shape = x.size()[2:]

        if self.model_mode == ModelModes.EVALUATION and self.training is False:
            n_encoder_downsamples = self.args.num_down
            factor = 2 ** n_encoder_downsamples
            x = utils.pad_factor(input_image=x, spatial_dims=spatial_shape, factor=factor)

        # Pass the input through the Encoder
        y = self.Encoder(x)

        if self.model_mode == ModelModes.EVALUATION and self.training is not False:
            n_hyperencoder_downsamples = self.Hyperprior.analysis_net.n_downsampling_layers
            factor = 2 ** n_hyperencoder_downsamples
            y = utils.pad_factor(input_image=y, spatial_dims=y.size()[2:], factor=factor)

        # Pass y through the hyperprior encoder
        compression_output = self.Hyperprior.compress_forward(y, spatial_shape)
        attained_hbpp = 32 * len(compression_output.hyperlatents_encoded) / np.prod(spatial_shape)
        attained_lbpp = 32 * len(compression_output.latents_encoded) / np.prod(spatial_shape)
        attained_bpp = 32 * ((len(compression_output.hyperlatents_encoded) +
                              len(compression_output.latents_encoded)) / np.prod(spatial_shape))

        if silent is False:
            self.logger.info('[ESTIMATED]')
            self.logger.info(f'BPP: {compression_output.total_bpp:.3f}')
            self.logger.info(f'HL BPP: {compression_output.hyperlatent_bpp:.3f}')
            self.logger.info(f'L BPP: {compression_output.latent_bpp:.3f}')

            self.logger.info('[ATTAINED]')
            self.logger.info(f'BPP: {attained_bpp:.3f}')
            self.logger.info(f'HL BPP: {attained_hbpp:.3f}')
            self.logger.info(f'L BPP: {attained_lbpp:.3f}')

        return compression_output

    def decompress(self, compression_output):
        """
            * Recover z* from compressed message.
            * Pass recovered hyperlatents through mean-scale hyperprior decoder obtain mean,
            scale over latents: z -> hyperdecoder() -> (mu, sigma).
            * Use latent entropy model to recover y* from compressed image.
            * Pass quantized latent through generator to obtain the reconstructed image.
            y* -> Generator() -> x*.
        """
        assert self.model_mode == ModelModes.EVALUATION and self.training is False, (
            f"Set the model mode to {ModelModes.EVALUATION} to enable decompress"
        )
        latent_decode = self.Hyperprior.decompress_forward(compression_output, device=utils.get_device())

        #  Feed the latent quantized to G
        reconstruct = self.Generator(latent_decode)

        if self.args.normalize_input_image:
            reconstruct = torch.tanh(reconstruct)

        # Undo padding
        image_dims = compression_output.spatial_shape
        reconstruct = reconstruct[:, :, :image_dims[0], :image_dims[1]]

        if self.args.normalize_input_image:
            # Convert range from [-1, 1] to [0, 1]
            reconstruct = (reconstruct + 1.0) / 2
            reconstruct = torch.clamp(reconstruct, min=0.0, max=1.0)

        return reconstruct

    def forward(self, x, train_generator=False, return_intermediates=False, writeout=True):
        self.writeout = writeout
        losses = dict()

        if train_generator:
            # Define a step as one cycle of G-D training
            self.step_counter += 1

        intermediates, hyperinfo = self.compression_forward(x)

        if self.model_mode == ModelModes.EVALUATION:
            reconstruction = intermediates.reconstruction

            if self.args.normalize_input_image is True:
                # [-1.,1.] -> [0.,1.]
                reconstruction = (reconstruction + 1.) / 2.

            reconstruction = torch.clamp(reconstruction, min=0., max=1.)

            return reconstruction, intermediates.q_bpp

        compression_model_loss = self.compression_loss(intermediates=intermediates, hyperinfo=hyperinfo)

        if self.use_discriminator is True:
            # Only send gradients to generator when training generator via
            # `train_generator` flag
            D_loss, G_loss = self.GAN_loss(intermediates, train_generator)
            weighted_G_loss = self.args.beta * G_loss

            compression_model_loss += weighted_G_loss
            losses['disc'] = D_loss

        losses['compression'] = compression_model_loss

        # Bookkeeping
        if self.step_counter % self.log_interval == 1:
            self.store_loss('weighted_compression_loss', compression_model_loss.item())

        if return_intermediates is True:
            return losses, intermediates
        else:
            return losses


if __name__ == '__main__':
    import itertools

    compress_test = False

    if compress_test is True:
        model_mode = ModelModes.EVALUATION
    else:
        model_mode = ModelModes.TRAINING

    logger = utils.logger_setup(logpath=os.path.join(directories.experiment, 'logs.txt'),
                                filepath=os.path.abspath(os.path.join(__file__, "../../../")))
    device = utils.get_device()
    logger.info(f'Using device {device}')
    storage_train = defaultdict(list)
    storage_test = defaultdict(list)

    model = HIFICModel(hific_args, logger, storage_train, storage_test, model_mode=model_mode, model_type=ModelTypes.COMPRESSION_GAN)
    model.to(device)

    logger.info(model)

    transform_param_names = list()
    transform_params = list()
    logger.info('ALL PARAMETERS')
    for n, p in model.named_parameters():
        if ('Encoder' in n) or ('Generator' in n):
            transform_param_names.append(n)
            transform_params.append(p)
        if ('analysis' in n) or ('synthesis' in n):
            transform_param_names.append(n)
            transform_params.append(p)
        logger.info(f'{n} - {p.shape}')

    logger.info('AMORTIZATION PARAMETERS')
    amortization_named_parameters = itertools.chain.from_iterable(
            [am.named_parameters() for am in model.amortization_models])
    for n, p in amortization_named_parameters:
        logger.info(f'{n} - {p.shape}')

    logger.info('AMORTIZATION PARAMETERS')
    for n, p in zip(transform_param_names, transform_params):
        logger.info(f'{n} - {p.shape}')

    logger.info('HYPERPRIOR PARAMETERS')
    for n, p in model.Hyperprior.hyperlatent_likelihood.named_parameters():
        logger.info(f'{n} - {p.shape}')

    if compress_test is False:
        logger.info('DISCRIMINATOR PARAMETERS')
        for n, p in model.Discriminator.named_parameters():
            logger.info(f'{n} - {p.shape}')

    logger.info("Number of trainable parameters: {}".format(utils.count_parameters(model)))
    logger.info("Estimated size: {} MB".format(utils.count_parameters(model) * 4. / 10**6))

    B = 10
    shape = [B, 3, 256, 256]
    x = torch.randn(shape).to(device)

    start_time = time.time()

    if compress_test is True:
        model.eval()
        logger.info('Starting compression with input shape {}'.format(shape))
        compression_output = model.compress(x)
        reconstruction = model.decompress(compression_output)

        logger.info(f"n_bits: {compression_output.total_bits}")
        logger.info(f"bpp: {compression_output.total_bpp}")
        logger.info(f"MSE: {torch.mean(torch.square(reconstruction - x)).item()}")
    else:
        logger.info('Starting forward pass with input shape {}'.format(shape))
        losses = model(x)
        compression_loss, disc_loss = losses['compression'], losses['disc']

    logger.info('Delta t {:.3f}s'.format(time.time() - start_time))
