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
    ["input_image",             # [0, 1] (after scaling from [0, 255])
     "reconstruction",          # [0, 1]
     "latents_quantized",       # Latents post-quantization.
     "n_bpp",                   # Differential entropy estimate.
     "q_bpp"]                   # Shannon entropy estimate.
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

        self.image_dims = self.args.image_dims      # (C, H, W)
        self.batch_size = self.args.batch_size      # B

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
        self.amortization_models.extend(self.Hyperprior.amortization_models)

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