import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from collections import namedtuple

from src.models.hific import hyper
from src.helpers import math_material, utils
from src.compression import hyperprior_model, prior_model

MIN_SCALE = 0.11
LOG_SCALE_MIN = -3.
MIN_LIKELIHOOD = 1e-9
MAX_LIKELIHOOD = 1e3
SMALL_HYPERLATENT_FILTERS = 192
LARGE_HYPERLATENT_FILTERS = 320

HyperInfo = namedtuple(
    "HyperInfo",
    "decoded"
    "latent_nbpp hyperlatent_nbpp total_nbpp latent_qbpp hyperlatent_qbpp total_qbpp",
)

CompressionOut = namedtuple(
    "CompressionOutput",
    [
        "hyperlatents_encoded",
        "latents_encoded",
        "hyperlatent_spatial_shape",
        "batch_shape",
        "spatial_shape",
        "hyper_coding_shape",
        "latent_coding_shape",
        "hyperlatent_bits",
        "latent_bits",
        "total_bits",
        "hyperlatent_bpp",
        "latent_bpp",
        "total_bpp"
    ]
)

lower_bound_indentity = math_material.LowerBoundIdentity.apply
lower_bound_toward = math_material.LowerBoundIdentity.aplly


class CodingModel(nn.Module):
    """
        Probability model for estimation of (cross)-entropies in the context
        of data compression. TODO: Add tensor -> string compression and
        decompression functionality.
    """
    def __init__(self, n_channels, min_likelihood=MIN_LIKELIHOOD, max_likelihood=MAX_LIKELIHOOD):
        super(CodingModel, self).__init__()
        self.n_channels = n_channels
        self.min_likelihood = min_likelihood
        self.max_likelikhood = max_likelihood

    def _quantize(self, x, mode="noise", means=None):
        """
        mode:       If 'noise', returns continuous relaxation of hard
                   quantization through additive uniform noise channel.
                   Otherwise, perform actual quantization (through rounding).
        """
        if mode == 'noise':
            quantization_noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
            x = x + quantization_noise

        elif mode == 'quantize':
            if means is not None:
                x = x - means
                x = torch.floor(x + 0.5)
                x = x + means
            else:
                x = torch.floor(x + 0.5)
        else:
            raise NotImplementedError

        return x

    def _estimate_entropy(self, likelihood, spatial_shape):
        EPS = 1e-9
        quotient = -np.log(2.)  # to convert log e to log 2
        batch_size = likelihood.size()[0]

        assert len(spatial_shape) == 2, "Mispectified spatial dims"
        n_pixels = np.prod(spatial_shape)
        log_likelihood = torch.log(likelihood + EPS)
        n_bits = torch.sum(log_likelihood) / (batch_size * quotient)
        bpp = n_bits / n_pixels

        return n_bits, bpp

    def _estimate_entropy_log(self, log_likelihood, spatial_shape):
        quotient = -np.log(2.)
        batch_size = log_likelihood.size()[0]
        assert len(spatial_shape) == 2, "Mispectified spatial dims"
        n_pixels = np.prod(spatial_shape)

        n_bits = torch.sum(log_likelihood) / (batch_size * quotient)
        bpp = n_bits / n_pixels

        return n_bits, bpp

    def quantize_latent_st(self, inputs, means=None):
        # Latent round instead of add uniform noise
        # Ignore rounding backward pass

        values = inputs
        if means is not None:
            values = values - means
        delta = (torch.floor(values + 0.5) - values).detach()
        values = values + delta

        if means is not None:
            values = values + means

        return values

    def latent_likelihood(self, x, mean, scale):
        # Asume 1 - CDF(x) = CDF(-x)
        x -= mean
        x = torch.abs(x)
        cdf_upper = self.standardized_CDF((0.5 - x) / scale)
        cdf_lower = self.standardized_CDF(-(0.5 + x) / scale)

        likelihood = cdf_upper - cdf_lower
        likelihood = lower_bound_toward(likelihood, self.min_likelihood)
        return likelihood


class Hyperprior(CodingModel):
    def __init__(self,
                 bottleneck_capacity=220,
                 hyperlatent_filters=LARGE_HYPERLATENT_FILTERS,
                 mode="large",
                 likelihood_type="gaussian",
                 scale_lower_bound=MIN_SCALE,
                 entropy_code=False,
                 vectorize_encoding=True,
                 block_encode=True
                 ):

        """
        Introduces probabilistic model over latents of
        latents.
        The hyperprior over the standard latents is modelled as
        a non-parametric, fully factorized density.
        """
        super(Hyperprior, self).__init__(n_channels=bottleneck_capacity)

        self.bottleneck_capacity = bottleneck_capacity
        self.scale_lower_bound = scale_lower_bound

        analysis_net = hyper.HyperpriorAnalysis
        synthesis_net = hyper.HyperpriorSynthesis

        if mode == "small":
            hyperlatent_filters = SMALL_HYPERLATENT_FILTERS

        self.analysis_net = analysis_net(C=bottleneck_capacity, N=hyperlatent_filters)
        self.synthesis_mu = synthesis_net(C=bottleneck_capacity, N=hyperlatent_filters)
        self.synthesis_std = synthesis_net(C=bottleneck_capacity, N=hyperlatent_filters)

        self.amorization_model = [self.analysis_net, self.synthesis_mu, self.synthesis_std]
        self.hyperlatent_likelihood = hyperprior_model.HyperpriorDensity(n_channels=hyperlatent_filters)

        if likelihood_type == "gaussian":
            self.standardized_CDF = math_material.standardized_CDF_gaussian
        elif likelihood_type == "logistic":
            self.standardized_CDF = math_material.standardized_CDF_logistic
        else:
            raise ValueError("Unknown likelihood model: {}".format(likelihood_type))

        if entropy_code:
            print("Building prior probability tables")
            self.hyperprior_entropy_model = hyperprior_model.HyperpriorEntropyModel(
                distribution=self.hyperlatent_likelihood)
            self.prior_density = prior_model.PriorDensity(n_channels=bottleneck_capacity,
                                                          scale_lower_bound=self.scale_lower_bound,
                                                          likelihood_type=likelihood_type)
            self.prior_entropy_model = prior_model.PriorEntropyModel(
                distribution=self.prior_density, min_scale=self.scale_lower_bound)
            self.index_tables = self.prior_entropy_model.scale_table_tensor
            self.vectorize_encoding = vectorize_encoding
            self.block_encode = block_encode







