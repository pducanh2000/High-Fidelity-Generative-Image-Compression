import torch


class ModelTypes(object):
    COMPRESSION = 'compression'
    COMPRESSION_GAN = 'compression_gan'


class ModelModes(object):
    TRAINING = 'training'
    VALIDATION = 'validation'
    EVALUATION = 'evaluation'  # actual entropy coding


class DatasetPaths(object):
    VIMEO = 'data/vimeo_interp_test/'


hyper_params = {
    # Path


    # Model params
    # For Encoder module
    "image_shape": (3, 256, 256),
    "num_down": 4,
    "activation": "relu",
    "out_channels_base": 60,
    "out_channels_bottleneck": 220,

    # For Generator module


}