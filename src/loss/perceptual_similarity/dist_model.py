from __future__ import absolute_import

import sys
import numpy as np
import torch
from torch import nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
from src.loss.perceptual_similarity.base_model import BaseModel
from scipy.ndimage import zoom
import fractions
import functools
import skimage.transform
from tqdm import tqdm

from . import networks_basic as networks


class DistModel(BaseModel):
    def __init__(self):
        super(DistModel, self).__init__()
        self.model = None
        self.net = None
        self.is_train = None
        self.spatial = None

    def initialize(self, model='net-lin', net='alex', colorspace='Lab', pnet_rand=False, pnet_tune=False,
                   model_path=None, use_gpu=True, printNet=False, spatial=False, is_train=False, lr=.0001, beta1=0.5,
                   gpu_ids=(0,)):
        BaseModel.initialize(self, use_gpu=True, gpu_ids=gpu_ids)

        self.model = model
        self.name = "%s_[%s]" % (self.model, self.net)
        self.net = net
        self.is_train = is_train
        self.spatial = spatial
        self.gpu_ids = gpu_ids



