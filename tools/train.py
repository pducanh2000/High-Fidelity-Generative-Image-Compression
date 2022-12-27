import os
import time
import torch

from src.models.hific.model import HIFICModel
from src.helpers import utils

# Use it when the input sizes for the network do not vary to look for optimal set of algorithms
# Dont use it when the input sizes changes frequently it can lead to worse run time
torch.backends.cudnn.benchmark = True


def create_model(args, device, logger, storage, storage_test):

    start_time = time.time()
    model = HIFICModel(args, logger, storage, storage_test, model_type=args.model_type)
    logger.info(model)
    logger.info('Trainable parameters:')

    for n, p in model.named_parameters():
        logger.info('{} - {}'.format(n, p.shape))

    logger.info("Number of trainable parameters: {}".format(utils.count_parameters(model)))
    logger.info("Estimated size (under fp32): {:.3f} MB".format(utils.count_parameters(model) * 4. / 10**6))
    logger.info('Model init {:.3f}s'.format(time.time() - start_time))

    return model


def optimize_loss(loss, optimizer, retain_graph=False):
    loss.backward(retain_graph=retain_graph)
    optimizer.step()
    optimizer.zero_grad()


# Optimize the Hyper prior and amortization models separately
def optimize_compression_loss(compression_loss, amortization_opt, hyperlatent_likelihood_opt):
    compression_loss.backward()
    amortization_opt.step()
    hyperlatent_likelihood_opt.step()
    amortization_opt.zero_grad()
    hyperlatent_likelihood_opt.zero_grad()

def test()