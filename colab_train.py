import os
import time
import datetime
import pickle
import argparse
import itertools
from collections import defaultdict
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm.notebook import tqdm

from src.models.hific.model import HIFICModel
from src.helpers import utils
from src.dataset import dataload
from config.config_hyp import ModelTypes, mse_lpips_args, hific_args

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
    logger.info("Estimated size (under fp32): {:.3f} MB".format(utils.count_parameters(model) * 4. / 10 ** 6))
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


def test(args, model, epoch, idx, data, test_data, test_bpp, device, epoch_test_loss, storage, best_test_loss,
         start_time, epoch_start_time, logger, train_writer, test_writer):
    model.eval()
    with torch.no_grad():
        data = data.to(device, dtype=torch.float)

        losses, intermediates = model(data, return_intermediates=True, writeout=False)
        utils.save_images(train_writer, model.step_counter, intermediates.input_image, intermediates.reconstruction,
                          fname=os.path.join(args.figures_save,
                                             "recon_epoch{}_idx{}_TRAIN_{:%Y_%m_%d_%H:%M}.jpg".format(
                                                 epoch, idx, datetime.datetime.now())))

        test_data = test_data.to(device, dtype=torch.float)
        losses, intermediates = model(test_data, return_intermediates=True, writeout=True)
        utils.save_images(test_writer, model.step_counter, intermediates.input_image, intermediates.reconstruction,
                          fname=os.path.join(args.figures_save,
                                             "recon_epoch{}_idx{}_TRAIN_{:%Y_%m_%d_%H:%M}.jpg".format(
                                                 epoch, idx, datetime.datetime.now())))

        compression_loss = losses["compression"]
        epoch_test_loss.append(compression_loss.item())
        mean_test_loss = np.mean(epoch_test_loss)

        best_test_loss = utils.log(model, storage, epoch, idx, mean_test_loss, compression_loss.item(),
                                   best_test_loss, start_time, epoch_start_time, batch_size=data.shape[0],
                                   avg_bpp=test_bpp.mean().item(),
                                   header="[TEST]", logger=logger, writer=test_writer)

    return best_test_loss, epoch_test_loss


def train(args, model, train_loader, test_loader, device, logger, optimizers):
    start_time = time.time()
    test_loader_iter = iter(test_loader)
    current_D_steps, train_generator = 0, True

    best_loss, best_test_loss, mean_epoch_loss = np.inf, np.inf, np.inf
    train_writer = SummaryWriter(os.path.join(args.tensorboard_runs, "train"))
    test_writer = SummaryWriter(os.path.join(args.tensorboard_runs, "test"))
    storage, storage_test = model.storage_train, model.storage_test

    amortization_opt, hyperlatent_likelihood_opt = optimizers["amort"], optimizers["hyper"]
    if model.use_discriminator:
        disc_opt = optimizers["disc"]

    for epoch in tqdm(args.n_epochs):
        epoch_loss, epoch_test_loss = [], []
        epoch_start_time = time.time()

        if epoch > 0:
            ckpt_path = utils.save_model(model, optimizers, mean_epoch_loss, epoch, device, args=args, logger=logger)

        model.train()

        for idx, (data, bpp) in tqdm(enumerate(train_loader, 0)):
            data = data.to(device, dtype=torch.float)
            try:
                if model.use_discriminator:
                    # Train D for D_steps, then G, using distinct batches
                    losses = model(data, train_generator=train_generator)
                    compression_loss = losses["compression"]
                    disc_loss = losses["disc"]

                    if train_generator:
                        optimize_compression_loss(compression_loss, amortization_opt, hyperlatent_likelihood_opt)
                        train_generator = False
                    else:
                        optimize_loss(disc_loss, disc_opt)
                        current_D_steps += 1

                        if current_D_steps == args.discriminator_steps:
                            current_D_steps = 0
                            train_generator = True
                        continue
                else:
                    # Rate, distortion, perceptual only
                    losses = model(data, train_generator=True)
                    compression_loss = losses['compression']
                    optimize_compression_loss(compression_loss, amortization_opt, hyperlatent_likelihood_opt)
            except KeyboardInterrupt:
                # Note: saving not guaranteed!
                if model.step_counter > args.log_interval + 1:
                    logger.warning('Exiting, saving ...')
                    ckpt_path = utils.save_model(model, optimizers, mean_epoch_loss, epoch, device, args=args,
                                                 logger=logger)
                    return model, ckpt_path
                else:
                    return model, None

            if model.step_counter % args.log_interval == 1:
                epoch_loss.append(compression_loss.item())
                mean_epoch_loss = np.mean(epoch_loss)

                best_loss = utils.log(model, storage, epoch, idx, mean_epoch_loss, compression_loss.item(),
                                      best_loss, start_time, epoch_start_time, batch_size=data.shape[0],
                                      avg_bpp=bpp.mean().item(), logger=logger, writer=train_writer)

            try:
                test_data, test_bpp = next(test_loader_iter)
            except StopIteration:
                test_loader_iter = iter(test_loader)
                test_data, test_bpp = next(test_loader_iter)

            best_test_loss, epoch_test_loss = test(args, model, epoch, idx, data, test_data, test_bpp, device,
                                                   epoch_test_loss, storage_test,
                                                   best_test_loss, start_time, epoch_start_time, logger, train_writer,
                                                   test_writer)

            with open(os.path.join(args.storage_save, 'storage_{}_tmp.pkl'.format(args.name)), 'wb') as handle:
                pickle.dump(storage, handle, protocol=pickle.HIGHEST_PROTOCOL)

            model.train()
            # LR scheduling
            utils.update_lr(args, amortization_opt, model.step_counter, logger)
            utils.update_lr(args, hyperlatent_likelihood_opt, model.step_counter, logger)
            if model.use_discriminator is True:
                utils.update_lr(args, disc_opt, model.step_counter, logger)

            if model.step_counter > args.n_steps:
                logger.info('Reached step limit [args.n_steps = {}]'.format(args.n_steps))
                break

        if (idx % args.save_interval == 1) and (idx > args.save_interval):
            ckpt_path = utils.save_model(model, optimizers, mean_epoch_loss, epoch, device, args=args, logger=logger)

            # End epoch
        mean_epoch_loss = np.mean(epoch_loss)
        mean_epoch_test_loss = np.mean(epoch_test_loss)

        logger.info('===>> Epoch {} | Mean train loss: {:.3f} | Mean test loss: {:.3f}'.format(epoch,
                                                                                               mean_epoch_loss,
                                                                                               mean_epoch_test_loss))

        if model.step_counter > args.n_steps:
            break

    with open(os.path.join(args.storage_save,
                           'storage_{}_{:%Y_%m_%d_%H:%M:%S}.pkl'.format(args.name, datetime.datetime.now())),
              'wb') as handle:
        pickle.dump(storage, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ckpt_path = utils.save_model(model, optimizers, mean_epoch_loss, epoch, device, args=args, logger=logger)
    args.ckpt = ckpt_path
    logger.info("Training complete. Time elapsed: {:.3f} s. Number of steps: {}".format((time.time() - start_time),
                                                                                        model.step_counter))

    return model, ckpt_path


if __name__ == "__main__":
    description = "Learnable generative compression."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General options - see `default_config.py` for full options
    general = parser.add_argument_group('General options')
    general.add_argument("-gpu", "--gpu", type=int, default=0, help="GPU ID.")
    general.add_argument("-mt", "--model_type", required=True,
                         choices=(ModelTypes.COMPRESSION, ModelTypes.COMPRESSION_GAN),
                         help="Type of model - with or without GAN component")
    general.add_argument("-regime", "--regime", choices=('low', 'med', 'high'), default='low',
                         help="Set target bit rate - Low (0.14), Med (0.30), High (0.45)")
    general.add_argument("-force_gpu", "--force_set_gpu", help="Set GPU to given ID", action="store_true")

    # Optimization-related options
    optim_args = parser.add_argument_group("Optimization-related options")
    optim_args.add_argument('-steps', '--n_steps', type=float, default=hific_args.n_steps,
        help="Number of gradient steps. Optimization stops at the earlier of n_steps/n_epochs.")

    # Warmstart adversarial training from autoencoder/hyperprior
    warmstart_args = parser.add_argument_group("Warmstart options")
    warmstart_args.add_argument("-warmstart", "--warmstart", help="Warmstart adversarial training from autoencoder + hyperprior ckpt.", action="store_true")
    warmstart_args.add_argument("-ckpt", "--warmstart_ckpt", default=None, help="Path to autoencoder + hyperprior ckpt.")

    cmd_args = parser.parse_args()

    if (cmd_args.gpu != 0) or (cmd_args.force_set_gpu is True):
        torch.cuda.set_device(cmd_args.gpu)

    if cmd_args.model_type == ModelTypes.COMPRESSION:
        args = mse_lpips_args
    elif cmd_args.model_type == ModelTypes.COMPRESSION_GAN:
        args = hific_args

    start_time = time.time()
    device = utils.get_device()

    # Override default arguments from config file with provided command line arguments
    dictify = lambda x: dict((n, getattr(x, n)) for n in dir(x) if not (n.startswith("__") or "logger" in n))
    args_d, cmd_args_d = dictify(args), vars(cmd_args)
    args_d.update(cmd_args_d)

    args = utils.Struct(**args_d)
    args = utils.setup_generic_signature(args, special_info=args.model_type)
    args.target_rate = args.target_rate_map[args.regime]
    args.lambda_A = args.lambda_A_map[args.regime]
    args.n_steps = int(args.n_steps)

    storage = defaultdict(list)
    storage_test = defaultdict(list)
    logger = utils.logger_setup(logpath=os.path.join(args.snapshot, 'logs'), filepath=os.path.abspath(__file__))

    if args.warmstart is True:
        assert args.warmstart_ckpt is not None, 'Must provide checkpoint to previously trained AE/HP model.'
        logger.info('Warmstarting discriminator/generator from autoencoder/hyperprior model.')
        if args.model_type != ModelTypes.COMPRESSION_GAN:
            logger.warning('Should warmstart compression-gan model.')
        args, model, optimizers = utils.load_model(args.warmstart_ckpt, logger, device,
                                                   model_type=args.model_type, current_args_d=dictify(args),
                                                   strict=False, prediction=False)
    else:
        model = create_model(args, device, logger, storage, storage_test)
        model = model.to(device)
        amortization_parameters = itertools.chain.from_iterable(
            [am.parameters() for am in model.amortization_models])

        hyperlatent_likelihood_parameters = model.Hyperprior.hyperlatent_likelihood.parameters()

        amortization_opt = torch.optim.Adam(amortization_parameters,
                                            lr=args.learning_rate)
        hyperlatent_likelihood_opt = torch.optim.Adam(hyperlatent_likelihood_parameters,
                                                      lr=args.learning_rate)
        optimizers = dict(amort=amortization_opt, hyper=hyperlatent_likelihood_opt)

        if model.use_discriminator is True:
            discriminator_parameters = model.Discriminator.parameters()
            disc_opt = torch.optim.Adam(discriminator_parameters, lr=args.learning_rate)
            optimizers['disc'] = disc_opt

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1 and args.multigpu is True:
        # Not supported at this time
        raise NotImplementedError('MultiGPU not supported yet.')
        logger.info('Using {} GPUs.'.format(n_gpus))
        model = nn.DataParallel(model)

    logger.info('MODEL TYPE: {}'.format(args.model_type))
    logger.info('MODEL MODE: {}'.format(args.model_mode))
    logger.info('BITRATE REGIME: {}'.format(args.regime))
    logger.info('SAVING LOGS/CHECKPOINTS/RECORDS TO {}'.format(args.snapshot))
    logger.info('USING DEVICE {}'.format(device))
    logger.info('USING GPU ID {}'.format(args.gpu))
    logger.info('USING DATASET: {}'.format(args.dataset))

    test_loader = dataload.get_dataloader(args.kodak_dataset,
                                           json_path=args.kodak_dataset_path,
                                           batch_size=args.batch_size,
                                           logger=logger,
                                           train_mode=False,
                                           shuffle=True,
                                           normalize=args.normalize_input_image)

    train_loader = dataload.get_dataloader(args.dataset,
                                            json_path=args.dataset_path,
                                            batch_size=args.batch_size,
                                            logger=logger,
                                            train_mode=True,
                                            shuffle=True,
                                            normalize=args.normalize_input_image)

    args.n_data = len(train_loader.dataset)
    args.image_dims = train_loader.dataset.image_dims
    logger.info('Training elements: {}'.format(args.n_data))
    logger.info('Input Dimensions: {}'.format(args.image_dims))
    logger.info('Optimizers: {}'.format(optimizers))
    logger.info('Using device {}'.format(device))

    metadata = dict((n, getattr(args, n)) for n in dir(args) if not (n.startswith('__') or 'logger' in n))
    logger.info(metadata)

    """
    Train
    """
    model, ckpt_path = train(args, model, train_loader, test_loader, device, logger, optimizers=optimizers)

    """
    TODO
    Generate metrics
    """
