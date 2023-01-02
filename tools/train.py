import os
import time
import datetime
import pickle
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm, trange

from src.models.hific.model import HIFICModel
from src.helpers import utils
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
        utils.save_images(train_writer, model.step_counter, intermediates.input_image, intermediates.reconstructions,
                          fname=os.path.join(args.figures_save,
                                             "recon_epoch{}_idx{}_TRAIN_{:%Y_%m_%d_%H:%M}.jpg".format(
                                                 epoch, idx, datetime.datetime.now())))

        test_data = test_data.to(device, dtype=torch.float)
        losses, intermediates = model(test_data, return_intermediates=True, writeout=True)
        utils.save_images(test_writer, model.step_counter, intermediates.input_image, intermediates.reconstructions,
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

    for epoch in trange(args.n_epochs, desc="Epoch"):
        epoch_loss, epoch_test_loss = [], []
        epoch_start_time = time.time()

        if epoch > 0:
            ckpt_path = utils.save_model(model, optimizers, mean_epoch_loss, epoch, device, args=args, logger=logger)

        model.train()

        for idx, (data, bpp) in enumerate(tqdm(train_loader, decs="Train"), 0):
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

            if model.step_counter % args.log_iterval == 1:
                epoch_loss.append(compression_loss.item())
                mean_epoch_loss = np.mean(epoch_loss)

                best_loss = utils.log(model, storage, epoch, idx, mean_epoch_loss, compression_loss.item(),
                                      best_loss, start_time, epoch_start_time, batch_size=data.shape[0])

            try:
                test_data, test_bpp = test_loader_iter.next()
            except StopIteration:
                test_loader_iter = iter(test_loader)
                test_data, test_bpp = test_loader_iter.next()

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
    cmd_args = parser.parse_args()

    if (cmd_args.gpu != 0) or (cmd_args.force_set_gpu is True):
        torch.cuda.set_device(cmd_args.gpu)

    if cmd_args.model_type == ModelTypes.COMPRESSION:
        args = mse_lpips_args
    elif cmd_args.model_type == ModelTypes.COMPRESSION_GAN:
        args = hific_args
