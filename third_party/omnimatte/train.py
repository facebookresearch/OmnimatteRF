# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Script for training an Omnimatte model on a video.

You need to specify the dataset ('--dataroot') and experiment name ('--name').

Example:
    python train.py --dataroot ./datasets/tennis --name tennis --gpu_ids 0,1

The script first creates a model, dataset, and visualizer given the options.
It then does standard network training. During training, it also visualizes/saves the images, prints/saves the loss
plot, and saves the model.
Use '--continue_train' to resume your previous training.

See options/base_options.py and options/train_options.py for more training options.
"""
import os
import shutil
import time
from typing import Callable, List, Optional

import sys
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))

import numpy as np
import torch
from third_party.omnimatte.options.train_options import TrainOptions
from third_party.omnimatte.third_party.data import create_dataset
from third_party.omnimatte.third_party.models import create_model
from third_party.omnimatte.third_party.util.visualizer import Visualizer
from torch.utils.tensorboard import SummaryWriter


def main(argv=None, sw: Optional[SummaryWriter] = None,
         load_data_callback: Optional[Callable] = None,
         checkpoint_callback: Optional[Callable[[List[str]], None]] = None):

    trainopt = TrainOptions()
    opt = trainopt.parse(argv)

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    # copy zbar.pth to output
    shutil.copy2(os.path.join(opt.dataroot, "zbar.pth"), os.path.join(opt.checkpoints_dir, "zbar.pth"))
    if load_data_callback is not None:
        load_data_callback()

    opt.n_epochs = int(opt.n_steps / np.ceil(dataset_size / opt.batch_size))
    opt.n_epochs_decay = int(opt.n_steps_decay / np.ceil(dataset_size / opt.batch_size))
    opt.lr_decay_epochs = int(opt.lr_decay_iters / np.ceil(dataset_size / opt.batch_size))

    model = create_model(opt)
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)

    train(model, dataset, visualizer, opt, sw, checkpoint_callback)


def train(model, dataset, visualizer, opt, sw: Optional[SummaryWriter] = None, checkpoint_callback: Optional[Callable[[List[str]], None]] = None):
    dataset_size = len(dataset)
    total_iters = 0  # the total number of training iterations
    global_step = 0

    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_lambdas(epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if i % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            global_step += 1
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            losses = model.get_current_losses()
            lr = model.optimizers[0].param_groups[0]['lr']
            if sw is not None:
                for name, value in losses.items():
                    sw.add_scalar(name, value, global_step=global_step)
                sw.add_scalar("lr", lr, global_step=global_step)

            if i % opt.print_freq == 0:  # print training losses and save logging information to the disk
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            iter_data_time = time.time()

        if epoch % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            save_result = epoch % opt.update_html_freq == 0
            model.compute_visuals()
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if epoch % opt.save_latest_freq == 0 or epoch == opt.n_epochs + opt.n_epochs_decay:   # cache our latest model every <save_latest_freq> epochs
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            for save_suffix in ['epoch_%d' % epoch, 'latest']:
                saved_files = model.save_networks(save_suffix)
                if checkpoint_callback is not None:
                    checkpoint_callback(saved_files)

        model.update_learning_rate()    # update learning rates at the end of every epoch.
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))


if __name__ == '__main__':
    main()
