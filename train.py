import sys
from collections import OrderedDict
from tqdm import tqdm

from options.train_options import TrainOptions

from trainers.trainer import Trainer, get_valid_bone_tensors
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer

from util.util import init_distributed

import torch
import data
import time
import wandb

# parse options
opt = TrainOptions().parse()
if opt.debug :
    opt.display_freq = 1
    opt.print_freq = 1
    opt.save_latest_freq = 1
    opt.save_epoch_freq = 1
# print options to help debugging
print(' '.join(sys.argv))

opt.deterministic = False
opt.benchmark = True
init_distributed()

# load the dataset
dataloader = data.create_dataloader(opt)

# Validation Setting
opt.phase = 'test'
dataloader_val = data.create_dataloader(opt, valid = True)
opt.phase = 'train'

# create trainer for our model
trainer = Trainer(opt)

iter_counter = IterationCounter(opt, len(dataloader))
visualizer = Visualizer(opt)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(tqdm(dataloader), start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()
        start = time.time()
        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)
        # print(-(time.time() - start))
        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            fake_image_t, fake_image_s = trainer.get_latest_generated()
            visuals = OrderedDict([('train_source', fake_image_s),
                                   ('train_target', fake_image_t),
                                   # ('train_4bone', data_i['B2']),
                                   ])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()
        # break

    for i, data_i in tqdm(enumerate(dataloader_val), desc='Validation images generating') :
        _, (fake_image_t, fake_image_s) = trainer.model(data_i, mode='inference')

        # bone_test = get_valid_bone_tensors(dataloader_val, trainer.model.module, data_i['P1'][0].cuda(), data_i['B2'][0].cuda())
        visuals = OrderedDict([('valid_source', fake_image_s),
                               ('valid_target', fake_image_t),

                               # ('valid_4bone', data_i['B2']),
                               # ('valid_b_test', bone_test)
                               ])

        visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)
        break

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')

