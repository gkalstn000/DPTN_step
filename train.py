import sys
from collections import OrderedDict
from tqdm import tqdm

from options.train_options import TrainOptions

from trainers.trainer import Trainer
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
import torch
import data
import time


# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

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
            fake_image = trainer.get_latest_generated()
            visuals = OrderedDict([('train_1texture', data_i['texture']),
                                   ('train_3bone', data_i['bone']),
                                   ('train_2fake_image', fake_image),
                                   ])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()
        # break

    for i, data_i in tqdm(enumerate(dataloader_val), desc='Validation images generating') :
        fake_target = trainer.model(data_i, mode='inference')
        valid_losses = {}
        valid_losses['valid_l1'] = trainer.model.L1loss(fake_target, data_i['tgt_img'])
        valid_losses['valid_l2'] = trainer.model.L2loss(fake_target, data_i['tgt_img'])
        visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                        valid_losses, iter_counter.time_per_iter)
        visualizer.plot_current_errors(valid_losses, iter_counter.total_steps_so_far)
        visuals = OrderedDict([('valid_1texture', data_i['texture']),
                               ('valid_4bone', data_i['bone']),
                               ('valid_2ground_truth', data_i['ground_truth']),
                               ('valid_3fake_image', fake_target),
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

