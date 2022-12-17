import sys
from collections import OrderedDict
from tqdm import tqdm

from options.train_options import TrainOptions

from trainers.trainer import Trainer
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
import torch
import data


# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# Validation Setting
opt.phase = 'test'
dataloader_val = data.create_dataloader(opt)
opt.phase = 'train'

# create trainer for our model
trainer = Trainer(opt)

iter_counter = IterationCounter(opt, len(dataloader))
visualizer = Visualizer(opt)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(tqdm(dataloader), start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            fake_target, fake_source = trainer.get_latest_generated()
            visuals = OrderedDict([('src_image', data_i['src_image']),
                                   ('synthesized_source_image', fake_source),
                                   ('canonical_image', data_i['canonical_image']),
                                   ('tgt_map', data_i['tgt_map']),
                                   ('real_image', data_i['tgt_image']),
                                   ('synthesized_target_image', fake_target),
                                   ])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    for i, data_i in tqdm(enumerate(dataloader_val), desc='Validation images generating') :
        fake_target, fake_source = trainer.model(data_i, mode='inference')
        # with torch.no_grad() :
        #     _, fake_target, fake_source = trainer.model(data_i, mode='generator')
        visuals = OrderedDict([('src_image_val', data_i['src_image']),
                               ('canonical_image_val', data_i['canonical_image']),
                               ('tgt_map_val', data_i['tgt_map']),
                               ('real_image_val', data_i['tgt_image']),
                               ('synthesized_target_image_val', fake_target),
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

