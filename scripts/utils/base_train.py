#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 10:26:23 2020

@author: zhantao
"""

from os.path import join as pjn

import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
tqdm.monitor_interval = 0

from .saver import CheckpointSaver
from .data_loader import CheckpointDataLoader


class BaseTrain(object):
    
    def __init__(self, options):
        
        self.options = options
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)
        
        self.init()    # override this function to define your dataset, model, optimizers etc.
        
        self.train_writer = SummaryWriter( pjn(self.options.summary_dir, 'train') )
        self.test_writer  = SummaryWriter( pjn(self.options.summary_dir, 'test') )       
        
        # check point manager
        self.checkpoint = None
        if self.options.resume and self.saver.exists_checkpoint():
            self.checkpoint = self.saver.load_checkpoint(self.models_dict, self.optimizers_dict, checkpoint_file=self.options.checkpoint)

        if self.checkpoint is None:
            self.epoch_count = 0
            self.step_count = 0
        else:
            self.epoch_count = self.checkpoint['epoch']
            self.step_count = self.checkpoint['total_step_count']
            
        # store the previous loss for comparison
        self.last_test_loss = 10
        
    def load_pretrained(self, checkpoint_file=None):
        """Load a pretrained checkpoint.
        This is different from resuming training using --resume.
        """
        if checkpoint_file is not None:
            checkpoint = torch.load(checkpoint_file)
            for model in self.models_dict:
                if model in checkpoint:
                    self.models_dict[model].load_state_dict(checkpoint[model])
                    print('Checkpoint loaded')
    
    def train(self):
        """Training process."""
        # Run training for num_epochs epochs
        for epoch in tqdm(range(self.epoch_count, self.options.num_epochs), total=self.options.num_epochs, initial=self.epoch_count):
            
            # Create new DataLoader every epoch and (possibly) resume from an arbitrary step inside an epoch
            train_data_loader = CheckpointDataLoader(
                self.train_ds, checkpoint=self.checkpoint,
                batch_size=self.options.batch_size,
                num_workers=self.options.num_workers,
                pin_memory=self.options.pin_memory,
                shuffle=self.options.shuffle_train)

            # Iterate over all batches in an epoch
            for step, batch in enumerate(tqdm(train_data_loader, desc='Epoch '+str(epoch),
                                              total=len(self.train_ds) // self.options.batch_size,
                                              initial=train_data_loader.checkpoint_batch_idx),
                                         train_data_loader.checkpoint_batch_idx):
                
                # convert data devices
                batch_toDEV = {}
                for key, val in batch.items():
                    if isinstance(val, torch.Tensor):
                        batch_toDEV[key] = val.to(self.device)
                    else:
                        batch_toDEV[key] = val
                        
                    if isinstance(val, dict):
                        batch_toDEV[key] = {}
                        for k, v in val.items():
                            if isinstance(v, torch.Tensor):
                                batch_toDEV[key][k] = v.to(self.device)
                
                out = self.train_step(batch_toDEV)
                self.step_count += 1
                
                # Do backprop
                self.optimizer.zero_grad()
                out['loss'].backward()
                self.optimizer.step()
                
                # Tensorboard logging every summary_steps steps
                if self.step_count % self.options.summary_steps == 0:
                    self.train_summaries(out)
                    
                # Save checkpoint every checkpoint_steps steps
                if self.step_count % self.options.checkpoint_steps == 0:
                    self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step+1, self.options.batch_size, train_data_loader.sampler.dataset_perm, self.step_count) 
                    tqdm.write('Checkpoint saved')

                # Run validation every test_steps steps
                if self.step_count % self.options.test_steps == 0:
                    self.test()

            # load a checkpoint only on startup, for the next epochs
            # just iterate over the dataset as usual
            self.checkpoint=None
            if (epoch+1) % 10 == 0:
                self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, self.options.batch_size, None, self.step_count) 
    
    def train_summaries(self, summary):
        """Tensorboard logging."""
        # Save results in Tensorboard
        for key, val in summary.items():
            if 'loss' in key:
                self.train_writer.add_scalar(key, val, self.step_count)
        
    def test_summaries(self, summary):
        """Tensorboard logging."""
        # Save results in Tensorboard
        for key, val in summary.items():
            if 'loss' in key:
                self.test_writer.add_scalar(key, val, self.step_count)
    
                    
    # The following methods (with the possible exception of test) have to be implemented in the derived classes
    def init(self):
        raise NotImplementedError('You need to provide an _init_fn method')

    def train_step(self):
        raise NotImplementedError('You need to provide a _train_step method')

    def test(self):
        raise NotImplementedError('You need to provide a _test_ method')
                   