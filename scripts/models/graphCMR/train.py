#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 22:12:46 2020

@editor: zhantao
"""

import torch
import torch.nn as nn
from tqdm import tqdm
tqdm.monitor_interval = 0
from tensorboardX import SummaryWriter

from dataset import MGNDataset, CheckpointDataLoader
from utils import Mesh, CheckpointSaver
from models import GraphCNN, res50_plus_Dec, Tex2Shape
from cfg import TrainOptions


class trainer(object):
    
    def __init__(self, options):
        self.options = options
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)
        self.summary_writer = SummaryWriter(self.options.summary_dir)
        
        # create training dataset
        self.train_ds = MGNDataset(self.options)

        # create Mesh object
        self.mesh = Mesh(options, options.num_downsampling)
        self.faces = self.mesh.faces.to(self.device)

        # create model
        if self.options.model == 'graphcnn':
            self.model = GraphCNN(
                self.mesh.adjmat,
                self.mesh.ref_vertices.t(),
                num_channels=self.options.num_channels,
                num_layers=self.options.num_layers
                ).to(self.device)
        elif self.options.model == 'sizernn':
            self.model = res50_plus_Dec(
                self.options.latent_size,
                self.mesh.ref_vertices.shape[0] * 3,    # numVert x 3 displacements
                ).to(self.device)
        elif self.options.model == 'tex2shape':
            self.model = Tex2Shape(
                input_shape = (self.options.img_res, self.options.img_res, 3), 
                output_dims = 3                         # only consider displacements currently
                ).to(self.device)
            
        # Setup a joint optimizer for the 2 models
        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=self.options.lr,
            betas=(self.options.adam_beta1, 0.999),
            weight_decay=self.options.wd)
        
        # loss terms
        self.criterion_shape = nn.L1Loss().to(self.device)
        
        # Pack models and optimizers in a dict - necessary for checkpointing
        self.models_dict = {self.options.model: self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}
        
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
            
        
    def shape_loss(self, pred_vertices, gt_vertices):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        return self.criterion_shape(pred_vertices, gt_vertices)
    
    def uvMap_loss(self, pred_UVMap, gt_UVMap):
        """Compute per-pixel loss on the uv texture map."""
        return self.criterion_shape(pred_UVMap, gt_UVMap)
        
    def train_step(self, input_batch):
        """Training step."""
        self.model.train()

        # Grab data from the batch
        images = input_batch['img']

        # Feed image and pose in the GraphCNN
        # Returns full mesh
        pose = (None, input_batch['smplGT']['pose'].to(self.device))[self.options.append_pose]

        # Compute losses
        if self.options.model == 'tex2shape':
            gt_uvMaps  = input_batch['UVmapGT'].to(self.device)
            pred_uvMap = self.model(images, pose)
            loss_shape = self.uvMap_loss(pred_uvMap, gt_uvMaps)
        else:
            gt_vertices_disp = input_batch['meshGT']['displacement'].to(self.device)
            pred_vertices_disp = self.model(images, pose)
            loss_shape = self.shape_loss(pred_vertices_disp, gt_vertices_disp)

        # Add losses to compute the total loss
        loss = loss_shape

        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Pack output arguments to be used for visualization in a list
        out_args = [pred_vertices_disp, loss_shape, loss]
        out_args = [arg.detach() for arg in out_args]
        return out_args

    def train_summaries(self, input_batch, pred_vertices_disp, loss_shape,  loss):
        """Tensorboard logging."""
         
        # Save results in Tensorboard
        self.summary_writer.add_scalar('loss_shape', loss_shape, self.step_count)
        self.summary_writer.add_scalar('loss', loss, self.step_count)
        
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
                
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
                out = self.train_step(batch)
                self.step_count += 1
                
                # Tensorboard logging every summary_steps steps
                if self.step_count % self.options.summary_steps == 0:
                    self.train_summaries(batch, *out)
                    
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
            # save checkpoint after each epoch
            if (epoch+1) % 10 == 0:
                # self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, self.step_count) 
                self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, self.options.batch_size, None, self.step_count) 
        
    def test(self):
        pass

if __name__ == '__main__':
            
    # read preparation configurations
    cfgs = TrainOptions()
    cfgs.parse_args()

    for arg in sorted(vars(cfgs.args)):
        print('%-25s:'%(arg), getattr(cfgs.args, arg)) 

    # require confirmation
    msg = 'Do you confirm that the settings are correct?'
    assert input("%s (Y/N) " % msg).lower() == 'y', 'Settings are not comfirmed.'
    
    mgn_trainer = trainer(cfgs.args)
    mgn_trainer.train()
    