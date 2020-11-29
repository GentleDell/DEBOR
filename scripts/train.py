#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 22:12:46 2020

@editor: zhantao
"""

import torch
import torch.nn as nn

from dataset import MGNDataset
from utils import Mesh, BaseTrain, CheckpointDataLoader
from models import GraphCNN, res50_plus_Dec, UNet
from train_cfg import TrainOptions


class trainer(BaseTrain):
    
    def init(self):
        # Create training and testing dataset
        self.train_ds = MGNDataset(self.options, split = 'train')
        self.test_ds  = MGNDataset(self.options, split = 'test')
        
        # test data loader is fixed
        self.test_data_loader = CheckpointDataLoader( self.test_ds,
                    batch_size  = self.options.batch_size,
                    num_workers = self.options.num_workers,
                    pin_memory  = self.options.pin_memory,
                    shuffle     = self.options.shuffle_train)

        # Create Mesh object
        self.mesh = Mesh(self.options, self.options.num_downsampling)
        self.faces = self.mesh.faces.to(self.device)

        # Create model
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
                self.mesh.ref_vertices.shape[0] * 3,    # only consider displacements currently
                ).to(self.device)
        elif self.options.model == 'unet':
            self.model = UNet(
                input_shape = (self.options.img_res, self.options.img_res, 3), 
                output_dims = 3                         # only consider displacements currently
                ).to(self.device)
            
        # Setup a optimizer for models
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
    
    def shape_loss(self, pred_vertices, gt_vertices):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        return self.criterion_shape(pred_vertices, gt_vertices)
    
    def edges_loss(self, pred_vertices, gt_vertices):
        
        return None
    
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
        if self.options.model == 'unet':
            gt_uvMaps  = input_batch['UVmapGT'].to(self.device)
            pred_uvMap = self.model(images, pose)
            loss_uv = self.uvMap_loss(pred_uvMap, gt_uvMaps)
            out_args = {'predict_unMap': pred_uvMap, 
                        'loss_basic': loss_uv
                        }
        else:
            gt_vertices_disp = input_batch['meshGT']['displacement'].to(self.device)
            pred_vertices_disp = self.model(images, pose)
            loss_shape = self.shape_loss(pred_vertices_disp, gt_vertices_disp)
            out_args = {'predict_vertices': pred_vertices_disp, 
                        'loss_basic': loss_shape
                        }
        # Add losses to compute the total loss
        loss = out_args['loss_basic']
        out_args['loss'] = loss
        
        return out_args

    def test(self):
        """"Testing process"""
        self.model.eval()    
        
        test_loss_basic, test_loss = 0, 0
        for step, batch in enumerate(self.test_data_loader):
            
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
            with torch.no_grad():    # disable grad
                out = self.train_step(batch)
                    
            test_loss += out['loss']
            test_loss_basic += out['loss_basic']
            
        test_loss = test_loss/len(self.test_data_loader)
        test_loss_basic = test_loss_basic/len(self.test_data_loader)
        
        lossSummary = {'test_loss': test_loss, 
                       'test_loss_basic' : test_loss_basic
                       }
        self.test_summaries(lossSummary)
        

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
    
    # path_to_object = '/home/zhantao/Documents/masterProject/DEBOR/datasets/Multi-Garment_dataset/125611487366942'
    # mgn_trainer.inference(path_to_object, cfgs.args)
    