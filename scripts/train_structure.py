#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 22:12:46 2020

@author: zhantao
"""
from os.path import abspath, isfile, join as pjn
import sys
if abspath('./') not in sys.path:
    sys.path.append(abspath('./'))
if abspath('./third_party/smpl_webuser') not in sys.path:
    sys.path.append(abspath('./third_party/smpl_webuser'))
if abspath('./third_party/pytorch-msssim') not in sys.path:
    sys.path.append(abspath('./third_party/pytorch-msssim'))
from collections import namedtuple

from tqdm import tqdm
tqdm.monitor_interval = 0
import torch
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from dataset import MGNDataset
from utils import Mesh, BaseTrain, CheckpointDataLoader
from utils.mesh_util import create_fullO3DMesh
from models import SMPL, DEBORNet
from models.structures_options import structure_options
from train_cfg import TrainOptions

# ignore all alert from open3d except error messages
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


class trainer(BaseTrain):
    
    def init(self):
        # load structure configuration 
        self.structuresCfg = namedtuple(
            'options', structure_options.keys())(**structure_options)
        
        # Create training and testing dataset
        self.train_ds = MGNDataset(self.options, split = 'train')
        self.test_ds  = MGNDataset(self.options, split = 'test')
        
        # test data loader is fixed and disable shuffle as it is unnecessary.
        self.test_data_loader = CheckpointDataLoader( self.test_ds,
                    batch_size  = self.options.batch_size,
                    num_workers = self.options.num_workers,
                    pin_memory  = self.options.pin_memory,
                    shuffle     = False)

        # Create Mesh (graph) object
        if self.structuresCfg.structureList\
            ['disp']['network'] == 'dispGraphNet':
            self.mesh = Mesh(self.options, self.options.num_downsampling)
            self.faces = torch.cat( self.options.batch_size * [
                                    self.mesh.faces.to(self.device)[None]
                                    ], dim = 0 )
            self.faces = self.faces.type(torch.LongTensor).to(self.device)
        
        # Create SMPL mesh object and edges
        self.smpl = SMPL(self.options.smpl_model_path, self.device)
        self.smplEdge = torch.Tensor(np.load(self.options.smpl_edges_path)) \
                        .long().to(self.device)
                        
        # Create model
        if self.structuresCfg.structureList\
                ['disp']['network'] == 'dispGraphNet':
            self.model = DEBORNet(
                self.structuresCfg, 
                self.mesh.adjmat,
                self.mesh.ref_vertices.t()).to(self.device)
        else:
            self.model = DEBORNet(self.structuresCfg).to(self.device)
            
        # Setup a optimizer for models
        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=self.options.lr,
            betas=(self.options.adam_beta1, 0.999),
            weight_decay=self.options.wd)
                
        # Pack models and optimizers in a dict - necessary for checkpointing
        self.models_dict = {self.options.model: self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}

            
    def train_step(self, input_batch):
        """Training step."""
        self.model.train()
 
        # prepare GT data
        smplGT = torch.cat([
            input_batch['smplGT']['betas'],
            input_batch['smplGT']['pose'].reshape(-1, 144),
            input_batch['smplGT']['trans']],
            dim = 1).float()
        dispGT = input_batch['meshGT']['displacement'].reshape([-1, 20670])
        GT = {'img' : input_batch['img'],
              'SMPL': smplGT,
              'disp': dispGT,
              'text': input_batch['meshGT']['texture'],
              'camera': input_batch['cameraGT'],
              'indexMap': input_batch['indexMapGT']}
        
        # forward pass
        pred, codes = self.model(input_batch['img'], GT)
        
        # loss
        loss = self.model.computeLoss(pred, codes, GT)
        
        out_args = loss
        out_args['prediction'] = pred 
        
        return out_args

    def test(self):
        """"Testing process"""
        self.model.eval()    
        
        test_loss_vertices, test_loss_uvmaps, test_loss = 0, 0, 0
        for step, batch in enumerate(tqdm(self.test_data_loader, desc='Test',
                                          total=len(self.test_ds) // self.options.batch_size,
                                          initial=self.test_data_loader.checkpoint_batch_idx),
                                     self.test_data_loader.checkpoint_batch_idx):
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
                                
            with torch.no_grad():    # disable grad
                out = self.train_step(batch_toDEV)
                    
            test_loss += out['loss']
            test_loss_vertices += out['loss_vertices']
            test_loss_uvmaps += out['loss_uvMap']
            
            # save for comparison
            if step == 0:
                self.save_sample(batch_toDEV, out)
            
        test_loss = test_loss/len(self.test_data_loader)
        test_loss_vertices = test_loss_vertices/len(self.test_data_loader)
        test_loss_uvmaps = test_loss_uvmaps/len(self.test_data_loader)
        
        lossSummary = {'test_loss': test_loss, 
                       'test_loss_vertices' : test_loss_vertices,
                       'test_loss_uvmaps': test_loss_uvmaps
                       }
        self.test_summaries(lossSummary)
        
        
    def save_sample(self, data, prediction, ind = 0):
        """Saving a sample for visualization and comparison"""
        folder = pjn(self.options.summary_dir, 'test/')
        _input = pjn(folder, 'input_image.png')
        
        # save the input image, if not saved
        if not isfile(_input):
            plt.imsave(_input, data['img'][ind].cpu().permute(1,2,0).clamp(0,1).numpy())

        # if self.options.model == 'graphcnn':
        #     # Grab GT SMPL parameters and create body body for displacements
        #     body_vertices = self.smpl(
        #         data['smplGT']['pose'][ind].to(torch.float)[None,:], 
        #         data['smplGT']['betas'][ind].to(torch.float)[None,:], 
        #         data['smplGT']['trans'][ind].to(torch.float)[None,:]).cpu()
            
        #     # Add displacements to naked body
        #     predictedBody = body_vertices + \
        #                     prediction['predict_vertices'][ind].cpu()[None,:]            
        #     # Create predicted mesh and save it
        #     predictedMesh = create_fullO3DMesh(predictedBody[0], self.faces.cpu()[0])
        #     savepath = pjn( folder, '%s_iters%d_prediction.obj'%
        #                    (data['imgname'][ind].split('/')[-1][:-4], 
        #                     self.step_count) )
        #     o3d.io.write_triangle_mesh(savepath, predictedMesh)
            
        #     # If there is no ground truth, create and save it
        #     savepath = pjn( folder, '%s_groundtruth.obj'%
        #                    (data['imgname'][ind].split('/')[-1][:-4]) )
        #     if not isfile(savepath):
        #         grndTruthBody = body_vertices + \
        #                     data['meshGT']['displacement'][ind].cpu()[None,:]
        #         grndTruthMesh = create_fullO3DMesh(grndTruthBody[0], self.faces.cpu()[0])
        #         o3d.io.write_triangle_mesh(savepath, grndTruthMesh)
            
        # elif self.options.model == 'unet':
        #     # save predicted uvMap
        #     savepath = pjn(folder, '%s_iters%d_prediction.png'%
        #                    (data['imgname'][ind].split('/')[-1][:-4], 
        #                     self.step_count) )
        #     plt.imsave(savepath, 
        #                prediction['predict_unMap'][ind].cpu().clamp(0,1).numpy())
            
        #     # If there is no ground truth, create and save it
        #     savepath = pjn(folder, '%s_groundtruth.png'%
        #                    (data['imgname'][ind].split('/')[-1][:-4])) 
        #     if not isfile(savepath):
        #         plt.imsave(savepath, data['UVmapGT'][ind].cpu().numpy())
                

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
    