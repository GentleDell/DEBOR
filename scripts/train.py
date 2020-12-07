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

from tqdm import tqdm
tqdm.monitor_interval = 0
import torch
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from dataset import MGNDataset
from utils import Mesh, BaseTrain, CheckpointDataLoader
from utils.mesh_util import create_fullO3DMesh
from models import GraphCNN, res50_plus_Dec, UNet, SMPL, Loss
from train_cfg import TrainOptions

# ignore all alert from open3d except error messages
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


class trainer(BaseTrain):
    
    def init(self):
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
        self.mesh = Mesh(self.options, self.options.num_downsampling)
        self.faces = torch.cat( self.options.batch_size * [
                                self.mesh.faces.to(self.device)[None]
                                ], dim = 0 )
        self.faces = self.faces.type(torch.LongTensor).to(self.device)
        
        # Create SMPL mesh object and edges
        self.smpl = SMPL(self.options.smpl_model_path, self.device)
        self.smplEdge = torch.Tensor(np.load(self.options.smpl_edges_path)) \
                        .long()  \
                        .to(self.device)     
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
        self.loss = Loss(self.options, self.faces, self.smplEdge, self.device)
        
        # Pack models and optimizers in a dict - necessary for checkpointing
        self.models_dict = {self.options.model: self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}

            
    def train_step(self, input_batch):
        """Training step."""
        self.model.train()

        # Grab input from the batch
        images = input_batch['img']

        # Grab GT SMPL parameters and create body body for displacements
        body_vertices = self.smpl(input_batch['smplGT']['pose'].to(torch.float), 
                                  input_batch['smplGT']['betas'].to(torch.float), 
                                  input_batch['smplGT']['trans'].to(torch.float)
                                  )
        # prepare pose of joints, not used yet
        pose = (None, input_batch['smplGT']['pose'])[self.options.append_pose]
        
        # training
        pred_uvMap, pred_vertices_disp = None, None
        if self.options.model == 'unet':
            gt_uvMaps  = input_batch['UVmapGT']
            pred_uvMap = self.model(images, pose)
            loss = self.loss(predUVMaps = pred_uvMap, GTUVMaps = gt_uvMaps) 
        else:
            gt_vertices_disp = input_batch['meshGT']['displacement']
            pred_vertices_disp = self.model(images, pose)
            
            dressbodyPred = pred_vertices_disp + body_vertices
            dressbodyGT = gt_vertices_disp + body_vertices
            
            loss = self.loss(predVerts = dressbodyPred, GTVerts = dressbodyGT)
            
        out_args = loss
        out_args['predict_unMap'] = pred_uvMap
        out_args['predict_vertices'] = pred_vertices_disp
        
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

        if self.options.model == 'graphcnn':
            # Grab GT SMPL parameters and create body body for displacements
            body_vertices = self.smpl(
                data['smplGT']['pose'][ind].to(torch.float)[None,:], 
                data['smplGT']['betas'][ind].to(torch.float)[None,:], 
                data['smplGT']['trans'][ind].to(torch.float)[None,:]).cpu()
            
            # Add displacements to naked body
            predictedBody = body_vertices + \
                            prediction['predict_vertices'][ind].cpu()[None,:]            
            # Create predicted mesh and save it
            predictedMesh = create_fullO3DMesh(predictedBody[0], self.faces.cpu()[0])
            savepath = pjn( folder, '%s_iters%d_prediction.obj'%
                           (data['imgname'][ind].split('/')[-1][:-4], 
                            self.step_count) )
            o3d.io.write_triangle_mesh(savepath, predictedMesh)
            
            # If there is no ground truth, create and save it
            savepath = pjn( folder, '%s_groundtruth.obj'%
                           (data['imgname'][ind].split('/')[-1][:-4]) )
            if not isfile(savepath):
                grndTruthBody = body_vertices + \
                            data['meshGT']['displacement'][ind].cpu()[None,:]
                grndTruthMesh = create_fullO3DMesh(grndTruthBody[0], self.faces.cpu()[0])
                o3d.io.write_triangle_mesh(savepath, grndTruthMesh)
            
        elif self.options.model == 'unet':
            # save predicted uvMap
            savepath = pjn(folder, '%s_iters%d_prediction.png'%
                           (data['imgname'][ind].split('/')[-1][:-4], 
                            self.step_count) )
            plt.imsave(savepath, 
                       prediction['predict_unMap'][ind].cpu().clamp(0,1).numpy())
            
            # If there is no ground truth, create and save it
            savepath = pjn(folder, '%s_groundtruth.png'%
                           (data['imgname'][ind].split('/')[-1][:-4])) 
            if not isfile(savepath):
                plt.imsave(savepath, data['UVmapGT'][ind].cpu().numpy())
            

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
    