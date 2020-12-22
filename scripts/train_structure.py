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
from models.geometric_layers import rot6d_to_axisAngle, rot6d_to_rotmat
from models.geometric_layers import axisAngle_to_Rot6d
from train_structure_cfg import TrainOptions

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
        self.mesh = Mesh(self.options, self.options.num_downsampling)
        self.faces = torch.cat( self.options.batch_size * [
                                self.mesh.faces.to(self.device)[None]
                                ], dim = 0 )
        self.faces = self.faces.type(torch.LongTensor).to(self.device)
        
        # Create SMPL mesh object and edges
        self.smpl = SMPL(self.options.smpl_model_path, self.device)
        self.smplEdge = torch.Tensor(np.load(self.options.smpl_edges_path)) \
                        .long().to(self.device)
                      
        # load average pose and shape 
        self.avgPose = \
            axisAngle_to_Rot6d(
                torch.Tensor(
                    np.load(self.options.MGN_avgPose_path)[None]\
                    .repeat(self.options.batch_size, axis = 0))
                    .reshape(-1, 3)).reshape(self.options.batch_size, -1)\
                .to(self.device)
        self.avgBeta = \
            torch.Tensor(np.load(self.options.MGN_avgBeta_path)[None]
                    .repeat(self.options.batch_size, axis = 0)).to(self.device)
        self.avgTrans= \
            torch.Tensor(np.load(self.options.MGN_avgTrans_path)[None]\
                    .repeat(self.options.batch_size, axis = 0)).to(self.device)
        self.avgCam  = \
            torch.Tensor([718, -1, 0, 0, 0, 0, -1])[None]\
                    .repeat_interleave(self.options.batch_size, dim = 0)\
                    .to(self.device)    # we know the rendered dataset
        
        # Create model
        if self.structuresCfg.structureList\
                ['disp']['network'] == 'dispGraphNet':
            self.model = DEBORNet(
                self.structuresCfg, 
                self.mesh.adjmat,
                self.mesh.ref_vertices.t(),
                avgPose = self.avgPose,
                avgBeta = self.avgBeta,
                avgTrans= self.avgTrans,
                ).to(self.device)
        else:
            self.model = DEBORNet(
                self.structuresCfg,
                avgPose = self.avgPose,
                avgBeta = self.avgBeta,
                avgTrans= self.avgTrans,
                avgCam  = self.avgCam,
                ).to(self.device)
            
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
            input_batch['smplGT']['pose'].reshape(-1, 144),   # 24 * 6 = 144
            input_batch['smplGT']['betas'],
            input_batch['smplGT']['trans']],
            dim = 1).float()
        dispGT = input_batch['meshGT']['displacement'].reshape([-1, 20670])
        GT = {'img' : input_batch['img'],
              'disp': dispGT,
              'text': input_batch['meshGT']['texture'],
              'camera': input_batch['cameraGT'],
              'indexMap': input_batch['indexMapGT'],
              'SMPL': 
                  (smplGT, torch.cat([smplGT,
                      input_batch['cameraGT']['f_rot'][:,0,:]],  dim = 1))\
              [self.structuresCfg.structureList['SMPL']['network']=='iterNet']
              }
        
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
        
        test_loss_supVis, test_loss = 0, 0
        test_loss_render, test_loss_latent = 0, 0
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
            # prepare GT
            smplGT = torch.cat([
                batch_toDEV['smplGT']['betas'],
                batch_toDEV['smplGT']['pose'].reshape(-1, 144),
                batch_toDEV['smplGT']['trans']],
                dim = 1).float()
            dispGT = batch_toDEV['meshGT']['displacement'].reshape([-1, 20670])
            GT = {
                'img' : batch_toDEV['img'],
                'disp': dispGT,
                'text': batch_toDEV['meshGT']['texture'],
                'camera': batch_toDEV['cameraGT'],
                'indexMap': batch_toDEV['indexMapGT'],
                'SMPL': 
                    (smplGT, torch.cat([smplGT,
                        batch_toDEV['cameraGT']['f_rot'][:,0,:]],  dim = 1))\
                [self.structuresCfg.structureList['SMPL']['network']=='iterNet']
                }           

            with torch.no_grad():    # disable grad
                pred, codes = self.model(GT['img'], GT)
                loss = self.model.computeLoss(pred, codes, GT)
                # save for comparison
                if step == 0:
                    self.save_sample(batch_toDEV, pred)
                
            test_loss += loss['loss']
            test_loss_supVis += loss['supVisLoss']
            test_loss_latent += loss['latCodeLoss']
            test_loss_render += loss['renderLoss']
                        
        test_loss = test_loss/len(self.test_data_loader)
        test_loss_supVis = test_loss_supVis/len(self.test_data_loader)
        test_loss_latent = test_loss_latent/len(self.test_data_loader)
        test_loss_render = test_loss_render/len(self.test_data_loader)
        
        lossSummary = {'test_loss': test_loss, 
                       'test_loss_supVis' : test_loss_supVis,
                       'test_loss_latent' : test_loss_latent,
                       'test_loss_render' : test_loss_render
                       }
        self.test_summaries(lossSummary)
        
    def save_sample(self, data, prediction, ind = 0):
        """Saving a sample for visualization and comparison"""
        folder = pjn(self.options.summary_dir, 'test/')
        _input = pjn(folder, 'input_image.png')
        batchs = self.options.batch_size
        
        # save the input image, if not saved
        if not isfile(_input):
            plt.imsave(_input, data['img'][ind].cpu().permute(1,2,0).clamp(0,1).numpy())
        # save indexMap
        savepath = pjn( 
            folder, '%s_iters%d_prediction.png'%
            (data['imgname'][ind].split('/')[-1][:-4], 
            self.step_count) )
        plt.imsave(savepath,
                   prediction['indexMap'][ind].cpu().permute(1,2,0).clamp(0,1).numpy())    
        
        # save body pose if available
        if self.structuresCfg.structureList['SMPL']['enable']:
            bodyList = {}
            
            # create GT undressed body vetirces
            jointsRot = rot6d_to_axisAngle(
                data['smplGT']['pose'][ind].to(torch.float)[None,:])
            bodyList['bodyGT'] = self.smpl(
                jointsRot[None].reshape(1, -1), 
                data['smplGT']['betas'][ind].to(torch.float)[None,:], 
                data['smplGT']['trans'][ind].to(torch.float)[None,:]).cpu()
            
            # create predicted undressed body vetirces
            jointsRot= rot6d_to_axisAngle(
                prediction['SMPL'][ind][:144].view(-1, 6))
            bodyList['bodyPred'] = self.smpl(
                jointsRot[None].reshape(1, -1), 
                prediction['SMPL'][ind][144:154][None], 
                prediction['SMPL'][ind][154:157][None]).cpu()

            if self.structuresCfg.structureList['disp']['enable']:
                bodyList['bodyGT_GTCloth'] = \
                    bodyList['bodyGT'] + data['disp'][ind].cpu()[None,:]      
                bodyList['bodyGT_PredCloth'] = \
                    bodyList['bodyGT'] + prediction['disp'][ind].cpu()[None,:]          
                bodyList['bodyPred_PredCloth']= \
                    bodyList['bodyPred']+prediction['disp'][ind].cpu()[None,:]  
                                
            # Create meshes and save them
            for key, val in bodyList.items():
                MeshGT = create_fullO3DMesh(val[0], self.faces.cpu()[0])    
                savepath = '_'.join(savepath.split('_')[:-1])+'_%s.obj'%key
                o3d.io.write_triangle_mesh(savepath, MeshGT)
        
        # save the projection if available
        if self.structuresCfg.structureList['camera']['enable']: 
            cameraPred = prediction['camera']
            predPixels, visibility = self.persCamera(
                fx = cameraPred[:,0].double(), 
                fy = cameraPred[:,0].double(),  
                cx = torch.ones(1)*self.options.img_res/2, 
                cy = torch.ones(1)*self.options.img_res/2, 
                rotation = rot6d_to_rotmat(cameraPred[:,1:7]).double(), 
                translation = data['camera']['t'][:,None,:].double(), 
                points= bodyList['bodyPred_PredCloth'].double(), 
                faces = self.faces.repeat_interleave(1, dim = 0).double(), 
                visibilityOn = True)
            
            # convert data type and remove pixels out of the image boundary.
            pixels = predPixels[0].round()
            keptPixs =  (pixels[:,0]<self.options.img_res)*(pixels[:,0]>=0)\
                       *(pixels[:,1]<self.options.img_res)*(pixels[:,1]>=0)
            pixels = pixels[keptPixs]
            
            # remove invisible vertices/pixels
            batchId, indices = torch.where(visibility)
            indices = indices[keptPixs]
            
            # create GT indexMap
            indexMap = torch.zeros(
                [self.options.img_res, self.options.img_res, 3])
            indexMap[pixels[:,1].long(), pixels[:,0].long()] = \
                self.indicesToCode(indices)
            

if __name__ == '__main__':
    
    import json
    
    # read preparation configurations
    cfgs = TrainOptions()
    cfgs.parse_args()
    
    with open( pjn(cfgs.args.log_dir, 'structures.json'), 'w') as file:
        json.dump(structure_options, file, indent=4)

    # confirm general settings
    for arg in sorted(vars(cfgs.args)):
        print('%-25s:'%(arg), getattr(cfgs.args, arg)) 
    msg = 'Do you confirm that the settings are correct?'
    assert input("%s (Y/N) " % msg).lower() == 'y', 'Settings are not comfirmed.'
    
    # confirm structure settings
    for key, val in structure_options.items():
        if isinstance(val, dict):
            print('%-25s:'%key)
            for ikey, ival in val.items():
                if isinstance(ival, dict):
                    print('\t%-21s:'%ikey)
                    for jkey, jval in ival.items():
                        print('\t\t%-17s:'%(jkey), jval) 
                else:
                    print('\t%-21s:'%(ikey), ival) 
        else:   
            print('%-25s:'%(key), val) 
    msg = 'Do you confirm that the structures are correct?'
    assert input("%s (Y/N) " % msg).lower() == 'y', 'Settings are not comfirmed.'

    mgn_trainer = trainer(cfgs.args)
    mgn_trainer.train()
    