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
from utils.vis_util import read_Obj
from models import SMPL, frameVIBE, simple_renderer
from models import camera as perspCamera
from models.loss import VIBELoss
from models.structures_options import structure_options
from models.geometric_layers import rot6d_to_axisAngle, rot6d_to_rotmat
from models.geometric_layers import axisAngle_to_Rot6d, rotationMatrix_to_axisAngle
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

        # Create SMPL Mesh (graph) object for GCN
        self.mesh = Mesh(self.options, self.options.num_downsampling)
        self.faces = torch.cat( self.options.batch_size * [
                                self.mesh.faces.to(self.device)[None]
                                ], dim = 0 )
        self.faces = self.faces.type(torch.LongTensor).to(self.device)
        
        # Create SMPL blending model and edges
        self.smpl = SMPL(self.options.smpl_model_path, self.device)
        # self.smplEdge = torch.Tensor(np.load(self.options.smpl_edges_path)) \
        #                 .long().to(self.device)
        
        # read SMPL .bj file to get uv coordinates
        _, self.smpl_tri_ind, uv_coord, tri_uv_ind = read_Obj(self.options.smpl_objfile_path)
        uv_coord[:, 1] = 1 - uv_coord[:, 1]
        expUV = uv_coord[tri_uv_ind.flatten()]
        unique, index = np.unique(self.smpl_tri_ind.flatten(), return_index = True)
        self.smpl_verts_uvs = torch.as_tensor(expUV[index,:]).float().to(self.device)
        self.smpl_tri_ind   = torch.as_tensor(self.smpl_tri_ind).to(self.device)
        
        # camera for projection 
        self.perspCam = perspCamera() 

        # mean and std of displacements
        self.dispPara = \
            torch.Tensor(np.load(self.options.MGN_dispMeanStd_path)).to(self.device)
        
        # load average pose and shape and convert to camera coodinate;
        # avg pose is decided by the image id we use for training (0-11) 
        avgPose_objCoord = np.load(self.options.MGN_avgPose_path)
        avgPose_objCoord[:3] = rotationMatrix_to_axisAngle(    # for 0,6, front only
            torch.tensor([[[1,  0, 0],
                           [0, -1, 0],
                           [0,  0,-1]]]))
        self.avgPose = \
            axisAngle_to_Rot6d(
                torch.Tensor(
                    avgPose_objCoord[None]\
                    .repeat(self.options.batch_size, axis = 0))
                    .reshape(-1, 3)).reshape(self.options.batch_size, -1)\
                .to(self.device)
        self.avgBeta = \
            torch.Tensor(np.load(self.options.MGN_avgBeta_path)[None]
                    .repeat(self.options.batch_size, axis = 0)).to(self.device)
        # 1.2755 is for our settings
        self.avgCam  = \
            torch.Tensor([1.2755, 0, 0])[None]\
                    .repeat_interleave(self.options.batch_size, dim = 0)\
                    .to(self.device)    # we know the rendered dataset
        
        self.model = frameVIBE(
            self.options.smpl_model_path, 
            self.mesh,                
            self.avgPose,
            self.avgBeta,
            self.avgCam,
            self.options.num_channels,
            self.options.num_layers ,  
            self.smpl_verts_uvs,
            self.smpl_tri_ind
            ).to(self.device)
            
        # Setup a optimizer for models
        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=self.options.lr,
            betas=(self.options.adam_beta1, 0.999),
            weight_decay=self.options.wd)
        
        self.criterion = VIBELoss(
            e_loss_weight=50,           # for kp 2d, help to estimate camera, global orientation
            e_3d_loss_weight=1,       # for kp 3d, bvt
            e_pose_loss_weight=10,     # for body pose parameters
            e_shape_loss_weight=1,   # for body shape parameters
            e_disp_loss_weight=1,   # for displacements 
            e_tex_loss_weight=1,       # for uv image 
            d_motion_loss_weight=0
            )
        
        # Pack models and optimizers in a dict - necessary for checkpointing
        self.models_dict = {self.options.model: self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}
        
    def convert_GT(self, input_batch):     
        
        smplGT = torch.cat([
            # cameraGT is not used, here is a placeholder
            torch.zeros([self.options.batch_size,3]).to(self.device),
            # the global rotation is incorrect as it is under the original 
            # coordinate system. The rest and betas are fine.
            rot6d_to_axisAngle(input_batch['smplGT']['pose']).reshape(-1, 72),   # 24 * 6 = 144
            input_batch['smplGT']['betas']],
            dim = 1).float()
        
        # vertices in the original coordinates
        vertices = self.smpl(
            pose = rot6d_to_axisAngle(input_batch['smplGT']['pose']).reshape(self.options.batch_size, 72),
            beta = input_batch['smplGT']['betas'].float(),
            trans = input_batch['smplGT']['trans']
            )
        
        # get joints in 3d and 2d in current camera coordiante
        joints_3d = self.smpl.get_joints(vertices.float())
        joints_2d, _, joints_3d= self.perspCam(
            fx = input_batch['cameraGT']['f_rot'][:,0,0], 
            fy = input_batch['cameraGT']['f_rot'][:,0,0], 
            cx = 112, 
            cy = 112, 
            rotation = rot6d_to_rotmat(input_batch['cameraGT']['f_rot'][:,0,1:]).float(),  
            translation = input_batch['cameraGT']['t'][:,None,:].float(), 
            points = joints_3d,
            visibilityOn = False,
            output3d = True
            )
        joints_3d = (joints_3d - input_batch['cameraGT']['t'][:,None,:]).float() # remove shifts
        
        # visind = 1
        # img = torch.zeros([224,224])
        # joints_2d[visind][joints_2d[visind].round() >= 224] = 223 
        # joints_2d[visind][joints_2d[visind].round() < 0] = 0 
        # img[joints_2d[visind][:,1].round().long(), joints_2d[visind][:,0].round().long()] = 1
        # plt.imshow(img)
        # plt.imshow(input_batch['img'][visind].cpu().permute(1,2,0))
        
        # convert to [-1, +1]
        joints_2d = (torch.cat(joints_2d, dim=0).reshape([self.options.batch_size, 24, 2]) - 112)/112
        joints_2d = joints_2d.float()
        
        # convert vertices to current camera coordiantes
        _,_,vertices = self.perspCam(
            fx = input_batch['cameraGT']['f_rot'][:,0,0], 
            fy = input_batch['cameraGT']['f_rot'][:,0,0], 
            cx = 112, 
            cy = 112, 
            rotation = rot6d_to_rotmat(input_batch['cameraGT']['f_rot'][:,0,1:]).float(),  
            translation = input_batch['cameraGT']['t'][:,None,:].float(), 
            points = vertices,
            visibilityOn = False,
            output3d = True
            )
        vertices = (vertices - input_batch['cameraGT']['t'][:,None,:]).float()
        
        # compute the displacements (mag * normal)
        dispGT = []
        for cnt in range(vertices.shape[0]):
            o3dBody = create_fullO3DMesh(vertices[cnt].cpu(), self.faces[cnt].cpu()) 
            normals = torch.as_tensor(o3dBody.compute_vertex_normals().vertex_normals).float().to(self.device)
            dispMag = input_batch['meshGT']['displacement'][cnt].norm(dim=1)
            dispGT.append( dispMag[:,None]*normals )
            
        # normalize the disp; divide by 10 is to limit the scale (outliers);
        dispGT = (torch.stack(dispGT, dim = 0)-self.dispPara[0])/self.dispPara[1]/10
        
        # prove the correctness of coord sys 
        # points = (dispGT*self.dispPara[1]*10+self.dispPara[0]) + vertices
        # localcam = perspCamera(smpl_obj = False)    # disable additional trans
        # img_points, _ = localcam(
        #     fx = input_batch['cameraGT']['f_rot'][:,0,0], 
        #     fy = input_batch['cameraGT']['f_rot'][:,0,0], 
        #     cx = 112, 
        #     cy = 112, 
        #     rotation = torch.eye(3)[None].repeat_interleave(2, dim = 0).to('cuda'),  
        #     translation = torch.zeros([2,1,3]).to('cuda'), 
        #     points = points,
        #     visibilityOn = False,
        #     output3d = False
        #     )
        # img = torch.zeros([224,224])
        # img_points[visind][img_points[visind].round() >= 224] = 223 
        # img_points[visind][img_points[visind].round() < 0] = 0 
        # img[img_points[visind][:,1].round().long(), img_points[visind][:,0].round().long()] = 1
        # plt.imshow(img)
        
        # self.renderer = renderer(batch_size = 1)
        # if input_batch['isAug'][0]:
        #     tex = input_batch['UVmapGT'].float()[0].flip(dims=(0,))
        # images = self.renderer(
        #     verts = vertices[0][None],
        #     faces = self.faces[0][None],
        #     verts_uvs = self.smpl_verts_uvs[None].repeat_interleave(2, dim=0)[0][None],
        #     faces_uvs = self.smpl_tri_ind[None].repeat_interleave(2, dim=0)[0][None],
        #     tex_image = tex[None],
        #     R = torch.eye(3)[None].repeat_interleave(2, dim = 0).to('cuda')[0][None],
        #     T = input_batch['cameraGT']['t'].float()[0][None],
        #     f = input_batch['cameraGT']['f_rot'][:,0,0][:,None].float()[0][None],
        #     C = torch.ones([2,2]).to('cuda')[0][None]*112,
        #     imgres = 224
        # )
        
        GT = {'img' : input_batch['img'].float(),
              'imgname' : input_batch['imgname'],
              'isAug': input_batch['isAug'],
              
              'text': input_batch['meshGT']['texture'].float(),    # verts tex
              'camera': input_batch['cameraGT'],                   # wcd 
              'indexMap': input_batch['indexMapGT'],
              'theta': smplGT.float(),        
              
              # joints_2d is col, row == x, y; joints_3d is x,y,z
              'target_2d': joints_2d.float(),
              'target_3d': joints_3d.float(),
              'target_bvt': vertices.float(),   # body vertices
              'target_dp': dispGT.float(),
              # 'target_dvt': (vertices + \
              #    (dispGT*self.dispPara[1]*10+self.dispPara[0])).float()    # vertices 
              
              'target_uv': input_batch['UVmapGT'].float()
              }
            
        return GT
    
    def train_step(self, input_batch):
        """Training step."""
        self.model.train()
        
        # prepare data
        GT = self.convert_GT(input_batch)
        
        # forward pass
        pred = self.model(GT['img'])
        
        # loss
        gen_loss, loss_dict = self.criterion(
            generator_outputs=pred,
            data_2d = GT['target_2d'],
            data_3d = GT,
            )
        # print(gen_loss)
        out_args = loss_dict
        out_args['loss'] = gen_loss
        out_args['prediction'] = pred 
        
        # save one training sample for vis
        if (self.step_count+1) % self.options.summary_steps == 0:
            with torch.no_grad():
                self.save_sample(GT, pred[0], saveTo = 'train')

        return out_args

    def test(self):
        """"Testing process"""
        self.model.eval()    
        
        test_loss, test_tex_loss = 0, 0
        test_loss_pose, test_loss_shape = 0, 0
        test_loss_kp_2d, test_loss_kp_3d = 0, 0
        test_loss_dsp_3d, test_loss_bvt_3d = 0, 0
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
            # prepare data
            GT = self.convert_GT(batch_toDEV)

            with torch.no_grad():    # disable grad
                # forward pass
                pred = self.model(GT['img'])
                
                # loss
                gen_loss, loss_dict = self.criterion(
                    generator_outputs=pred,
                    data_2d = GT['target_2d'],
                    data_3d = GT,
                    )
                # save for comparison
                if step == 0:
                    self.save_sample(GT, pred[0])
                
            test_loss += gen_loss
            test_loss_pose  += loss_dict['loss_pose']
            test_loss_shape += loss_dict['loss_shape']
            test_loss_kp_2d += loss_dict['loss_kp_2d']
            test_loss_kp_3d += loss_dict['loss_kp_3d']
            test_loss_bvt_3d += loss_dict['loss_bvt_3d']
            test_loss_dsp_3d += loss_dict['loss_dsp_3d']
            test_tex_loss += loss_dict['loss_tex']
                        
        test_loss = test_loss/len(self.test_data_loader)
        test_loss_pose  = test_loss_pose/len(self.test_data_loader)
        test_loss_shape = test_loss_shape/len(self.test_data_loader)
        test_loss_kp_2d = test_loss_kp_2d/len(self.test_data_loader)
        test_loss_kp_3d = test_loss_kp_3d/len(self.test_data_loader)
        test_loss_bvt_3d = test_loss_bvt_3d/len(self.test_data_loader)
        test_loss_dsp_3d = test_loss_dsp_3d/len(self.test_data_loader)
        test_tex_loss = test_tex_loss/len(self.test_data_loader)
        
        lossSummary = {'test_loss': test_loss, 
                       'test_loss_pose' : test_loss_pose,
                       'test_loss_shape' : test_loss_shape,
                       'test_loss_kp_2d' : test_loss_kp_2d,
                       'test_loss_kp_3d' : test_loss_kp_3d,
                       'test_loss_bvt_3d': test_loss_bvt_3d,
                       'test_loss_dsp_3d': test_loss_dsp_3d,
                       'test_tex_loss': test_tex_loss
                       }
        self.test_summaries(lossSummary)
        
        if test_loss < self.last_test_loss:
            self.last_test_loss = test_loss
            self.saver.save_checkpoint(self.models_dict, 
                                       self.optimizers_dict, 
                                       0, 
                                       step+1, 
                                       self.options.batch_size, 
                                       self.test_data_loader.sampler.dataset_perm, 
                                       self.step_count) 
            tqdm.write('Better test checkpoint saved')
        
    def save_sample(self, data, prediction, ind = 0, saveTo = 'test'):
        """Saving a sample for visualization and comparison"""
        
        assert saveTo in ('test', 'train'), 'save to train or test folder'
        
        folder = pjn(self.options.summary_dir, '%s/'%(saveTo))
        _input = pjn(folder, 'input_image.png')
        # batchs = self.options.batch_size
        
        # save the input image, if not saved
        if not isfile(_input):
            plt.imsave(_input, data['img'][ind].cpu().permute(1,2,0).clamp(0,1).numpy())
            
        # overlap the prediction to the real image
        save_renderer = simple_renderer(batch_size = 1)
        predTrans = torch.stack(
            [prediction['theta'][ind, 1],
             prediction['theta'][ind, 2],
             2 * 1000. / (224. * prediction['theta'][ind, 0] + 1e-9)], dim=-1)
        tex = (prediction['tex_image'][ind][None], 
               prediction['tex_image'][ind].flip(dims=(0,))[None])\
              [data['isAug'][ind]]
        pred_img = save_renderer(
            verts = prediction['verts'][ind][None],
            faces = self.faces[ind][None],
            verts_uvs = self.smpl_verts_uvs[None],
            faces_uvs = self.smpl_tri_ind[None],
            tex_image = tex,
            R = torch.eye(3)[None].to('cuda'),
            T = predTrans,
            f = torch.ones([1,1]).to('cuda')*1000,
            C = torch.ones([1,2]).to('cuda')*112,
            imgres = 224)
        overlayimg = 0.9*pred_img[0,:,:,:3] + 0.1*data['img'][ind].permute(1,2,0)
        save_path = pjn(folder,'overlay_%s_iters%d.png'%(data['imgname'][ind].split('/')[-1][:-4], self.step_count))
        plt.imsave(save_path, (overlayimg.clamp(0, 1)*255).cpu().numpy().astype('uint8'))
        save_path = pjn(folder,'tex_%s_iters%d.png'%(data['imgname'][ind].split('/')[-1][:-4], self.step_count))
        plt.imsave(save_path, (pred_img[0,:,:,:3].clamp(0, 1)*255).cpu().numpy().astype('uint8'))
        save_path = pjn(folder,'unwarptex_%s_iters%d.png'%(data['imgname'][ind].split('/')[-1][:-4], self.step_count))
        plt.imsave(save_path, (prediction['unwarp_tex'][ind].clamp(0, 1)*255).cpu().numpy().astype('uint8'))
        
        
        # save body pose if available
        bodyList = {}
        
        # create GT undressed body vetirces. Do not use orignal beta and rot
        # since the global orientation is incorrect
        bodyList['bodyGT'] = data['target_bvt'].cpu()
        
        # create predicted undressed body vetirces
        bodyList['bodyPred'] = self.smpl(
            prediction['theta'][ind][3:75][None],
            prediction['theta'][ind][75:][None]).cpu()
        
        # consider cloth
        target_dp = (data['target_dp'][ind]*self.dispPara[1]*10+self.dispPara[0]).cpu()[None,:] 
        predict_dp = (prediction['verts_disp'][ind]*self.dispPara[1]*10+self.dispPara[0]).cpu()[None,:] 
        
        bodyList['bodyGT_GTCloth'] = bodyList['bodyGT'] + target_dp  
        bodyList['bodyGT_PredCloth'] = bodyList['bodyGT'] + predict_dp         
        bodyList['bodyPred_PredCloth']= bodyList['bodyPred'] + predict_dp
                            
        # Create meshes and save them
        for key, val in bodyList.items():
            Meshbody = create_fullO3DMesh(val[0], self.faces.cpu()[0])    
            savepath = pjn(folder,'%s_iters%d__%s.obj'%\
                    (data['imgname'][ind].split('/')[-1][:-4], self.step_count, key))
            o3d.io.write_triangle_mesh(savepath, Meshbody)


if __name__ == '__main__':
    
    # read preparation configurations
    cfgs = TrainOptions()
    cfgs.parse_args()
    
    # confirm general settings
    for arg in vars(cfgs.args):
        print('%-25s:'%(arg), getattr(cfgs.args, arg)) 
    msg = 'Do you confirm that the settings are correct?'
    assert input("%s (Y/N) " % msg).lower() == 'y', 'Settings are not comfirmed.'
    
    # # confirm structure settings
    # import json
    # with open( pjn(cfgs.args.log_dir, 'structures.json'), 'w') as file:
    #   json.dump(structure_options, file, indent=4)
    # for key, val in structure_options.items():
    #     if isinstance(val, dict):
    #         print('%-25s:'%key)
    #         for ikey, ival in val.items():
    #             if isinstance(ival, dict):
    #                 print('\t%-21s:'%ikey)
    #                 for jkey, jval in ival.items():
    #                     print('\t\t%-17s:'%(jkey), jval) 
    #             else:
    #                 print('\t%-21s:'%(ikey), ival) 
    #     else:   
    #         print('%-25s:'%(key), val) 
    # msg = 'Do you confirm that the structures are correct?'
    # assert input("%s (Y/N) " % msg).lower() == 'y', 'Settings are not comfirmed.'

    mgn_trainer = trainer(cfgs.args)
    mgn_trainer.train()
    