#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 10:07:35 2021

@author: zhantao
"""

from os.path import abspath, join as pjn
import sys
if abspath('./') not in sys.path:
    sys.path.append(abspath('./'))
if abspath('./third_party/smpl_webuser') not in sys.path:
    sys.path.append(abspath('./third_party/smpl_webuser'))
from collections import namedtuple

import json
import torch
from tqdm import tqdm
import numpy as np

from utils.vis_util import read_Obj
from utils.mesh_util import create_fullO3DMesh
from utils import Mesh, CheckpointSaver, CheckpointDataLoader
from dataset import MGNDataset
from models import camera as perspCamera, SMPL, frameVIBE
from models.geometric_layers import axisAngle_to_Rot6d, rot6d_to_axisAngle
from models.geometric_layers import rotationMatrix_to_axisAngle, rot6d_to_rotmat


def convert_GT(options, input_batch, smpl, faces, perspCam, dispPara, device):     
    
    smplGT = torch.cat([
        torch.zeros([options.batch_size,3]).to(device),
        rot6d_to_axisAngle(input_batch['smplGT']['pose']).reshape(-1, 72), 
        input_batch['smplGT']['betas']],
        dim = 1).float()
    
    # vertices in the original coordinates
    vertices = smpl(
        pose = rot6d_to_axisAngle(input_batch['smplGT']['pose']).reshape(options.batch_size, 72),
        beta = input_batch['smplGT']['betas'].float(),
        trans = input_batch['smplGT']['trans']
        )
    
    # get joints in 3d and 2d in current camera coordiante
    joints_3d = smpl.get_joints(vertices.float())
    joints_2d, _, joints_3d= perspCam(
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
    joints_3d = (joints_3d - input_batch['cameraGT']['t'][:,None,:]).float() 
    
    # convert to [-1, +1]
    joints_2d = (torch.cat(joints_2d, dim=0).reshape([options.batch_size, 24, 2]) - 112)/112
    joints_2d = joints_2d.float()
    
    # convert vertices to current camera coordiantes
    _,_,vertices = perspCam(
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
        o3dBody = create_fullO3DMesh(vertices[cnt].cpu(), faces[cnt].cpu()) 
        normals = torch.as_tensor(o3dBody.compute_vertex_normals().vertex_normals).float().to(device)
        dispMag = input_batch['meshGT']['displacement'][cnt].norm(dim=1)
        dispGT.append( dispMag[:,None]*normals )
        
    # normalize the disp; divide by 10 is to limit the scale (outliers);
    dispGT = (torch.stack(dispGT, dim = 0)-dispPara[0])/dispPara[1]/10
        
    GT = {'img' : input_batch['img'].float(),
          'imgname' : input_batch['imgname'],
          'isAug': input_batch['isAug'],
          'text': input_batch['meshGT']['texture'].float(),    # verts tex
          'camera': input_batch['cameraGT'],                   # wcd 
          'indexMap': input_batch['indexMapGT'],
          'theta': smplGT.float(),        
          'target_2d': joints_2d.float(),
          'target_3d': joints_3d.float(),
          'target_bvt': vertices.float(),   # body vertices
          'target_dp': dispGT.float(),
          'target_uv': input_batch['UVmapGT'].float()
          }
        
    return GT  


def evaluation_structure(pathCkp: str):
    
    # Load configuration
    with open( pjn(pathCkp, 'config.json') , 'r' ) as f:
        options = json.load(f)
        options = namedtuple('options', options.keys())(**options)
    
    # prepare evaluation dataset
    eva_ds  = MGNDataset(options, split = 'validation')
    eva_data_loader = CheckpointDataLoader( 
                eva_ds,
                batch_size  = options.batch_size,
                num_workers = options.num_workers,
                pin_memory  = options.pin_memory,
                shuffle     = False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mesh = Mesh(options, options.num_downsampling)
    faces = torch.cat( options.batch_size * [
                        mesh.faces.to(device)[None]
                        ], dim = 0 )
    faces = faces.type(torch.LongTensor).to(device)
        
    # read SMPL .bj file to get uv coordinates
    _, smpl_tri_ind, uv_coord, tri_uv_ind = read_Obj(options.smpl_objfile_path)
    uv_coord[:, 1] = 1 - uv_coord[:, 1]
    expUV = uv_coord[tri_uv_ind.flatten()]
    unique, index = np.unique(smpl_tri_ind.flatten(), return_index = True)
    smpl_verts_uvs = torch.as_tensor(expUV[index,:]).float().to(device)
    smpl_tri_ind   = torch.as_tensor(smpl_tri_ind).to(device)
    
    smpl = SMPL(options.smpl_model_path, device)
    
    # load average pose and shape and convert to camera coodinate;
    # avg pose is decided by the image id we use for training (0-11) 
    avgPose_objCoord = np.load(options.MGN_avgPose_path)
    avgPose_objCoord[:3] = rotationMatrix_to_axisAngle(    # for 0,6, front only
        torch.tensor([[[1,  0, 0],
                       [0, -1, 0],
                       [0,  0,-1]]]))
    avgPose = \
        axisAngle_to_Rot6d(
            torch.Tensor(
                avgPose_objCoord[None]\
                .repeat(options.batch_size, axis = 0))
                .reshape(-1, 3)).reshape(options.batch_size, -1)\
            .to(device)
    avgBeta = \
        torch.Tensor(np.load(options.MGN_avgBeta_path)[None]
                .repeat(options.batch_size, axis = 0)).to(device)
    # 1.2755 is for our settings
    avgCam  = \
        torch.Tensor([1.2755, 0, 0])[None]\
                .repeat_interleave(options.batch_size, dim = 0)\
                .to(device)    # we know the rendered dataset
    # displacement mean and std     
    dispPara = torch.Tensor(np.load(options.MGN_dispMeanStd_path)).to(device)
    
    perspCam = perspCamera() 
    
    # Create model
    model = frameVIBE(
            options.smpl_model_path, 
            mesh,
            avgPose,
            avgBeta,
            avgCam,
            options.num_channels,
            options.num_layers,
            smpl_verts_uvs,
            smpl_tri_ind
            ).to(device)
    
    optimizer = torch.optim.Adam(params=list(model.parameters()))
    models_dict = {options.model: model}
    optimizers_dict = {'optimizer': optimizer}
    
    # Load pretrained model
    saver = CheckpointSaver(save_dir=options.checkpoint_dir)
    saver.load_checkpoint(models_dict, optimizers_dict, checkpoint_file=options.checkpoint)
    
    textureMSE = torch.nn.MSELoss()
    displaceMSE = torch.nn.MSELoss()
    poseMSE = torch.nn.MSELoss()
    shapeMSE = torch.nn.MSELoss()
    with torch.no_grad():
        textErr, dispErr, poseErr, shapeErr = 0, 0, 0, 0
        for step, batch in enumerate(tqdm(eva_data_loader, desc='Epoch0',
                                                  total=len(eva_ds) // options.batch_size,
                                                  initial=eva_data_loader.checkpoint_batch_idx),
                                          eva_data_loader.checkpoint_batch_idx):
            # convert data devices
            batch_toDEV = {}
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch_toDEV[key] = val.to(device)
                else:
                    batch_toDEV[key] = val
                    
                if isinstance(val, dict):
                    batch_toDEV[key] = {}
                    for k, v in val.items():
                        if isinstance(v, torch.Tensor):
                            batch_toDEV[key][k] = v.to(device)
            
            GT = convert_GT(options, batch_toDEV, smpl, faces, perspCam, dispPara, device)
            pred = model(GT['img'])
            
            textErr += textureMSE(pred[0]['tex_image'], GT['target_uv']).sqrt()
            dispErr += displaceMSE(pred[0]['verts_disp'], GT['target_dp']).sqrt()
            poseErr += poseMSE(pred[0]['theta'][:,6:75], GT['theta'][:,6:75]).sqrt()
            shapeErr+= shapeMSE(pred[0]['theta'][:,75:], GT['theta'][:,75:]).sqrt()            
            
    textErr = textErr/len(eva_ds)
    dispErr = dispErr/len(eva_ds)
    poseErr = poseErr/len(eva_ds)
    shapeErr= shapeErr/len(eva_ds)
    
    return {'texture RMSE' : textErr,
            'clothes RMSE' : dispErr,
            'pose RMSE' : poseErr,
            'shape RMSE': shapeErr}


if __name__ == '__main__':
    
    print('be aware of the input to the model, some need img_orig.')
    
    path_to_chkpt = '../logs/local/structure_ver1_all'
    
    evaErrs = evaluation_structure(path_to_chkpt)
    
    for key, val in evaErrs.items():
        print('%-25s:'%(key), val.item())
    