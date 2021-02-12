#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 11:05:05 2020

@author: zhantao
"""
import pickle
from glob import glob
from ast import literal_eval
from os.path import join as pjn, isfile, abspath
from collections import namedtuple
import sys
if abspath('./') not in sys.path:
    sys.path.append(abspath('./'))
if abspath('./third_party/smpl_webuser') not in sys.path:
    sys.path.append(abspath('./third_party/smpl_webuser'))
if abspath('./dataset') not in sys.path:
    sys.path.append(abspath('./dataset'))
if abspath('./dataset/MGN_helper') not in sys.path:
    sys.path.append(abspath('./dataset/MGN_helper'))
    
import json
import torch
import cv2
import numpy as np
from numpy import array
import open3d as o3d
import matplotlib.pyplot as plt
from torchvision.transforms import Normalize
from psbody.mesh import Mesh as psMesh, MeshViewers

from MGN_helper.lib.ch_smpl import Smpl
from utils import Mesh, CheckpointSaver
from utils.imutils import crop, background_replacing
from utils.mesh_util import generateSMPLmesh, create_fullO3DMesh, create_smplD_psbody
from models import frameVIBE, SMPL, simple_renderer
from models.geometric_layers import rotationMatrix_to_axisAngle, axisAngle_to_Rot6d
from utils.vis_util import read_Obj


def visbodyPrediction(img_in, prediction, options, path_object,
                      device = 'cuda', ind = 0):
    
    prediction = prediction[0]
    
    # <==== vis predicted body and displacements in 3D
    # displacements mean and std
    dispPara = np.load(options.MGN_offsMeanStd_path)
        
    # SMPLD model
    smplD = Smpl( options.smpl_model_path ) 
    
    # gt body and offsets
    gt_offsets_t = np.load(pjn(path_object, 'gt_offsets/offsets_std.npy'))
    pathRegistr  = pjn(path_object, 'registration.pkl')
    registration = pickle.load(open(pathRegistr, 'rb'),  encoding='iso-8859-1')
    gtBody_p = create_smplD_psbody(
        smplD, 
        gt_offsets_t, 
        registration['pose'], 
        registration['betas'], 
        0,
        rtnMesh = True)[1]
    
    # naked posed body
    nakedBody_p = create_smplD_psbody(
        smplD, 0, 
        prediction['theta'][ind][3:75][None].cpu(), 
        prediction['theta'][ind][75:][None].cpu(), 
        0, 
        rtnMesh=True)[1]
    
    # offsets in t-pose
    displacements = prediction['verts_disp'].cpu().numpy()[ind]
    offPred_t  = (displacements*dispPara[1]+dispPara[0])
    
    # create predicted dressed body       
    dressbody_p = create_smplD_psbody(
        smplD, offPred_t, 
        prediction['theta'][ind][3:75].cpu().numpy(), 
        prediction['theta'][ind][75:].cpu().numpy(), 
        0, 
        rtnMesh=True)[1]
    
    mvs = MeshViewers((1, 3))
    mvs[0][0].set_static_meshes([gtBody_p])
    mvs[0][1].set_static_meshes([nakedBody_p])
    mvs[0][2].set_static_meshes([dressbody_p])    
    
    offset_p = torch.tensor(dressbody_p.v - nakedBody_p.v).to(device).float()
    
    # <==== vis the overall prediction, i.e. render the image
    
    dispPara = torch.tensor(dispPara).to(device)
    
    # smpl Mesh
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
    
    # vis texture
    vis_renderer = simple_renderer(batch_size = 1)
    predTrans = torch.stack(
        [prediction['theta'][ind, 1],
         prediction['theta'][ind, 2],
         2 * 1000. / (224. * prediction['theta'][ind, 0] + 1e-9)], dim=-1)
    tex = prediction['tex_image'][ind].flip(dims=(0,))[None]
    pred_img = vis_renderer(
        verts = prediction['verts'][ind][None]+offset_p,
        faces = faces[ind][None],
        verts_uvs = smpl_verts_uvs[None],
        faces_uvs = smpl_tri_ind[None],
        tex_image = tex,
        R = torch.eye(3)[None].to('cuda'),
        T = predTrans,
        f = torch.ones([1,1]).to('cuda')*1000,
        C = torch.ones([1,2]).to('cuda')*112,
        imgres = 224).cpu()
    overlayimg = 0.9*pred_img[0,:,:,:3] + 0.1*img_in.permute(1,2,0)

    if 'tex_image' in prediction.keys():
        plt.figure()
        plt.imshow(prediction['tex_image'][ind].cpu())
        plt.figure()
        plt.imshow(prediction['unwarp_tex'][ind].cpu())
        plt.figure()
        plt.imshow(pred_img[ind].cpu())
        plt.figure()
        plt.imshow(overlayimg.cpu())
        plt.figure()
        plt.imshow(img_in.cpu().permute(1,2,0))
            
def inference_structure(pathCkp: str, pathImg: str = None, 
                        pathBgImg: str = None):
    
    print('If trained locally and renamed the workspace, do not for get to '
          'change the "checkpoint_dir" in config.json. ')
    
    # Load configuration
    with open( pjn(pathCkp, 'config.json') , 'r' ) as f:
        options = json.load(f)
        options = namedtuple('options', options.keys())(**options)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mesh = Mesh(options, options.num_downsampling)
    
    # read SMPL .bj file to get uv coordinates
    _, smpl_tri_ind, uv_coord, tri_uv_ind = read_Obj(options.smpl_objfile_path)
    uv_coord[:, 1] = 1 - uv_coord[:, 1]
    expUV = uv_coord[tri_uv_ind.flatten()]
    unique, index = np.unique(smpl_tri_ind.flatten(), return_index = True)
    smpl_verts_uvs = torch.as_tensor(expUV[index,:]).float().to(device)
    smpl_tri_ind   = torch.as_tensor(smpl_tri_ind).to(device)
    
    # load average pose and shape and convert to camera coodinate;
    # avg pose is decided by the image id we use for training (0-11) 
    avgPose_objCoord = np.load(options.MGN_avgPose_path)
    avgPose_objCoord[:3] = rotationMatrix_to_axisAngle(    # for 0,6, front only
        torch.tensor([[[1,  0, 0],
                       [0, -1, 0],
                       [0,  0,-1]]]))   
    avgPose = \
        axisAngle_to_Rot6d(
            torch.Tensor(avgPose_objCoord[None]).reshape(-1, 3)
            ).reshape(1, -1).to(device)
    avgBeta = \
        torch.Tensor(
            np.load(options.MGN_avgBeta_path)[None]).to(device)
    avgCam  = torch.Tensor([1.2755, 0, 0])[None].to(device)    # 1.2755 is for our settings
    
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

    # Prepare and preprocess input image
    pathToObj = '/'.join(pathImg.split('/')[:-2])
    cameraIdx = int(pathImg.split('/')[-1].split('_')[0][6:])
    with open( pjn( pathToObj,'rendering/camera%d_boundingbox.txt'%(cameraIdx)) ) as f:
        boundbox = literal_eval(f.readline())
    IMG_NORM_MEAN = [0.485, 0.456, 0.406]
    IMG_NORM_STD = [0.229, 0.224, 0.225]
    normalize_img = Normalize(mean=IMG_NORM_MEAN, std=IMG_NORM_STD)  
    
    path_to_rendering = '/'.join(pathImg.split('/')[:-1])
    cameraPath, lightPath = pathImg.split('/')[-1].split('_')[:2]
    cameraIdx, _ = int(cameraPath[6:]), int(lightPath[5:])
    with open( pjn( path_to_rendering,'camera%d_boundingbox.txt'%(cameraIdx)) ) as f:
        boundbox = literal_eval(f.readline())
    img  = cv2.imread(pathImg)[:,:,::-1].astype(np.float32)
    
    # prepare background
    if options.replace_background:
        if pathBgImg is None:
            bgimages = []
            for subfolder in sorted(glob( pjn(options.bgimg_dir, 'images/validation/*') )):
                for subsubfolder in sorted( glob( pjn(subfolder, '*') ) ):
                    if 'room' in subsubfolder:
                        bgimages += sorted(glob( pjn(subsubfolder, '*.jpg')) )
            bgimg =  cv2.imread(bgimages[np.random.randint(0, len(bgimages))])[:,:,::-1].astype(np.float32)
        else:
            bgimg = cv2.imread(pathBgImg)[:,:,::-1].astype(np.float32)
        img = background_replacing(img, bgimg)
    
    # augment image
    center =  [(boundbox[0]+boundbox[2])/2, (boundbox[1]+boundbox[3])/2] 
    scale  =  max((boundbox[2]-boundbox[0])/200, (boundbox[3]-boundbox[1])/200)
    img    = torch.Tensor(crop(img, center, scale, [224, 224], rot=0)).permute(2,0,1)/255
    img_in = normalize_img(img)
    
    # Inference
    with torch.no_grad():    # disable grad
        model.eval()    
        prediction = model(img_in[None].repeat_interleave(options.batch_size, dim = 0).to(device),
                           img[None].repeat_interleave(options.batch_size, dim = 0).to(device))

    return prediction, img_in, options

if __name__ == '__main__':
    
    path_to_SMPL  = '../body_model/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl' 
    path_to_chkpt = '../logs/local/structure_ver1_full_doubleEnc_newdataset_8_ver1.3'
    path_to_object= '../datasets/Multi-Garment_dataset/125611487366942/'
    path_to_image = pjn(path_to_object, 'rendering/camera0_light0_smpl_registered.png')

    prediction, img_in, options = inference_structure(path_to_chkpt, path_to_image)
    visbodyPrediction(img_in, prediction, options, path_to_object)

    