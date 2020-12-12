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
    sys.path.append(abspath('./third_party/smpl_webuser'))
    
import json
import torch
import cv2
import numpy as np
from numpy import array
import open3d as o3d
import matplotlib.pyplot as plt
from torchvision.transforms import Normalize

from utils.imutils import crop, background_replacing
from utils import Mesh, CheckpointSaver
from utils.mesh_util import generateSMPLmesh, create_fullO3DMesh
from models import GraphCNN, res50_plus_Dec, UNet
from utils.vis_util import read_Obj


def visPrediction(path_object: str, path_SMPLmodel: str, displacement: array,
                  chooseUpsample: bool = True):
    # upsampled mesh
    pathtext = pjn(path_object, 'GroundTruth/vertex_colors_oversample_%s.npy'%
                       (['OFF', 'ON'][chooseUpsample]))
    pathsegm = pjn(path_object, 'GroundTruth/segmentations_oversample_%s.npy'%
                       (['OFF', 'ON'][chooseUpsample]))
    pathobje = pjn('/'.join(path_SMPLmodel.split('/')[:-1]), 'text_uv_coor_smpl.obj')
    
    # SMPL parameters
    pathRegistr  = pjn(path_object, 'registration.pkl')
    
    # load SMPL parameters and model; transform vertices and joints.
    registration = pickle.load(open(pathRegistr, 'rb'),  encoding='iso-8859-1')
    SMPLvert_posed, joints = generateSMPLmesh(
            path_SMPLmodel, registration['pose'], registration['betas'], 
            registration['trans'], asTensor=True)
    
    # load SMPL obj file
    smplVert, smplMesh, smpl_text_uv_coord, smpl_text_uv_mesh = read_Obj(pathobje)
    smpl_text_uv_coord[:, 1] = 1 - smpl_text_uv_coord[:, 1]    # SMPL .obj need this conversion
    
    # upsample the orignal mesh if choose upsampled GT
    if chooseUpsample:
        o3dMesh = o3d.geometry.TriangleMesh()
        o3dMesh.vertices = o3d.utility.Vector3dVector(SMPLvert_posed)
        o3dMesh.triangles= o3d.utility.Vector3iVector(smplMesh) 
        o3dMesh = o3dMesh.subdivide_midpoint(number_of_iterations=1)
        
        SMPLvert_posed = np.asarray(o3dMesh.vertices)
        smplMesh = np.asarray(o3dMesh.triangles)
    
    # create meshes
    if isfile(pathtext):
        vertexcolors = np.load(pathtext)
        segmentation = np.load(pathsegm)
        
        growVertices = SMPLvert_posed + displacement
        clrBody = create_fullO3DMesh(growVertices, smplMesh, 
                                     vertexcolor=vertexcolors/vertexcolors.max(), 
                                     use_vertex_color=True)
        
        SMPLvert_posed += np.array([0.6,0,0])
        segBody = create_fullO3DMesh(SMPLvert_posed, smplMesh, 
                                     vertexcolor=vertexcolors/vertexcolors.max(), 
                                     use_vertex_color=True)    

        o3d.visualization.draw_geometries([clrBody, segBody])


def inference(pathCkp: str, pathImg: str = None, pathBgImg: str = None):

    # Load configuration
    with open( pjn(pathCkp, 'config.json') , 'r' ) as f:
        options = json.load(f)
        options = namedtuple('options', options.keys())(**options)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mesh = Mesh(options, options.num_downsampling)
    
    # Create model
    if options.model == 'graphcnn':
        model = GraphCNN(
            mesh.adjmat,
            mesh.ref_vertices.t(),
            num_channels=options.num_channels,
            num_layers=options.num_layers
            ).to(device)
    elif options.model == 'sizernn':
        model = res50_plus_Dec(
            options.latent_size,
            mesh.ref_vertices.shape[0] * 3,    # only consider displacements currently
            ).to(device)
    elif options.model == 'unet':
        model = UNet(
            input_shape = (options.img_res, options.img_res, 3), 
            output_dims = 3                    # only consider displacements currently
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
    img  = crop(img, center, scale, [224, 224], rot=0)
    
    # img  = torch.Tensor(img).permute(2,0,1)/255
    img_in = normalize_img( torch.Tensor(img).permute(2,0,1)/255 )
    
    # Inference
    model.eval()    
    pose = None
    prediction = model(img_in[None,:,:,:].to(device), pose)

    return prediction, img_in
    

if __name__ == '__main__':
    
    path_to_SMPL  = '../body_model/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl' 
    path_to_chkpt = '../logs/GCMR_single_brighter_green'
    path_to_object= '/home/zhantao/Documents/masterProject/DEBOR/datasets/MGN_brighter/125611515860795/'
    path_to_image = pjn(path_to_object, 'rendering/camera0_light0_smpl_registered.png')
                    
    prediction, img_in = inference(path_to_chkpt, path_to_image)
    prediction = prediction[0].detach().cpu()
    
    if prediction.shape[1] == 3:
        visPrediction(path_to_object, path_to_SMPL, prediction, False)
        plt.imshow(img_in.permute(1,2,0))
    else:
        plt.imshow(prediction)
        plt.show()
        
        plt.imshow(img_in.permute(1,2,0))
        plt.show()
        