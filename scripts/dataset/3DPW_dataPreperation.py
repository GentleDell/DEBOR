#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 12:42:18 2020

@author: zhantao
"""

import os
import sys
sys.path.append( "../" )
import argparse
from glob import glob
from os.path import join as pjn
import pickle as pkl

import yaml
import numpy as np
import chumpy as ch
import open3d as o3d
import matplotlib.pyplot as plt
from smpl.smpl_webuser.serialization import load_model
from smpl.smpl_webuser.lbs import verts_core as _verts_core
from utils.mesh_util import generateSMPLmesh, create_fullO3DMesh
from utils.vis_util import read_Obj


# def boundingbox(cameraIntrinsic: np.array, cameraExtrinsic: np.array):
    
#     # read vertices and create camera class
#     vertices, _, _, _ = read_Obj(meshfile)
#     cameraCLS = camera(projModel="perspective", intrinsics=cameraIntrinsic)
#     cameraCLS.setExtrinsic(rotation = cameraExtrinsic[0], 
#                            location = cameraExtrinsic[1])
    
#     # convert point cloud to homogenous coordinate system and project to image.
#     vertices = np.hstack( (vertices, np.ones([vertices.shape[0], 1])) ).transpose()
#     pixels = cameraCLS.projection(vertices).astype('int')
    
#     # prepare to visualize the projection
#     visImage = np.zeros( [int(cameraCLS.resolution_y), int(cameraCLS.resolution_x)] )
#     visImage[pixels[1,:], pixels[0,:]] = 1
#     plt.imsave( pjn( imagefolder, 'camera%d_light0_projection.png'%(cameraIdx) ), visImage )
    
#     # bounding box of the body
#     max_col = max(min(pixels[0,:].max() + marginSize, cameraCLS.resolution_x), 0)
#     min_col = min(max(pixels[0,:].min() - marginSize, 0), cameraCLS.resolution_x) 
#     max_row = max(min(pixels[1,:].max() + marginSize, cameraCLS.resolution_y), 0)
#     min_row = min(max(pixels[1,:].min() - marginSize, 0), cameraCLS.resolution_y)
    
#     with open(pjn( imagefolder, 'camera%d'%cameraIdx+'_boundingbox.txt'), "w") as output:
#         output.write(str([min_row, min_col, max_row, max_col]))
    

if __name__ == "__main__":
    '''
    It iterates over all squences to compute the boundingboxes for all images
    and compute displacements(unposed). Then, it will project the vertice to 
    the image plane to verify the cameras if required.
    '''
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_preparation_cfg', action='store', type=str, 
                        help='Path to the configuration for rendering.', 
                        default = './dataset_preparation_cfg.yaml')
    args = parser.parse_args()
    
    # read render configurations
    with open(args.dataset_preparation_cfg) as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)
    
    for sequence in glob( pjn(cfgs['dataroot3DPW'], 'imageFiles/*') ):
        
        sequenceName = sequence.split('/')[-1]
        pathToSeqSet = pjn(cfgs['dataroot3DPW'], 'sequenceFiles/' + sequenceName+'.pkl')        
       
        sequenceSet  = pkl.load(open(pathToSeqSet, 'rb'),  encoding='iso-8859-1')
        numActoress  = len(sequenceSet['genders'])
        numImgFrames = sequenceSet['img_frame_ids'].shape[0]
        cameraIntric = sequenceSet['cam_intrinsics']
        
        displacement = {}
        boundingbox  = {}
        cameraParams = {}
        for bodyInd in range(numActoress):
            if sequenceSet['genders'][bodyInd] == 'm':
                SMPLmodel = load_model(cfgs['smplModel_male'])
            else:
                SMPLmodel = load_model(cfgs['smplModel_female'])
            
            SMPLmodel.betas[:10] = sequenceSet['betas'][bodyInd][:10]
            unposedVerts, _  = _verts_core(
                SMPLmodel[bodyInd].pose, SMPLmodel[bodyInd].v_posed, SMPLmodel[bodyInd].J, 
                SMPLmodel[bodyInd].weights,SMPLmodel[bodyInd].kintree_table, want_Jtr=True, xp=ch)
        
            # debug
            # smplObj = pjn('/'.join(cfgs['smplModel'].split('/')[:-1]), 'text_uv_coor_smpl.obj')
            # _, smplMesh, smpl_text_uv_coord, smpl_text_uv_mesh = read_Obj(smplObj)
            # o3dMesh = create_fullO3DMesh(unposedVerts, smplMesh)
            
            # o3d.visualization.draw_geometries([o3dMesh])
            displacement['actor%d'%(bodyInd)] = sequenceSet['v_template_clothed'] - unposedVerts
            
            for imagePath in sorted( glob(pjn(sequence, '*.jpg')) ):
                
                imageIdx = int(imagePath.split('/')[-1].split('.')[0][-5:])
                
                cameraEx = sequenceSet['cam_poses'][imageIdx]

                SMPLmodel[bodyInd].pose[:] = sequenceSet['poses'][bodyInd][imageIdx]
                SMPLmodel[bodyInd].trans[:] = sequenceSet['trans'][bodyInd][imageIdx]
                [v,Jtr] = _verts_core(
                    SMPLmodel[bodyInd].pose, SMPLmodel[bodyInd].v_posed, SMPLmodel[bodyInd].J, 
                    SMPLmodel[bodyInd].weights,SMPLmodel[bodyInd].kintree_table, want_Jtr=True, xp=ch)
                
                                
            
        displacement = np.array(displacement)
        boundingbox  = np.array(boundingbox)
        
        print('Saving its displacements')
        with open( pjn(sequence, 'displacement_before_posing.npy'), 'wb' ) as f:
            np.save(f, displacement)
        
        print('Saving boundingboxes')
        with open( pjn(sequence, 'boundingBoxes.npy'), 'wb' ) as f:
            np.save(f, boundingbox)