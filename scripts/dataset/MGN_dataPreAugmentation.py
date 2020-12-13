#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 20:17:32 2020

@author: zhantao
"""
import pickle
from glob import glob
import os
from os.path import join as pjn
from shutil import copy2
import sys 
sys.path.append( "../" )

import torch
import numpy as np
import open3d as o3d

from models import SMPL
from utils.vis_util import read_Obj
from utils.mesh_util import generateSMPLmesh, create_fullO3DMesh


class MGN_subjectPreAugmentation(object):
    
    def __init__(self, path_MGN: str, pathSMPL: str):
        
        self.path_smpl = pathSMPL
        self.path_subjects = sorted( glob(pjn(path_MGN, '*')) )
        
        self.betas, self.poses, self.text, self.disp = [], [], [], []
        for subjectPath in self.path_subjects:
            self.text.append( np.load(pjn(subjectPath, 
                                      'GroundTruth/vertex_colors_oversample_OFF.npy') ))
            self.disp.append( np.load(pjn(subjectPath, 
                                      'GroundTruth/normal_guided_displacements_oversample_OFF.npy')))
            
            pathRegistr  = pjn(subjectPath, 'registration.pkl')
            registration = pickle.load(open(pathRegistr, 'rb'),  encoding='iso-8859-1')
            self.betas.append( registration['betas'] )
            self.poses.append( registration['pose'] )
    
    def diversity(self, dataList: list, keepN: int = 15, 
                  threshold: int= 2.5) -> list:
        
        data = torch.Tensor(np.asarray(dataList))
        
        distance = (data[:,None,:] - data[None,:,:]).norm(dim=2)
        pair, _  = torch.where(distance == distance.max())
        
        independ, indices = [], []
        indices.append(pair[0])
        indices.append(pair[1])
        independ.append(data[pair[0],:][None,:])
        independ.append(data[pair[1],:][None,:])
        
        mask = torch.ones(data.shape[0])
        mask[pair[0]] = 0
        mask[pair[1]] = 0
        
        for cnt in range(keepN-2):
            
            distance = (data[:,None,:] - torch.cat(independ, dim = 0)[None,:,:]).norm(dim=2).sum(dim=1)*mask
            _, pair = distance.max(dim = 0)
            
            indices.append(pair)
            independ.append(data[pair,:][None,:])
            mask[pair] = 0
            
        # betas = torch.zeros([1,10])
        # trans = torch.zeros([1,3])
        # smpl = SMPL(pathSMPL, 'cpu')
        
        # pathobje = pjn('/'.join(pathSMPL.split('/')[:-1]), 'text_uv_coor_smpl.obj')
        # _, smplMesh, smpl_text_uv_coord, smpl_text_uv_mesh = read_Obj(pathobje)
        
        # for ind in range(keepN):
        #     v = smpl(independ[ind], betas, trans)        
        #     body = create_fullO3DMesh(v[0], smplMesh)    
        #     o3d.visualization.draw_geometries([body])
        
        return independ, indices
    
    def augment_MGN(self, aug_data: list = ['pose']):
        
        for augTar in aug_data:
            
            if augTar == 'pose':
                
                independ, indices = self.diversity(self.poses)
                smpl = SMPL(self.path_smpl, 'cpu')
                
                for ind, path in enumerate(self.path_subjects):
                    subname = path.split('/')[-1]
                                      
                    pathobje = pjn('/'.join(self.path_smpl.split('/')[:-1]), 'text_uv_coor_smpl.obj')
                    _, smplMesh, smpl_text_uv_coord, smpl_text_uv_mesh = read_Obj(pathobje)
                    
                    
                    v = smpl(independ[ind], betas, trans)        
                    body = create_fullO3DMesh(v[0], smplMesh)    
                    o3d.visualization.draw_geometries([body])
                
                    for pose, poseInd in zip(independ, indices):
                        if poseInd == ind:
                            continue
                        
                        newfolder = pjn(path.split('/')[:-1], subname+'_poseInd')
                        os.mkdir(newfolder)
                        
                        copy2( pjn(path, 'registered_tex.jpg'), newfolder)
                        copy2( pjn(path, 'smpl_registered.obj'), newfolder)
                        copy2( pjn(path, 'segmentation.png'), newfolder)
                        
                        
                        o3d.io.write_triangle_mesh( pjn(newfolder, 'smpl_registered.obj'), grndTruthMesh )
                        
                        
                
            else:
                raise NotImplementedError('other augmentations not implemented yet')
                
    
        
pathmodel = '/home/zhantao/Documents/masterProject/DEBOR/body_model/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
path = '../../datasets/MGN_brighter'
mgn  = MGN_subjectPreAugmentation(path, pathmodel)
mgn.diversity(mgn.poses, pathmodel)
