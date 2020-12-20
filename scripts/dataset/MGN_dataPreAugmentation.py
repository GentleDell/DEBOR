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
from distutils.dir_util import copy_tree
import sys 
sys.path.append( "../" )

import torch
import numpy as np
import open3d as o3d

from models import SMPL
from utils.vis_util import read_Obj
from utils.mesh_util import create_fullO3DMesh


def remove_irrelevant(folder):
    
    # rename and convert the image format
    tex_image = pjn(folder, 'registered_tex.jpg')
    seg_image = pjn(folder, 'segmentation.png')
    img = o3d.io.read_image( pjn(folder, 'smpl_registered_0.png') )
    o3d.io.write_image(tex_image, img)
    img = o3d.io.read_image( pjn(folder, 'smpl_registered_1.png') )
    o3d.io.write_image(seg_image, img)
    
    # remove irrelevant data
    os.remove( pjn(folder, 'smpl_registered.mtl') )
    os.remove( pjn(folder, 'smpl_registered_0.png') )
    os.remove( pjn(folder, 'smpl_registered_1.png') )
    
    # remove mtl in the obj file which brings conflict during rendering
    rmLines = 5
    nfirstlines = []
    targetFile  = pjn(folder, 'smpl_registered.obj')
    tempFile    = pjn(folder, 'temp.obj')
    with open(targetFile) as f, open(tempFile, "w") as out:
        for x in range(rmLines):
            nfirstlines.append(next(f))
        for line in f:
            out.write(line)
    os.remove(targetFile)
    os.rename(tempFile, targetFile)
        
    
class MGN_subjectPreAugmentation(object):
    
    def __init__(self, path_MGN: str, pathSMPL: str):
        
        self.path_smpl = pathSMPL
        self.path_subjects = sorted( glob(pjn(path_MGN, '*')) )
        
        self.text, self.disp = [], []
        self.betas, self.poses, self.trans = [], [], []
        for subjectPath in self.path_subjects:
            self.text.append( np.load(pjn(subjectPath, 
                                      'GroundTruth/vertex_colors_oversample_OFF.npy') ))
            self.disp.append( np.load(pjn(subjectPath, 
                                      'GroundTruth/normal_guided_displacements_oversample_OFF.npy')))
            
            pathRegistr  = pjn(subjectPath, 'registration.pkl')
            registration = pickle.load(open(pathRegistr, 'rb'),  encoding='iso-8859-1')
            self.betas.append( torch.Tensor(registration['betas'][None,:]) )
            self.poses.append( torch.Tensor(registration['pose'][None,:]) )
            self.trans.append( torch.Tensor(registration['trans'][None,:]) )
    
    def diversity(self, dataList: list, keepN: int = 15, 
                  threshold: int= 2.5) -> list:
        
        data = torch.cat(dataList, dim = 0)
        
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
        '''
        Augment the data required by the aug_data. 
        
        A greedy method is used to obtain the diverse samples for augmentation.
        Then each subject in the original MGN dataset will be duplicated with 
        the new/augmented parameters, such as pose, disp, etc. New subjects 
        will be save to a new folder.
        

        Parameters
        ----------
        aug_data : list, optional
            The data to be augmented. The default is ['pose'].

        Raises
        ------
        NotImplementedError
            Augmentation for other data are not implemented.

        Returns
        -------
        None.

        '''
        for augTar in aug_data:
            
            if augTar == 'pose':
                # obtain diverse samples
                independ, indices = self.diversity(self.poses)
                
                # prepare smpl model on pytorch
                smpl = SMPL(self.path_smpl, 'cpu')
                
                # read smpl object file
                pathobje = pjn('/'.join(self.path_smpl.split('/')[:-1]), 'text_uv_coor_smpl.obj')
                _, smplMesh, smpl_text_uv_coord, smpl_text_uv_mesh = read_Obj(pathobje)
                smpl_text_uv_coord[:, 1] = 1 - smpl_text_uv_coord[:, 1]    # SMPL .obj need this conversion
                
                # augment all objects
                for ind, path in enumerate(self.path_subjects):
                    
                    subname = path.split('/')[-1]
                    
                    # for each subject, augment it with all samples
                    for pose, poseInd in zip(independ, indices):
                        # skip if duplicated
                        if poseInd == ind:
                            continue
                                               
                        # produce a new body
                        v = smpl(pose, self.betas[ind], self.trans[ind])        
                        dressed_body = create_fullO3DMesh(
                            v[0] + self.disp[ind],
                            smplMesh,
                            texturePath=pjn(path, 'registered_tex.jpg'),
                            segmentationPath=pjn(path, 'segmentation.png'),
                            triangle_UVs=smpl_text_uv_coord[smpl_text_uv_mesh.flatten()],
                            use_text=True)    
                        # o3d.visualization.draw_geometries([dressed_body])
                        
                        
                        # create new folder for the new subject
                        newfolder = pjn('/'.join(path.split('/')[:-1]), subname+'_pose%d'%(poseInd))
                        os.mkdir(newfolder)
                        os.mkdir(pjn(newfolder, 'GroundTruth'))
                        
                        # copying related files
                        copy_tree( pjn(path, 'GroundTruth'), pjn(newfolder, 'GroundTruth') )
                        
                        # saving object
                        o3d.io.write_triangle_mesh( pjn(newfolder, 'smpl_registered.obj'), dressed_body )
                        remove_irrelevant(newfolder)
                        
                        # save new SMPL parameters
                        smplParams = {'betas': self.betas[ind],
                                      'pose':  pose,
                                      'trans': self.trans[ind]
                                     }
                        pickle.dump( smplParams, open( pjn(newfolder, "registration.pkl"), "wb" ) )
            else:
                raise NotImplementedError('other augmentations not implemented yet')
      
        
if __name__ == "__main__":
    '''
    To do augmentation, folders of subjects must have the "/GroundTruth"
    subfolder containing displacements for augmentation.
    
    Thus, a proper order of the dataset properation is:
        1. run MGN_dataPreperation.py with 'enable_displacement' on and 
        "enable_rendering" off.
        
        2. run this script(MGN_dataPreAugmentation.py) to augment the dataset.
        
        3. run MGN_dataPreperation.py with 'enable_displacement' off and 
        "enable_rendering" on to render RGB images.
    
    '''
    pathmodel = '/home/zhantao/Documents/masterProject/DEBOR/body_model/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
    path = '../../datasets/MGN_brighter'
    mgn  = MGN_subjectPreAugmentation(path, pathmodel)
    # mgn.augment_MGN()