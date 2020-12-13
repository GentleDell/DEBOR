#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 23:05:02 2020

@author: zhantao
"""
from os.path import abspath
import sys
if __name__ == '__main__':
    if abspath('../') not in sys.path:
        sys.path.append(abspath('../'))
from collections import namedtuple

import torch
import torch.nn as nn

from resnet import resnet50
from utils import Mesh
from subtasks_nn import cameraNet, poseNet, textupNet, dispNet

structure_options = {'cut_feature': False,    # cut feature for diff tasks
                     'condition_dispOnPose' : True,
                     'condition_clrOnCam' : True,
                     'backwardNetOn': False,
                     'graph_matrics_path' : '/home/zhantao/Documents/masterProject/DEBOR/body_model/mesh_downsampling.npz',
                     'smpl_model_path' : '/home/zhantao/Documents/masterProject/DEBOR/body_model/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
                     }


class multiple_downstream(nn.Module):
    
    def __init__(self, structure_options, A, ref_vertices, device = 'cuda'):
        super(multiple_downstream, self).__init__()
        
        self.options = structure_options
        self.infeature = 2048
        if self.options.cut_feature:
            self.infeature = self.infeature/4
        
        # common feature extractor, output {Bx2048x7x7}
        self.resnet50 = resnet50(pretrained = True,
                                 outAvgPool = False,
                                 outdownLayers = True).to(device)
        
        # downstream MLP for camera parameters
        self.cameraNN  = cameraNet(self.infeature, inShape=7).to(device)
        
        # downstream upnet for texture
        self.textureNN = textupNet(infeature=self.infeature, inShape=7).to(device)
        
        # downstream pose net
        self.smplPoseNN = poseNet(self.infeature, inShape=7).to(device)
        
        # downstream displacement graph net
        self.clothesNN = dispNet(A, ref_vertices).to(device)
        
        # backward subnet
        if options.backwardNetOn:
            self. cameraBackNet = None
            raise NotImplementedError('to be implemented')
    
    def forward(self, x, init_cam, init_pose, init_shape):
        
        # extract features separately for camera, SMPL, disp and color
        # shape {Bx2048x4}
        x, connections = self.resnet50(x)
        
        if self.options.cut_feature:
            raise NotImplementedError('cut feature to be implemented')
        else:
            # predict all components
            camera = self.cameraNN(x, init_cam)
            smpl   = self.smplPoseNN(x, init_pose, init_shape)
            disps  = self.clothesNN(x)
            color  = self.textureNN(x,connections)
        
        out = {'camera': camera,
               'smpl': smpl,
               'disp': disps,
               'color': color }        
        
        return out  
    
    
class multiple_parallel(nn.Module):
    def __init__(self, structure_options, A, ref_vertices, device = 'cuda'):
        super(multiple_downstream, self).__init__()
    

options = namedtuple('options', structure_options.keys())(**structure_options)
mesh = Mesh(options, 0, device = 'cpu')

verif = multiple_downstream(options, mesh.adjmat, mesh.ref_vertices.t(), 'cpu')
out = verif(torch.Tensor(1, 3,224,224),
            init_cam = torch.tensor([0.9, 0., 0.]).view(1, 3),
            init_pose = torch.zeros(144)[None,:],
            init_shape = torch.zeros(10)[None, :])