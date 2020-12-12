#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 21:09:40 2020

@author: zhantao
"""
from __future__ import division

import torch 
import torch.nn as nn

from .graph_layers import GraphResBlock, GraphLinear
from .unet import upsampleLayer

class dispNet(nn.Module):
    
    def __init__(self, A, ref_vertices, num_layers=5, num_channels=512, 
                 infeature = 2048, inShape = 7):
        super(dispNet, self).__init__()
        
        self.A = A
        self.ref_vertices = ref_vertices
        self.inconv = nn.Conv2d(infeature, infeature, inShape)

        layers = [GraphLinear(3 + 2048, 2 * num_channels)]
        layers.append(GraphResBlock(2 * num_channels, num_channels, A))
        
        for i in range(num_layers):
            layers.append(GraphResBlock(num_channels, num_channels, A))
        
        self.shape = nn.Sequential(GraphResBlock(num_channels, 64, A),
                                   GraphResBlock(64, 32, A),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, 3))
        self.gc = nn.Sequential(*layers)

    def forward(self, x, pose = None):
        """Forward pass
        Inputs:
            x    : size = (B, 2048, 7, 7)
            pose : size = (B, 1, 24)
        Returns:
            Regressed (subsampled) non-parametric shape: size = (B, 6890, 3)
        """
        
        if pose is not None:
            raise NotImplementedError('combining pose is not implemented yet, detials see above info')
        
        batch_size = x.shape[0]
        ref_vertices = self.ref_vertices[None, :, :].expand(batch_size, -1, -1)
                
        image_enc = self.inconv(x)
        x = torch.cat([ref_vertices, image_enc], dim=1)
        x = self.gc(x)
        shape = self.shape(x).permute(0,2,1)

        return shape
    
class textureNet(nn.Module):
    def __init__(self):
        super(textureNet, self).__init__()
        
    def forward(self, x, layers: list):
        
        return x

class poseNet(nn.Module):
    '''
    Bring ideas and codes from TexturePose[1], pytorch_HMR[2] and expose[3].
    
    [1] https://github.com/geopavlakos/TexturePose
    [2] https://github.com/MandyMo/pytorch_HMR/
    [3] https://github.com/vchoutas/expose

    '''    
    def __init__(self, infeature, inShape, npose, numiters):
        super(poseNet, self).__init__()
        
        self.numiters = numiters
        
        self.inconv = nn.Conv2d(infeature, infeature, inShape)
        self.fc1   = nn.Linear(2048  + npose + 10, 1024)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout()
        self.fc2   = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout()
        
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        
    def forward(self, x, init_pose, init_shape):
        '''
        Iteratively estiamte the pose and shape parameters

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        init_pose : TYPE
            DESCRIPTION.
        init_shape : TYPE
            DESCRIPTION.

        Returns
        -------
        out : TYPE
            DESCRIPTION.

        '''
        out = []
        
        xf = self.inconv(x)
        xf = xf.view(xf.size(0), -1)
        
        for cnt in range(self.numiters):
            
            xf = torch.cat([xf,init_pose,init_shape],1)
            xf = self.fc1(xf)
            xf = self.relu1(xf)
            xf = self.drop1(xf)        
            xf = self.fc2(xf)
            xf = self.relu2(xf)
            xf = self.drop2(xf)
            
            init_pose  = self.decpose(xf) + init_pose
            init_shape = self.decshape(xf) + init_shape
            
            out.append(init_pose)
            out.append(init_shape)

        return out
    
class cameraNet(nn.Module):
    
    def __init__(self, infeature, inShape, numiters):
        super(cameraNet, self).__init__()
        
        self.numiters = numiters
        
        self.inconv = nn.Conv2d(infeature, infeature, inShape)
        self.fc1   = nn.Linear(2048 + 3, 1024)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout()
        self.fc2   = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout()
        
        self.deccam = nn.Linear(1024, 3)
        
    def forward(self, x, init_cam):
        out = []
    
        xf = self.inconv(x)
        xf = xf.view(xf.size(0), -1)
        
        for cnt in range(self.numiters):
            
            xf = torch.cat([xf,init_cam],1)
            xf = self.fc1(xf)
            xf = self.relu1(xf)
            xf = self.drop1(xf)        
            xf = self.fc2(xf)
            xf = self.relu2(xf)
            xf = self.drop2(xf)
            
            init_cam  = self.deccam(xf) + init_cam
            
            out.append(init_cam)

        return out
    