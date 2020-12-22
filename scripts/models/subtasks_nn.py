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
from .unet import upsampleLayer, downsampleLayer


class simpleMLP(nn.Module):
    '''
    Simple MLP structure for encoding and decoding.
    
    Structure adapted from:
        Instant recovery of shape from spectrum via latentspace connections
        https://github.com/riccardomarin/InstantRecoveryFromSpectrum
    '''
    def __init__(self, infeature, outfeature,
                 layers = [80, 160, 320, 640, 320, 160, 80],
                 bactchNormOn = True, actFunc = 'selu'):
        super(simpleMLP, self).__init__()
        
        MLPlayers = []
        
        MLPlayers.append(nn.Linear(infeature, layers[0]))
        for n in range(1, len(layers)):
            
            if actFunc.lower() == 'selu':
                MLPlayers.append(nn.SELU())
            else:
                MLPlayers.append(nn.ReLU())
            
            if bactchNormOn:
                MLPlayers.append(nn.BatchNorm1d(layers[n-1]))
                
            MLPlayers.append(nn.Linear(layers[n-1], layers[n]))
        
        MLPlayers.append(nn.SELU())
        MLPlayers.append(nn.BatchNorm1d(layers[n]))
        MLPlayers.append(nn.Linear(layers[n], outfeature))          
        
        self.NN = nn.Sequential(*MLPlayers)
        
    def forward(self, x):
        
        return self.NN(x)        

class dispGraphNet(nn.Module):
    
    def __init__(self, A, ref_vertices, num_layers=5, num_channels=512, 
                 infeature = 2048, inShape = 7):
        super(dispGraphNet, self).__init__()
        
        self.A = A
        self.ref_vertices = ref_vertices
        self.inconv = nn.Conv2d(infeature, infeature, inShape)

        layers = [GraphLinear(3 + infeature, 2 * num_channels)]
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
        image_enc = image_enc.flatten(start_dim=-2).expand(-1, -1, ref_vertices.shape[-1])
        
        x = torch.cat([ref_vertices, image_enc], dim=1)
        x = self.gc(x)
        shape = self.shape(x).permute(0,2,1)

        return shape

class downNet(nn.Module):
    def __init__(self, dropRate: float = 0, batchNormalization: bool = True):
        super(downNet, self).__init__()
        
        base_depth = 64
        kernelSize = 4
        
        # structure for downsampling 
        self.d1 = downsampleLayer(3, base_depth, kernelSize)
        self.d2 = downsampleLayer(base_depth  , base_depth*2, kernelSize, bn=True)
        self.d3 = downsampleLayer(base_depth*2, base_depth*4, kernelSize, bn=True)
        self.d4 = downsampleLayer(base_depth*4, base_depth*8, kernelSize, bn=True)
        self.d5 = downsampleLayer(base_depth*8, base_depth*16, kernelSize, paddings=2, bn=True)
        self.d6 = downsampleLayer(base_depth*16, base_depth*32, kernelSize, bn=True) 
        
        self.dn = nn.AvgPool2d(kernelSize)
        
    def forward(self, x):
        
        d1 = self.d1(x)     # 112x112x64
        d2 = self.d2(d1)    # 56x56x128
        d3 = self.d3(d2)    # 28x28x256
        d4 = self.d4(d3)    # 14x14x512
        d5 = self.d5(d4)    # 8x8x1024
        d6 = self.d6(d5)    # 4x4x2048
        
        dn = self.dn(d6)    # 1x1x2048 for latent loss
        dn = dn.view(dn.shape[0], 2048, -1)
        
        return d6, [d1, d2, d3, d4, d5], dn.squeeze()
    
class upNet(nn.Module):
    def __init__(self, dropRate: float = 0, batchNormalization: bool = True,
                 infeature = 2048, outdim = 3, outkernelSize = 3):
        super(upNet, self).__init__()
        
        base_depth = 64
        kernelSize = 4
                
        self.u1 = upsampleLayer(infeature, int(infeature/2), kernelSize-1,                      
                                dropout_rate=dropRate, bn=batchNormalization)
        self.u2 = upsampleLayer(int(infeature/2) + int(infeature/2), int(infeature/4), kernelSize-1,     
                                paddings=0, dropout_rate=dropRate, bn=batchNormalization) 
        self.u3 = upsampleLayer(int(infeature/4) + int(infeature/4), base_depth*4, kernelSize-1,      
                                dropout_rate=dropRate, bn=batchNormalization)
        self.u4 = upsampleLayer(base_depth*8, base_depth*2, kernelSize-1,
                                dropout_rate=dropRate, bn=batchNormalization)
        self.u5 = upsampleLayer(base_depth*4, base_depth  , kernelSize-1,                    
                                dropout_rate=dropRate, bn=batchNormalization)
        self.u6 = upsampleLayer(base_depth*2, base_depth    , kernelSize-1,                    
                                dropout_rate=dropRate, bn=batchNormalization)
        
        self.outLayer = nn.Conv2d(base_depth, outdim, outkernelSize, padding=1)
        
    def forward(self, x, layers: list):
        
        assert len(layers) == 5, 'to keep fine details, pass all 5 layers'
        
        u1 = self.u1(x , layers[4])     # 2048x8x8
        u2 = self.u2(u1, layers[3])     # 1024x14x14
        u3 = self.u3(u2, layers[2])     # 512x28x28
        u4 = self.u4(u3, layers[1])     # 256x56x56
        u5 = self.u5(u4, layers[0])     # 128x112x112
        u6 = self.u6(u5)                # 64x224x224
        
        out = self.outLayer(u6)         # outdimx224x224
        
        return out

class iterativeNet(nn.Module):
    '''
    Code adapted from from TexturePose[1], pytorch_HMR[2] and expose[3]. 
    
    This class iteratively regress the delta pose and shape and camera parameters
    from corresponding mean values, given the latent representation from up
    stream network.
    
    In some implementations, there are activation function layers but not in 
    other implementations.
    
    [1] https://github.com/geopavlakos/TexturePose
    [2] https://github.com/MandyMo/pytorch_HMR/
    [3] https://github.com/vchoutas/expose

    '''    
    def __init__(self, infeature, npose=144, ncam=7, 
                 initPose=None, initBeta=None, initTrans=None, initCam=None,
                 numiters=3, withActFunc=False):
        super(iterativeNet, self).__init__()
        
        print('Settings for camera net will be ignored as iterNet is used.')
        
        self.initPose  = initPose
        self.initBeta  = initBeta
        self.initTrans = initTrans
        self.initCam   = initCam
        
        self.numiters = numiters
        self.actFuncOn= withActFunc
        
        self.fc1   = nn.Linear(infeature + npose + 10 + ncam + 3, 1024)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout()
        self.fc2   = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout()
        
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.dectrans = nn.Linear(1024, 3)
        self.deccamera = nn.Linear(1024, ncam)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.dectrans.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccamera.weight, gain=0.01)
        
    def forward(self, xin, init_pose=None, init_shape=None, 
                init_trans=None, init_cam=None):
        '''
        Iteratively estiamte the pose, shape and camera parameters.

        Parameters
        ----------
        x : 
            Latent feature representation from up stream network.
        init_pose, init_shape, init_cam : 
            Initial value of the parameters
        Returns
        -------
        out : dict of predicted parameters.

        '''
        out = {}
        
        predPose = (init_pose, self.initPose)[init_pose is None]
        predBeta = (init_shape, self.initBeta)[init_shape is None]
        predTran = (init_trans, self.initTrans)[init_trans is None]
        predCam  = (init_cam, self.initCam)[init_cam is None]
        
        for cnt in range(self.numiters):
            
            xf = torch.cat([xin,predPose, predBeta, predTran, predCam],1)
            xf = self.fc1(xf)
            if self.actFuncOn:
                xf = self.relu1(xf)
            xf = self.drop1(xf)        
            
            xf = self.fc2(xf)
            if self.actFuncOn:
                xf = self.relu2(xf)
            xf = self.drop2(xf)
            
            predPose  = self.decpose(xf) + predPose
            predBeta = self.decshape(xf) + predBeta
            predTran  =  self.dectrans(xf) + predTran
            predCam  = self.deccam(xf) + predCam
            
            out['iter%d'%cnt] = \
                torch.cat([predPose, predBeta, predTran, predCam], dim = 1)

        return out[-1]
          