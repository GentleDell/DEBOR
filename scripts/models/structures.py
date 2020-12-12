#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 23:05:02 2020

@author: zhantao
"""

import torch
import torch.nn as nn

from resnet import resnet50


class multiple_downstream(nn.Module):
    
    def __init__(self, structure_options = None):
        super(multiple_downstream, self).__init__()
        
        self.options = structure_options
        
        # feature extractor, output {Bx2048x7x7}
        self.resnet50 = resnet50(pretrained = True,
                                 outAvgPool = False)
        
        # downstream MLP for camera parameters, regresse the 6D representation
        # 6D rotation representation and 3D translation.       
        if self.options.camereMLPseparate:
            
            self.rotation = _make_layer(2048, 7, 6, layerSize)
            
            
            self.cameraNet= nn.Sequential(
                    nn.Linear(2048, self.options.camereMLP[0]),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.options.camereMLP[0], 
                              self.options.camereMLP[1]),
                    nn.ReLU(inplace=True),
                    nn.Linear(9, 3))    # 6 rot, 3 trans
        
    
    def forward(self, x):
        
        # extract features separately for camera, SMPL, disp and color
        # shape {Bx2048x4}
        x = self.resnet50(x)
        
        # 
        
        return x    
    
    
verif = multiple_downstream()
out = verif(torch.Tensor(1, 3,224,224))