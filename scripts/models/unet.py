#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 11:21:39 2020

@author: zhantao
"""

import torch
import torch.nn as nn


class downsampleLayer(nn.Module):
    '''
    A downsample layer of UNet. LeakyReLU is used as the activation func. 
    '''
    def __init__(self, infeature, outfeature, kernelSize, 
                 strides = 2, paddings = 1, bn = False):
        super(downsampleLayer, self).__init__()
            
        self.conv = nn.Conv2d(infeature, outfeature, kernelSize, 
                              stride=strides, padding=paddings)
        self.acti = nn.LeakyReLU(negative_slope = 0.2)
        
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(outfeature, momentum = 0.8)
        
    def forward(self, x):
        y = self.acti(self.conv(x))
        
        if self.bn is not None:
            y = self.bn(y)
            
        return y

class upsampleLayer(nn.Module):
    '''
    A upsample layer of UNet. ReLU is the activation func. The skip connection 
    can be cutted if not given. Because RGB-UV is not a completion task but a
    image transition task.
    '''
    def __init__(self, infeature, outfeature, kernelSize, 
                 strides = 1, paddings = 1, bn = False, dropout_rate = 0):
        super(upsampleLayer, self).__init__()
        
        self.upsp = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(infeature, outfeature, kernelSize, 
                              stride=strides, padding=paddings)
        self.acti = nn.ReLU()
        
        self.drop = None
        if dropout_rate != 0:
            self.drop = nn.Dropout2d(p=dropout_rate)
        
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(outfeature, momentum = 0.8)
            
    def forward(self, x, skip_input = None):
        y = self.conv(self.upsp(x))
        
        if self.drop is not None:
            y = self.drop(y)
        
        if self.bn is not None:
            y = self.bn(y)
        
        if skip_input is not None:
            y = torch.cat( (y, skip_input), 1)
        
        return y
    
class unet_core(nn.Module):
    '''
    Pytorch reimplementation of the UNet used in tex2shape with some changes.
    '''
    def __init__(self, dropRate: float = 0, batchNormalization: bool = True):
        super(unet_core, self).__init__()

        base_depth = 64
        kernelSize = 4
                
        # structure for downsampling 
        self.d1 = downsampleLayer(3, base_depth, kernelSize)
        self.d2 = downsampleLayer(base_depth  , base_depth*2, kernelSize, bn=True)
        self.d3 = downsampleLayer(base_depth*2, base_depth*4, kernelSize, bn=True)
        self.d4 = downsampleLayer(base_depth*4, base_depth*8, kernelSize, bn=True)
        self.d5 = downsampleLayer(base_depth*8, base_depth*8, kernelSize, paddings=2, bn=True)
        self.d6 = downsampleLayer(base_depth*8, base_depth*8, kernelSize, bn=True)        
        
        # structure for upsampling
        self.u1 = upsampleLayer(base_depth*8, base_depth*8,  kernelSize-1, dropout_rate=dropRate, bn=batchNormalization)
        self.u2 = upsampleLayer(base_depth*16, base_depth*8, kernelSize-1, paddings=0, dropout_rate=dropRate, bn=batchNormalization) 
        self.u3 = upsampleLayer(base_depth*16, base_depth*4, kernelSize-1, dropout_rate=dropRate, bn=batchNormalization)
        self.u4 = upsampleLayer(base_depth*8, base_depth*2, kernelSize-1, dropout_rate=dropRate, bn=batchNormalization)
        self.u5 = upsampleLayer(base_depth*4, base_depth  , kernelSize-1, dropout_rate=dropRate, bn=batchNormalization)
        self.u6 = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, d0):
        
        d1 = self.d1(d0)    # 112x112x64
        d2 = self.d2(d1)    # 56x56x128
        d3 = self.d3(d2)    # 28x28x256
        d4 = self.d4(d3)    # 14x14x512
        d5 = self.d5(d4)    # 8x8x512
        d6 = self.d6(d5)    # 4x4x512
        
        u1 = self.u1(d6, d5)    # 8x8x1024
        u2 = self.u2(u1, d4)    # 14x14x1024
        u3 = self.u3(u2, d3)    # 28x28x512
        u4 = self.u4(u3, d2)    # 56x56x256
        u5 = self.u5(u4, d1)    # 112x112x128
        
        u6 = self.u6(u5)        # 224x224x128
        
        return u6
          
class UNet(nn.Module):
    '''
    Pytorch reimplementation with our modifications of the original UNet used 
    in tex2shape.
    
    To use the UNet for RGB -> UV texture transition, we cut the high-level 
    skip connections.
    '''
    def __init__(self, input_shape = (512, 512, 3), output_dims = 6,
                 kernel_size = 3, dropout_rate = 0, bn = True):
        '''
        output shape could be larger, instead of 224x224 again.
        '''
        
        super(UNet, self).__init__()
        
        self.input_shape = input_shape
        self.output_dims = input_shape[2] if output_dims is None else output_dims
        
        self.dropout_rate = dropout_rate
        self.bn = bn
        
        self.unet_core = unet_core(dropRate=dropout_rate, batchNormalization=bn)
        self.outLayer = nn.Conv2d( 128, output_dims, kernel_size, padding=1)
        
    def forward(self, x, pose = None):
        
        if pose is not None:
            print('things to consider:\n'+\
                  '1. should we use axis angle or rotation matrix'+\
                  '2. the problem of continuous representation mentioned in the paper'+\
                  '3. should we append or concatenate to the latent vector or append to all layers with MLP belencing the dimension or both'+\
                  '4. try to support Sizer first')
            raise NotImplementedError('combining pose is not implemented yet, detials see above info')
            
        x = self.unet_core(x)    # 224x224x128
        x = self.outLayer(x)     # 224x224x output_dim
        
        return x.permute(0,2,3,1)

if __name__ == "__main__":
    model = UNet()
    model.summary()
