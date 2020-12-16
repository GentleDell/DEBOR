#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 23:13:35 2020

@author: zhantao
"""

import torch 
import torch.nn as nn

class latentCode_loss(nn.Module):
    '''
    Latent codes loss is to constrain the learned low dimension latent 
    representation to contain the information of the target field, such as
    pose, shape, displacements. 
    
    Plane Unet and Graph net are easily overfitted to our dataset and the 
    learned knowledge is turned out more like memory. 
    
    These losses are defined as:
        loss = || Enc(Img)[i:j,:] - Enc_x(X) ||, x can be smpl, disp, etc.
        
    '''
    def __init__(self):
        super(latentCode_loss, self).__init__()
        
        self.smplCode_loss
        self.dispCode_loss
        self.textCode_loss
        self.cameraCode_loss
        
class supervision_loss(nn.Module):
    '''
    Supervision loss supervises networks to learn the target information 
    explicitly. 
    
    The loss is defined as:
        loss = || gt_x - Dec_x(Enc(img)[i:j,:]) ||, x can be smpl, disp, etc.
        
    '''
    def __init__(self):
        super(supervision_loss, self).__init__()
        
        self.smplSupv_loss
        self.dispSupv_loss
        self.textSupv_loss    # maybe after indexMap better
        self.cameraSupv_loss
        self.indexMapSupv_loss
        
class rendering_loss(nn.Module):
    '''
    To make sure that the predicted parameters are realistic and reasonable, 
    the rendering loss is added.
    '''
    def __init__(self):
        super(self.indexMapSupv_loss, self).__init__()
        
        self.indexMap_textloss
        self.renderImg_loss
            
class batch_loss(nn.Module):
    def __init__(self):
        super(batch_loss, self).__init__()
        
        self.smplBatch_loss
        self.dispBatch_loss
        self.textBatch_loss
        self.cameraBatch_loss
        self.indexMapBatch_loss
        
