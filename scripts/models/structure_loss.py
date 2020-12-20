#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 23:13:35 2020

@author: zhantao
"""
import random
from os.path import join as pjn
from collections import namedtuple

import torch 
import torch.nn as nn

from .camera import cameraPerspective as camera
from .smpl import SMPL
from .geometric_layers import rot6d_to_axisAngle, rot6d_to_rotmat
from utils.vis_util import read_Obj

class latentCode_loss(nn.Module):
    '''
    Latent codes loss is to constrain the learned low dimension latent 
    representation to contain the information of the target field, such as
    pose, shape, displacements. 
    
    Plane Unet and Graph net can easily overfit to our dataset while they
    still underfit the testing set. This could indicate that the network 
    does not learn things/distructions correctly. So the latent code loss 
    is added.
    
    These losses are defined as:
        loss = || Enc(Img)[i:j,:] - Enc_x(X) ||, x can be smpl, disp, etc.

    According to the structures, this class computes the loss for 
    SMPLCode, dispCode, textCode, cameraCode and returns a dict of
    
    {'smplCode_loss', 'dispCode_loss', 'textCode_loss', 'cameraCode_loss'}
        
    '''
    def __init__(self, options):
        super(latentCode_loss, self).__init__()
        
        self.options = options
        
        self.SMPLcfg = namedtuple(
            'options', 
            self.options.structureList['SMPL'].keys())\
                (**self.options.structureList['SMPL'])
        self.dispcfg = namedtuple(
            'options', 
            self.options.structureList['disp'].keys())\
            (**self.options.structureList['disp'])
        self.textcfg = namedtuple(
            'options', 
            self.options.structureList['text'].keys())\
            (**self.options.structureList['text'])
        self.cameracfg = namedtuple(
            'options', 
            self.options.structureList['camera'].keys())\
            (**self.options.structureList['camera'])
            
        self.SMPLlossfunc = (nn.L1Loss(), nn.MSELoss())\
                [self.SMPLcfg.latent_lossFunc == 'L2']
        self.displossfunc = (nn.L1Loss(), nn.MSELoss())\
                [self.dispcfg.latent_lossFunc == 'L2']
        self.textlossfunc = (nn.L1Loss(), nn.MSELoss())\
                [self.textcfg.latent_lossFunc == 'L2']
        self.cameralossfunc = (nn.L1Loss(), nn.MSELoss())\
                [self.cameracfg.latent_lossFunc == 'L2']
        
    def forward(self, latentCode):
        latCodeLoss = {'smplCode_loss': 0,
                       'dispCode_loss': 0,
                       'textCode_loss': 0,
                       'cameraCode_loss' : 0}

        imgCode = latentCode['imgAggCode']
        if self.SMPLcfg.enable:
            assert latentCode['SMPL'] is not None                
            target = imgCode[:,
                    self.SMPLcfg.latent_start:
                    self.SMPLcfg.latent_start + self.SMPLcfg.latent_shape]
            latCodeLoss['smplCode_loss'] = \
                self.SMPLlossfunc(latentCode['SMPL'], target)
        if self.dispcfg.enable:
            assert latentCode['disp'] is not None
            target = imgCode[:,
                    self.dispcfg.latent_start:
                    self.dispcfg.latent_start + self.dispcfg.latent_shape]
            latCodeLoss['dispCode_loss'] = \
                self.displossfunc(latentCode['disp'], target)
        if self.textcfg.enable:
            assert latentCode['text'] is not None
            target = imgCode[:,
                    self.textcfg.latent_start:
                    self.textcfg.latent_start + self.textcfg.latent_shape]
            latCodeLoss['textCode_loss'] = \
                self.textlossfunc(latentCode['text'], target)
        if self.cameracfg.enable:
            assert latentCode['camera'] is not None
            target = imgCode[:,
                    self.cameracfg.latent_start:
                    self.cameracfg.latent_start + self.cameracfg.latent_shape]
            latCodeLoss['cameraCode_loss'] = \
                self.cameralossfunc(latentCode['camera'], target)
        
        return latCodeLoss
        
class supervision_loss(nn.Module):
    '''
    Supervision loss supervises networks to learn the target information 
    explicitly. 
    
    The loss is defined as:
        loss = || gt_x - Dec_x(Enc(img)[i:j,:]) ||, x can be smpl, disp, etc.
    
    Returns a dict of:
    
    {'smplSupv_loss', 'dispSupv_loss', 'textSupv_loss', 'cameraSupv_loss',
     'indexMapSupv_loss'}
        
    '''
    def __init__(self, options):
        super(supervision_loss, self).__init__()
        self.options = options
        
        self.SMPLcfg = namedtuple(
            'options', 
            self.options.structureList['SMPL'].keys())\
                (**self.options.structureList['SMPL'])
        self.dispcfg = namedtuple(
            'options', 
            self.options.structureList['disp'].keys())\
            (**self.options.structureList['disp'])
        self.textcfg = namedtuple(
            'options', 
            self.options.structureList['text'].keys())\
            (**self.options.structureList['text'])
        self.cameracfg = namedtuple(
            'options', 
            self.options.structureList['camera'].keys())\
            (**self.options.structureList['camera'])
        self.indMapcfg = namedtuple(
            'options', 
            self.options.structureList['indexMap'].keys())\
            (**self.options.structureList['indexMap'])
        
        self.SMPLlossfunc = (nn.L1Loss(), nn.MSELoss())\
                [self.SMPLcfg.supVis_lossFunc == 'L2']
        self.displossfunc = (nn.L1Loss(), nn.MSELoss())\
                [self.dispcfg.supVis_lossFunc == 'L2']
        self.textlossfunc = (nn.L1Loss(), nn.MSELoss())\
                [self.textcfg.supVis_lossFunc == 'L2']
        self.cameralossfunc = (nn.L1Loss(), nn.MSELoss())\
                [self.cameracfg.supVis_lossFunc == 'L2']
        self.indMapLossfunc = (nn.L1Loss(), nn.MSELoss())\
                [self.indMapcfg.supVis_lossFunc == 'L2']
        
    def forward(self, prediction, GT):
        supVisLoss = {'smplSupv_loss': 0,
                      'dispSupv_loss': 0,
                      'textSupv_loss': 0,
                      'cameraSupv_loss': 0,
                      'indexMapSupv_loss': 0}
        if self.SMPLcfg.enable:
            assert GT['SMPL'] is not None
            supVisLoss['smplSupv_loss'] = \
                self.SMPLlossfunc(prediction['SMPL'], GT['SMPL'])
        if self.dispcfg.enable:
            assert GT['disp'] is not None
            supVisLoss['dispSupv_loss'] = \
                self.displossfunc(prediction['disp'], GT['disp'])
        if self.textcfg.enable:
            assert GT['text'] is not None
            supVisLoss['textSupv_loss'] = \
                self.textlossfunc(prediction['text'], GT['text'])
        if self.cameracfg.enable:
            assert GT['camera'] is not None
            supVisLoss['cameraSupv_loss'] = \
                self.cameralossfunc(
                    prediction['camera'],
                    GT['camera']['f_rot'].squeeze(dim = 1).float())
        
        supVisLoss['indexMapSupv_loss'] = \
            self.indMapLossfunc(
                prediction['indexMap'], 
                GT['indexMap'].permute(0,3,1,2))
        
        return supVisLoss
        
class rendering_loss(nn.Module):
    '''
    To make sure that the predicted parameters are realistic and reasonable, 
    the rendering loss is added.
    
    Currently, only the projection loss is implemented.
    '''
    def __init__(self, options):
        super(rendering_loss, self).__init__()
        self.options = options
        self.rendercfg = namedtuple(
            'options', 
            self.options.structureList['rendering'].keys())\
            (**self.options.structureList['rendering'])
        
        self.projLossFunc = \
            (nn.L1Loss(), nn.MSELoss())[self.rendercfg.proj_lossFunc == 'L2']
        self.renderLossFunc = \
            (nn.L1Loss(), nn.MSELoss())[self.rendercfg.render_lossFunc == 'L2']
        
        # prepare data for projection
        _, self.faces, _, _ = read_Obj(options.smpl_objfile_path)
        self.faces = torch.tensor(self.faces)[None]
        self.SMPLmodel = SMPL(options.smpl_model_path, 'cuda')
        self.persCamera= camera()
        self.numSamples= 6890*self.rendercfg.sampleRate
        
    def forward(self, image, predictions, GT):

        batchSize= image.shape[0]
        imageSize= image.shape[2]
        
        SMPLPred = predictions['SMPL']
        dispPred = predictions['disp'].reshape(batchSize, -1, 3)
        cameraPred = predictions['camera']
        
        # convert to axis angle
        jointsRot= rot6d_to_axisAngle(
            SMPLPred[:, :144].view(batchSize, -1, 6))
        predBody = self.SMPLmodel(
            jointsRot.reshape([batchSize, -1]), 
            SMPLPred[:, 144:154], 
            SMPLPred[:, 154:157])
        predPixels, _ = self.persCamera(
            fx = cameraPred[:,0].double(), 
            fy = cameraPred[:,0].double(),  
            cx = torch.ones(batchSize)*imageSize/2, 
            cy = torch.ones(batchSize)*imageSize/2, 
            rotation = rot6d_to_rotmat(cameraPred[:,1:7]).double(), 
            translation = GT['camera']['t'][:,None,:].double(), 
            points= (predBody + dispPred).double(), 
            faces = self.faces.repeat_interleave(batchSize, dim = 0).double(), 
            visibilityOn = False)
        
        GTjoint= rot6d_to_axisAngle(
            GT['SMPL'][:, :144].view(batchSize, -1, 6))
        GTBody = self.SMPLmodel(
            GTjoint.reshape([batchSize, -1]), 
            GT['SMPL'][:, 144:154], 
            GT['SMPL'][:, 154:157])
        GTPixels, _ = self.persCamera(
            fx = GT['camera']['f_rot'][:,0,0], 
            fy = GT['camera']['f_rot'][:,0,0],  
            cx = torch.ones(batchSize)*imageSize/2,
            cy = torch.ones(batchSize)*imageSize/2,
            rotation = \
                rot6d_to_rotmat(GT['camera']['f_rot'][:,0,1:]).double(), 
            translation = GT['camera']['t'][:,None,:].double(),
            points= (GTBody + GT['disp'].reshape(batchSize, -1, 3)).double(), 
            faces = self.faces.repeat_interleave(batchSize, dim = 0).double(), 
            visibilityOn = False)
        
        indice = random.sample(range(6890), int(self.numSamples))
        sampleInd = torch.tensor(indice)
        
        # use textloss after roughly converge    
        renderLoss = {'projLoss': 0,
                      'textLoss': 0}        
        for cnt in range(batchSize):        
            renderLoss['projLoss'] = \
                renderLoss['projLoss'] + \
                    self.projLossFunc(
                        torch.stack(predPixels, dim=0)[:, sampleInd,:],
                        torch.stack(GTPixels  , dim=0)[:, sampleInd,:] )
        return renderLoss
        
class batch_loss(nn.Module):
    '''
    This loss function tries to leverage training samples of the same batch.
    
    If two samples are from the same object, then they should have similar
    dispCode, textCode, smplCode and corresponding predictions. But since we 
    augment input images, the batch_loss_similar does not apply to cameraCode
    and prediction as well as the indexMap prediction.
    '''
    def __init__(self, options):
        super(batch_loss, self).__init__()
        self.options = options
        print('batch loss currently only support L2 loss')
        print('only use batch loss after roughly converging!')
        self.lossfunc = nn.MSELoss()
        
    def forward(self, inputNames, predictions, latentCodes):
        
        batchLoss = {'smplBatchLoss_similarCode': 0,
                     'dispBatchLoss_similarCode': 0,
                     'textBatchLoss_similarCode': 0,
                     'smplBatchLoss_similarPred': 0,
                     'dispBatchLoss_similarPred': 0,
                     'textBatchLoss_similarPred': 0,
                     }
        
        for srcInd in range(len(inputNames)-1):
            for dstInd in range(1, len(inputNames)):
                
                # +1 means similar, -1 means dissimilar
                sign = (-1,+1)[inputNames['srcInd'] == inputNames['dstInd']]
    
                # ---------- SMPL ----------
                batchLoss['smplBatchLoss_similarCode'] = \
                    batchLoss['smplBatchLoss_similarCode'] + \
                        sign*self.lossfunc(
                            latentCodes['SMPL'][srcInd],
                            latentCodes['SMPL'][dstInd])
                
                batchLoss['smplBatchLoss_similarPred'] = \
                    batchLoss['smplBatchLoss_similarPred'] + \
                        sign*self.lossfunc(
                            predictions['SMPL'][srcInd],
                            predictions['SMPL'][dstInd])
                    
                # ---------- disp ----------
                batchLoss['dispBatchLoss_similarCode'] = \
                    batchLoss['dispBatchLoss_similarCode'] + \
                        sign*self.lossfunc(
                            latentCodes['disp'][srcInd],
                            latentCodes['disp'][dstInd])
                
                batchLoss['dispBatchLoss_similarPred'] = \
                    batchLoss['dispBatchLoss_similarPred'] + \
                        sign*self.lossfunc(
                            predictions['disp'][srcInd],
                            predictions['disp'][dstInd])
                        
                # ---------- text ----------
                batchLoss['textBatchLoss_similarCode'] = \
                    batchLoss['textBatchLoss_similarCode'] + \
                        sign*self.lossfunc(
                            latentCodes['text'][srcInd],
                            latentCodes['text'][dstInd])
                
                batchLoss['textBatchLoss_similarPred'] = \
                    batchLoss['textBatchLoss_similarPred'] + \
                        sign*self.lossfunc(
                            predictions['text'][srcInd],
                            predictions['text'][dstInd])    
                
                # ---------- camera ----------
                # not support camera and indexMap yet
        
        return batchLoss
    