#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 23:13:35 2020

@author: zhantao
"""
import random
from os.path import join as pjn

import torch 
import torch.nn as nn

import camera
import SMPL
from utils.vis_util import read_Obj

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
    def __init__(self, options):
        super(latentCode_loss, self).__init__()
        
        self.options = options
            
    def forward(self, imgCode, SMPLCode = None, dispCode = None, 
                textCode = None, cameraCode = None):
        latCodeLoss = {}
        if self.options.structureList.SMPL.enable:
            assert SMPLCode is not None
            if self.options.structureList.SMPL.latent_lossFunc == 'L1':
                lossfunc = nn.L1Loss()
            elif self.options.structureList.SMPL.latent_lossFunc == 'L2':
                lossfunc = nn.L2Loss()
                
            target = imgCode[:,
                             self.options.structureList.SMPL.latent_start:
                             self.options.structureList.SMPL.latent_shape]
            latCodeLoss['smplCode_loss'] = lossfunc(SMPLCode, target)
        
        if self.options.structureList.disp.enable:
            assert dispCode is not None
            if self.options.structureList.disp.latent_lossFunc == 'L1':
                lossfunc = nn.L1Loss()
            elif self.options.structureList.disp.latent_lossFunc == 'L2':
                lossfunc = nn.L2Loss()
                
            target = imgCode[:,
                             self.options.structureList.disp.latent_start:
                             self.options.structureList.disp.latent_shape]
            latCodeLoss['dispCode_loss'] = lossfunc(dispCode, target)
        
        if self.options.structureList.text.enable:
            assert textCode is not None
            if self.options.structureList.text.latent_lossFunc == 'L1':
                lossfunc = nn.L1Loss()
            elif self.options.structureList.text.latent_lossFunc == 'L2':
                lossfunc = nn.L2Loss()
                
            target = imgCode[:,
                             self.options.structureList.text.latent_start:
                             self.options.structureList.text.latent_shape]
            latCodeLoss['textCode_loss'] = lossfunc(textCode, target)
        
        if self.options.structureList.camera.enable:
            assert cameraCode is not None
            if self.options.structureList.camera.latent_lossFunc == 'L1':
                lossfunc = nn.L1Loss()
            elif self.options.structureList.camera.latent_lossFunc == 'L2':
                lossfunc = nn.L2Loss()
                
            target = imgCode[:,
                             self.options.structureList.camera.latent_start:
                             self.options.structureList.camera.latent_shape]
            latCodeLoss['cameraCode_loss'] = lossfunc(cameraCode, target)
        
        return latCodeLoss
        
class supervision_loss(nn.Module):
    '''
    Supervision loss supervises networks to learn the target information 
    explicitly. 
    
    The loss is defined as:
        loss = || gt_x - Dec_x(Enc(img)[i:j,:]) ||, x can be smpl, disp, etc.
        
    '''
    def __init__(self, options):
        super(supervision_loss, self).__init__()
        self.options = options
        
    def forward(self, prediction, imageGT, SMPLGT = None, 
                dispGT = None, textGT = None, cameraGT = None):
        supVisLoss = {}
        if self.options.structureList.SMPL.enable:
            assert SMPLGT is not None
            if self.options.structureList.SMPL.supVis_lossFunc == 'L1':
                lossfunc = nn.L1Loss()
            elif self.options.structureList.SMPL.supVis_lossFunc == 'L2':
                lossfunc = nn.L2Loss()
            elif self.options.structureList.SMPL.supVis_lossFunc == 'cross_entropy':
                raise NotImplementedError()
            supVisLoss['smplSupv_loss'] = lossfunc(prediction['SMPL'], SMPLGT)
        
        if self.options.structureList.disp.enable:
            assert dispGT is not None
            if self.options.structureList.disp.supVis_lossFunc == 'L1':
                lossfunc = nn.L1Loss()
            elif self.options.structureList.disp.supVis_lossFunc == 'L2':
                lossfunc = nn.L2Loss()
            supVisLoss['dispSupv_loss'] = lossfunc(prediction['disp'], dispGT)
        
        if self.options.structureList.text.enable:
            assert textGT is not None
            if self.options.structureList.text.supVis_lossFunc == 'L1':
                lossfunc = nn.L1Loss()
            elif self.options.structureList.text.supVis_lossFunc == 'L2':
                lossfunc = nn.L2Loss()
            supVisLoss['textSupv_loss'] = lossfunc(prediction['text'], textGT)
        
        if self.options.structureList.camera.enable:
            assert cameraGT is not None
            if self.options.structureList.camera.supVis_lossFunc == 'L1':
                lossfunc = nn.L1Loss()
            elif self.options.structureList.camera.supVis_lossFunc == 'L2':
                lossfunc = nn.L2Loss()
            supVisLoss['cameraSupv_loss'] = \
                lossfunc(prediction['camera'], cameraGT)

        if self.options.structureList.indexMap.supVis_lossFunc == 'L1':
            lossfunc = nn.L1Loss()
        elif self.options.structureList.indexMap.supVis_lossFunc == 'L2':
            lossfunc = nn.L2Loss()
        supVisLoss['indexMapSupv_loss'] = \
            lossfunc(prediction['indexMap'], imageGT)
        
        return supVisLoss
        
class rendering_loss(nn.Module):
    '''
    To make sure that the predicted parameters are realistic and reasonable, 
    the rendering loss is added.
    
    Currently, only the indexMap -> color loss is implemented.
    '''
    def __init__(self, options):
        super(self.indexMapSupv_loss, self).__init__()
        self.options = options
        
        if options.structureList.rendering.proj_lossFunc == 'L1':
            self.projLossFunc = nn.L1Loss()
        elif options.structureList.rendering.proj_lossFunc == 'L2':
            self.projLossFunc = nn.L2Loss()
            
        if options.structureList.rendering.render_lossFunc == 'L1':
            self.renderLossFunc = nn.L1Loss()
        elif options.structureList.rendering.render_lossFunc == 'L2':
            self.renderLossFunc = nn.L2Loss()
        
        pathobje = pjn('/'.join(options.smpl_model_path.split('/')[:-1]), 
                       'text_uv_coor_smpl.obj')
        _, self.faces, _, _ = read_Obj(pathobje)
        
        self.faces = torch.tensor(self.faces)[None]
        self.SMPLmodel = SMPL(options.smpl_model_path, 'cuda')
        self.persCamera= camera()
        self.numSamples= 6890*options.structureList.rendering.sampleRate
        
    def forward(self, image, predictions, GT):
        
        renderLoss = {'projLoss': 0,
                      'textLoss': 0}    # use after roughly convering
        batchSize= image.shape[0]
        
        SMPLPred = predictions['SMPL']
        dispPred = predictions['disp']
        # indMapPred = predictions['indexMap']
        cameraPred = predictions['camera']
        
        predBody = self.SMPLmodel(SMPLPred[:, :72], 
                                  SMPLPred[:, 72:82], 
                                  SMPLPred[:, 82:85])
        predPixels, _ = self.persCamera(
            fx=cameraPred[:,0], 
            fy=cameraPred[:,1],  
            cx=cameraPred[:,2], 
            cy=cameraPred[:,3], 
            rotation=cameraPred[:,4:13], 
            translation=cameraPred[:,13:], 
            points=predBody + dispPred, 
            faces=self.faces.repeat_interleave(batchSize, dim = 0), 
            visibilityOn = False)
        
        GTBody = self.SMPLmodel(GT['SMPL'][:, :72], 
                                GT['SMPL'][:, 72:82], 
                                GT['SMPL'][:, 82:85])
        GTPixels, _ = self.persCamera(
            fx=GT['camera'][:,0], 
            fy=GT['camera'][:,1],  
            cx=GT['camera'][:,2], 
            cy=GT['camera'][:,3], 
            rotation=GT['camera'][:,4:13], 
            translation=GT['camera'][:,13:], 
            points=GTBody+ GT['disp'], 
            faces=self.faces.repeat_interleave(batchSize, dim = 0), 
            visibilityOn = False)
        
        indice = random.sample(range(6890), self.numSamples)
        sampleInd = torch.tensor(indice)
        
        for cnt in range(batchSize):        
            renderLoss['projLoss'] = \
                renderLoss['projLoss'] + \
                    self.projLossFunc(
                        predPixels[:, sampleInd],
                        GTPixels[:, sampleInd])
        
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
        self.lossfunc = nn.L2Loss()
        
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